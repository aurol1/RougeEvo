from __future__ import annotations

from .complement import ComplementSearch
from .core import rng_for, write_json
from .demands import consumer_contract, declared_kinds, task_affinity


class DemandComplementSearch(ComplementSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_spent = 0.0
        self.probe_cap = float(self.options['probe_budget_fraction']) * self.budget.limit
        self.training_memory = {}
        self.memory_execution_id = self.best.execution_id
        self.allocations = []
        self.consumer_screens = {}
        self._probe_stage_started = 0.0

    def _kind(self, round_id):
        kinds = self.options['support_kinds']
        cycle = rng_for(self.seed, 'demand_kind_cycle', round_id // len(kinds)).sample(kinds, len(kinds))
        return cycle[round_id % len(kinds)]

    def _context(self, build, evaluation, round_id):
        context = super()._context(build, evaluation, round_id)
        context.fact_kind = self._kind(round_id)
        context.chemistry += '\nShared demand contract (not an experimental result):\n' + consumer_contract(context.fact_kind)
        return context

    def _batches(self, train, round_id):
        if round_id < int(self.search['warmup_rounds']):
            return super()._batches(train, round_id)
        if self.memory_execution_id != self.best.execution_id:
            self.training_memory.clear()
            self.memory_execution_id = self.best.execution_id
        order = rng_for(self.seed, 'complement_batch', round_id).sample(train, len(train))
        kind = self._kind(round_id)


        regression = next((t for t in order if self.training_memory.get(t.id) is True), order[0])
        remaining = [t for t in order if t.id != regression.id]
        remaining.sort(key=lambda t: (self.training_memory.get(t.id) is False,
                                      task_affinity(kind, t.instruction)), reverse=True)
        size = int(self.search['batch_size'])
        panel = remaining[:size - 1]
        panel.insert(1, regression)
        confirm_pool = [t for t in order if t not in panel]
        confirm = confirm_pool[:int(self.search['confirm_size'])]
        if len(panel) != size or len(confirm) != int(self.search['confirm_size']):
            raise ValueError('Not enough training tasks for disjoint demand panel and confirmation')
        self.events.emit('demand_panel', round=round_id, fact_kind=kind,
                         panel_ids=[t.id for t in panel], regression_id=regression.id,
                         confirm_ids=[t.id for t in confirm], source='ordinary_training_only')
        return panel, confirm

    def _remember_parent(self, evaluation):
        if self.memory_execution_id != self.best.execution_id:
            self.training_memory.clear()
            self.memory_execution_id = self.best.execution_id
        allowed = set(self.evaluator.dataset.splits['train'])
        for result in evaluation.results:
            if result.task_id not in allowed:
                raise PermissionError('Demand curriculum cannot use selection or report outcomes')
            self.training_memory[result.task_id] = bool(result.success)

    def _remember_consumer(self, p, evaluation):
        self.consumer_screens[(self.best.execution_id, p.id, tuple(evaluation.task_ids))] = evaluation

    def _weak_pool(self, base, screen, round_id):
        self._probe_stage_started = self.budget.spent
        if round_id < int(self.search['warmup_rounds']):
            return []
        pool = [e for e in self.archive.values() if declared_kinds(e.item)]
        rng_for(self.seed, 'demand_offers', round_id).shuffle(pool)

        pool.sort(key=lambda e: (self._kind(round_id) in declared_kinds(e.item), e.born), reverse=True)
        return [e.item for e in pool[:int(self.options['probe_projects'])]]

    def _probe_room(self, tasks, arms=3):
        current = self.probe_spent + self.budget.spent - self._probe_stage_started
        predicted = self.evaluator.task_cost * len(tasks) * arms
        return current + predicted <= self.probe_cap

    def _projects(self, pool, base, screen, panel, round_id):
        candidates = []
        for p in pool:
            kind = next((k for k in declared_kinds(p) if k in self.options['support_kinds']), None)
            if kind is None:
                continue


            key = (self.best.execution_id, p.id, tuple(t.id for t in screen))
            ep = self.consumer_screens.get(key)
            if ep is None:
                if not self._probe_room(screen, arms=1):
                    self.events.emit('weak_screen_skipped_at_cap', round=round_id, p=p.id)
                    continue
                ep = self.evaluator.evaluate(self.best.with_item(p), screen)
                self._remember_consumer(p, ep)
            if ep.score <= base.score:
                candidates.append((p, kind))
        if not candidates:
            return []
        rng = rng_for(self.seed, 'demand_investment', round_id)
        direct = list(candidates)
        rng.shuffle(direct)
        if self.method in ('complement_direct', 'complement_crossover'):
            return [(p, k, 'demand_matched_direct', '') for p, k in direct[:int(self.options['completion_projects'])]]
        exploratory = (round_id == int(self.search['warmup_rounds']) or
                       (round_id - int(self.search['warmup_rounds']) + 1) % int(self.options['explore_every']) == 0)
        rows = []
        for p, kind in candidates:
            if not self._probe_room(screen):
                continue
            row = self._probe(p, kind, base, screen, round_id, 'screen')
            rows.append((p, kind, row))
        ranked = sorted([row for row in rows if row[2]['eligible']],
                        key=lambda row: (row[2]['priority'], row[2]['assisted_gain']), reverse=True)
        exploring = not ranked and exploratory
        if exploring:
            ranked = list(rows)
            rng.shuffle(ranked)
        picked = []
        for p, kind, row in ranked:
            expanded = panel[:max(len(screen) + 1, min(4, len(panel)))]
            extras = [t for t in expanded if t not in screen]
            if not self._probe_room(extras, arms=4):
                break
            full_base = self.evaluator.evaluate(self.best, expanded)
            full = self._probe(p, kind, full_base, expanded, round_id, 'expanded')
            survives = full['eligible']
            if not survives and not exploring:
                self.events.emit('support_project_rejected', round=round_id, p=p.id,
                                 support_kind=kind, reason='response_did_not_survive_expansion')
                continue
            description = self._paired_support_evidence(p, kind, expanded, full)
            reason = 'demand_specific_response' if survives else 'declared_exploration_quota'
            picked.append((p, kind, reason, description))
            if len(picked) >= int(self.options['completion_projects']):
                break
        if not picked:
            reason = 'probe_cap_direct_completion' if not rows else 'no_confirmed_probe_direct_completion'
            picked = [(p, k, reason, '') for p, k in direct[:int(self.options['completion_projects'])]]
        return picked

    def _record_probe_cost(self, amount):
        self.probe_spent += amount
        row = {'weak_screen_and_probe_spent': self.probe_spent, 'soft_cap': self.probe_cap,
               'round_cost': amount, 'fraction_of_limit': self.probe_spent / self.budget.limit,
               'scope': 'all weak screening and temporary probes, including aborted chains',
               'policy': 'stop initiating probes at estimated cap; ordinary direct completion continues'}
        self.allocations.append(row)
        self.events.emit('probe_allocation', **row)

    def persist(self):
        super().persist()
        write_json(self.output / 'demand_state.json', {
            'search_policy_version': 3, 'probe_spent': self.probe_spent, 'probe_cap': self.probe_cap,
            'training_memory': self.training_memory, 'memory_execution_id': self.memory_execution_id,
            'allocations': self.allocations,
            'note': 'No validation/test outcomes in curriculum; shared representation and editor for controls'})

    def run(self):
        result = super().run()
        result.update(search_policy_version=3, demand_aware=True, probe_spent=self.probe_spent,
                      probe_soft_cap=self.probe_cap,
                      selection_rule=('declared_need_matched_direct_completion' if self.method == 'complement_direct'
                                      else 'matched_specific_response_with_cost_cap_and_direct_fallback'))
        return result
