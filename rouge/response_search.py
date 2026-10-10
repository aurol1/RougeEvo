from __future__ import annotations

from .complement import Choice
from .core import rng_for, seed_for, write_json
from .demand_search import DemandComplementSearch
from .demands import consumer_contract, declared_kinds
from .evolution import Context, focused_evidence, trace_excerpt
from .support import make_source_item


def training_quality(evaluation):

    return sum(r.training_progress['score'] if r.training_progress and
               r.training_progress.get('available') else float(r.success)
               for r in evaluation.results) / len(evaluation.results)


class ResponseComplementSearch(DemandComplementSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel_cache = {}
        self.project_attempts = {}
        self.deferred = {}
        self.completion_spent = 0.0
        self.completion_cap = self.budget.limit * float(self.options['completion_budget_fraction'])
        self.generation_probes = []
        self._active_round = 0

    def _kind(self, round_id):
        warm = int(self.search['warmup_rounds'])
        block = round_id if round_id < warm else warm + (round_id - warm) // int(self.search['panel_reuse_rounds'])
        return super()._kind(block)

    def _batches(self, train, round_id):
        self._active_round = round_id
        warm = int(self.search['warmup_rounds'])
        if round_id < warm:
            return super()._batches(train, round_id)
        block = (round_id - warm) // int(self.search['panel_reuse_rounds'])
        key = (self.best.execution_id, block)
        if key not in self.panel_cache:


            if self.memory_execution_id != self.best.execution_id:
                self.training_memory.clear()
                self.memory_execution_id = self.best.execution_id
            order = rng_for(self.seed, 'response_panel', self.best.execution_id, block).sample(train, len(train))
            regression = next((t for t in order if self.training_memory.get(t.id) is True), order[0])
            remaining = [t for t in order if t.id != regression.id]

            remaining.sort(key=lambda t: self.training_memory.get(t.id) is False, reverse=True)
            size = int(self.search['batch_size'])
            panel = remaining[:size - 1]
            panel.insert(1, regression)
            confirm = [t for t in order if t not in panel][:int(self.search['confirm_size'])]
            if len(panel) != size or len(confirm) != int(self.search['confirm_size']):
                raise ValueError('Not enough disjoint training tasks')
            self.panel_cache[key] = (panel, confirm)
        panel, confirm = self.panel_cache[key]
        self.events.emit('response_panel', round=round_id, block=block, fact_kind=self._kind(round_id),
                         panel_ids=[t.id for t in panel], confirm_ids=[t.id for t in confirm],
                         source='ordinary_training_only')
        return panel, confirm

    def _context(self, build, evaluation, round_id):
        if any(t not in self.evaluator.dataset.splits['train'] for t in evaluation.task_ids):
            raise PermissionError('Refinement context is training only')
        context = Context(build, focused_evidence(self.tasks, evaluation,
                          rng_for(self.seed, 'focused_evidence', round_id),
                          int(self.config['proposal']['evidence_chars'])))
        if round_id >= int(self.search['warmup_rounds']):
            context.fact_kind = self._kind(round_id)
            context.chemistry = consumer_contract(context.fact_kind)
        return context

    def _generation_context(self, build, evaluation, panel, round_id):
        normal = self.evaluator.evaluate(build, panel)
        context = self._context(build, normal, round_id)
        if round_id < int(self.search['warmup_rounds']) or self.method == 'complement_direct':
            return context
        screen = panel[:int(self.search['screen_size'])]
        self._probe_stage_started = self.budget.spent
        if not self._response_room([(build, context.fact_kind)], screen):
            self.events.emit('generation_probe_skipped', round=round_id, reason='probe_soft_cap')
            return context
        before = self.budget.spent
        try:
            assisted = self.evaluator.evaluate_supported(build, screen, context.fact_kind)
            plain = self.evaluator.evaluate(build, screen)
            self.generation_probes.append({'round': round_id, 'kind': context.fact_kind,
                'task_ids': plain.task_ids, 'hard_scores': [plain.score, assisted.score],
                'cell_scores': [training_quality(plain), training_quality(assisted)],
                'spent': self.budget.spent - before})
            blocks = ['Actual paid H/Hz training contrast; this is not a deployment result.']
            for index, task in enumerate(screen, 1):
                a, b = plain.by_task()[task.id], assisted.by_task()[task.id]
                facts = self.evaluator.support_evidence(build, [task], context.fact_kind, limit_chars=900)
                blocks.append(f'Example {index}: {task.instruction[:350]}\n'
                    f'H success={a.success}; Hz success={b.success}\n'
                    f'H procedure: {trace_excerpt(a.trace, 650)}\n'
                    f'Hz procedure: {trace_excerpt(b.trace, 650)}\nInput-only facts: {facts}')
            context.chemistry += '\n\n' + '\n\n'.join(blocks)[:int(self.config['proposal']['evidence_chars'])]
        finally:
            self._record_probe_cost(self.budget.spent - before)
        return context

    def _response_room(self, arms, tasks):

        fresh = 0
        for build, kind in arms:
            for task in tasks:
                if hasattr(self.evaluator, '_key'):
                    key = self.evaluator._key(build, task, kind)
                else:
                    key = (build.execution_id, task.id, kind)
                fresh += key not in self.evaluator.cache
        current = self.probe_spent + self.budget.spent - self._probe_stage_started
        return fresh == 0 or current + fresh * self.evaluator.task_cost <= self.probe_cap

    def _paired_support_evidence(self, p, kind, panel, full):
        hp = self.best.with_item(p)
        plain = self.evaluator.evaluate(hp, panel).by_task()
        assisted = self.evaluator.evaluate_supported(hp, panel, kind).by_task()
        def quality(result):
            return result.training_progress['score'] if result.training_progress and result.training_progress.get('available') else float(result.success)
        order = sorted(panel, key=lambda t: quality(assisted[t.id]) - quality(plain[t.id]), reverse=True)
        limit = int(self.config['proposal']['evidence_chars'])
        summary = (f'Paid {kind} intervention; hard scores={full["scores"]}; '
                   f'training cell scores={full["cell_scores"]}; cell interaction={full["cell_interaction"]:.4f}. '
                   'Training search signal only; target values are not supplied.\n')
        each = max(1, (limit - len(summary)) // min(3, len(order)))
        blocks = []
        for index, task in enumerate(order[:3], 1):
            a, b = plain[task.id], assisted[task.id]
            facts = self.evaluator.support_evidence(hp, [task], kind, limit_chars=each // 3)
            block = (f'Example {index}: {task.instruction[:350]}\n'
                     f'Hp pass={a.success}, credit={quality(a):.4f}; Hpz pass={b.success}, credit={quality(b):.4f}\n'
                     f'Hp procedure: {trace_excerpt(a.trace, each // 4)}\n'
                     f'Hpz procedure: {trace_excerpt(b.trace, each // 4)}\nInput-only facts: {facts}')
            blocks.append(block[:each])
        return (summary + '\n\n'.join(blocks))[:limit]

    def _draft_consumer(self, slot, context, seed):
        return self.proposer.draft_refinement(slot, context, seed,
                                             max_chars=int(self.options['refinement_chars']))

    def _consumer_count(self, warmup):
        return int(self.search['candidates']) if warmup else int(self.options['post_warmup_candidates'])

    def _consumer_slot(self, round_id, index, warmup):
        if warmup:
            return super()._consumer_slot(round_id, index, warmup)
        offers = rng_for(self.seed, 'response_slot_offers', round_id).sample(self.slots, len(self.slots))
        return offers[index % len(offers)]

    def _can_start_round(self, round_id):
        minimum = (int(self.search['batch_size']) + int(self.search['confirm_size']) +
                   int(self.search['validation_limit'])) * self.evaluator.task_cost
        room = self.budget.limit - self.budget.spent - self.budget.reserved
        if room < minimum:
            self.events.emit('round_reserve_stop', round=round_id, remaining=room,
                             estimated_minimum_chain=minimum, rule='estimate_not_a_hard_cost_guarantee')
            return False
        return True

    def _order(self, choices, round_id):
        rows = super()._order(choices, round_id)

        rows.sort(key=lambda c: (c.evaluation.score, training_quality(c.evaluation),
                                 -self._changes(c.build)), reverse=True)
        return rows

    def _promising(self, evaluation, parent):
        return (evaluation.score > parent.score or (evaluation.score == parent.score and
                training_quality(evaluation) > training_quality(parent) + 1e-9))

    def _selection_candidates(self):
        return int(self.options['selection_candidates'])

    def _probe(self, p, kind, base, panel, round_id, stage):
        row = super()._probe(p, kind, base, panel, round_id, stage)
        hp = self.best.with_item(p)
        ep = self.evaluator.evaluate(hp, panel)
        ez = self.evaluator.evaluate_supported(self.best, panel, kind)
        epz = self.evaluator.evaluate_supported(hp, panel, kind)
        h, a, b, ab = [training_quality(e) for e in (base, ep, ez, epz)]
        dense_interaction, dense_gain = ab - a - b + h, ab - h
        row.update(cell_scores={'H': h, 'Hp': a, 'Hz': b, 'Hpz': ab},
                   cell_interaction=dense_interaction, cell_assisted_gain=dense_gain,
                   hard_eligible=row['eligible'])
        dense_eligible = (epz.score >= base.score and dense_gain > 1e-9 and
                          dense_interaction > float(self.options['min_interaction']) + 1e-9)
        row.update(eligible=row['eligible'] or dense_eligible,
                   priority=max(row['interaction'], dense_interaction) /
                            max(1.0, self.budget.tool_price * len(panel)),
                   eligibility_source='task_pass' if row['hard_eligible'] else
                                      'training_cell_credit' if dense_eligible else 'none')
        self.events.emit('response_probe_credit', **row)
        return row

    def _normal_pair(self, p, q, base, tasks, round_id, tag):
        choices = super()._normal_pair(p, q, base, tasks, round_id, tag)
        h, a, b, ab = [training_quality(e) for e in [base] + [c.evaluation for c in choices]]
        self.permanent_pairs[-1].update(cell_scores={'H': h, 'Hp': a, 'Hq': b, 'Hpq': ab},
            cell_interaction=ab - a - b + h, pair_cell_gain=ab - h)
        self.events.emit('permanent_pair_credit', **self.permanent_pairs[-1])
        return choices

    def _project_key(self, p, kind):
        return (self.best.execution_id, p.id, kind)

    def _projects(self, pool, base, screen, panel, round_id):
        if self.completion_spent + self.evaluator.task_cost * len(panel) * 3 > self.completion_cap:
            self.events.emit('completion_skipped', round=round_id, reason='completion_soft_cap')
            return []
        pending = [(key, row) for key, row in self.deferred.items() if key[0] == self.best.execution_id]
        if pending:
            _, row = pending[0]
            return [(row['p'], row['kind'], 'source_rejected_then_bounded_synthesis', row['support'])]
        candidates = []
        for p in pool:
            kinds = [k for k in declared_kinds(p) if k in self.options['support_kinds']]
            if not kinds:
                continue
            kind = kinds[0]
            if self.project_attempts.get(self._project_key(p, kind), 0) >= int(self.options['max_project_attempts']):
                continue
            hp = self.best.with_item(p)
            if not self._response_room([(hp, None)], screen):
                continue
            ep = self.evaluator.evaluate(hp, screen)
            self._remember_consumer(p, ep)

            if ep.score <= base.score:
                candidates.append((p, kind))
        if not candidates:
            return []
        rng = rng_for(self.seed, 'response_investment', round_id)
        rng.shuffle(candidates)
        if self.method == 'complement_direct':
            return [(p, k, 'bounded_direct', '') for p, k in candidates[:int(self.options['completion_projects'])]]
        warm = int(self.search['warmup_rounds'])
        exploring = round_id == warm or (round_id - warm + 1) % int(self.options['explore_every']) == 0
        rows = []
        for p, kind in candidates:
            hp = self.best.with_item(p)
            if not self._response_room([(hp, None), (self.best, kind), (hp, kind)], screen):
                break
            row = self._probe(p, kind, base, screen, round_id, 'screen')
            rows.append((p, kind, row))
        ranked = sorted([r for r in rows if r[2]['eligible']],
                        key=lambda r: (r[2]['priority'], r[2]['assisted_gain']), reverse=True)
        exploratory = not ranked and exploring
        if exploratory:
            ranked = list(rows)
            rng.shuffle(ranked)
        for p, kind, row in ranked:
            expanded = panel[:min(4, len(panel))]
            extra = [t for t in expanded if t not in screen]
            hp = self.best.with_item(p)
            if not self._response_room([(self.best, None), (hp, None), (self.best, kind), (hp, kind)], extra):
                break
            full_base = self.evaluator.evaluate(self.best, expanded)
            full = self._probe(p, kind, full_base, expanded, round_id, 'expanded')
            if not full['eligible'] and not exploratory:
                continue
            reason = 'cell_or_task_specific_response' if full['eligible'] else 'bounded_exploration'
            return [(p, kind, reason, self._paired_support_evidence(p, kind, expanded, full))]
        if exploring:

            p, kind = candidates[0]
            return [(p, kind, 'unassisted_exploration_at_cap_or_no_response', '')]
        self.events.emit('completion_skipped', round=round_id, reason='no_response_outside_exploration')
        return []

    def _complete_project(self, p, kind, support, base, screen, panel, round_id):
        before = self.budget.spent
        key = self._project_key(p, kind)
        try:
            pending = self.deferred.pop(key, None)
            if pending is None:
                self.project_attempts[key] = self.project_attempts.get(key, 0) + 1
                full_base = self.evaluator.evaluate(self.best, panel)
                choices = self._normal_pair(p, make_source_item(kind), full_base, panel, round_id, 'source_adoption')
                if any(self._promising(c.evaluation, full_base) for c in choices):


                    self.deferred[key] = {'p': p, 'kind': kind, 'support': support, 'born': round_id}
                    self.events.emit('synthesis_deferred', round=round_id, p=p.id, support_kind=kind,
                                     reason='test_existing_source_before_paying_for_synthesis')
                    return [Choice(c.build, self.evaluator.evaluate(c.build, screen), c.tag, c.p, c.q)
                            for c in choices]
            else:
                choices = []
            normal = self.evaluator.evaluate(self.best.with_item(p), panel)
            context = self._context(self.best, normal, round_id)
            for attempt in range(int(self.options['q_attempts'])):
                q = self.proposer.draft_producer(context, p,
                    seed_for(self.seed, 'response_producer', round_id, p.id, attempt), support_evidence=support)
                if q is not None:
                    try:
                        choices += self._normal_pair(p, q, base, screen, round_id, 'synthesized')
                    except ValueError as exc:
                        self.events.emit('invalid_permanent_completion', round=round_id, p=p.id,
                                         q=q.id, error=str(exc)[:500])
            return [Choice(c.build, self.evaluator.evaluate(c.build, screen), c.tag, c.p, c.q)
                    for c in choices]
        finally:
            self.completion_spent += self.budget.spent - before

    def persist(self):
        super().persist()
        write_json(self.output / 'demand_state.json', {
            'search_policy_version': 4, 'probe_spent': self.probe_spent, 'probe_cap': self.probe_cap,
            'training_memory': self.training_memory, 'memory_execution_id': self.memory_execution_id,
            'allocations': self.allocations,
            'note': 'Includes paid pre-generation Hz; train-cell credit never replaces held-out task-pass gate'})
        write_json(self.output / 'response_state.json', {
            'search_policy_version': 4, 'probe_spent': self.probe_spent, 'probe_cap': self.probe_cap,
            'completion_spent': self.completion_spent, 'completion_cap': self.completion_cap,
            'generation_probes': self.generation_probes,
            'project_attempts': [{'key': list(k), 'count': v} for k, v in self.project_attempts.items()],
            'deferred': [{'key': list(k), 'p': v['p'].to_dict(), 'kind': v['kind'], 'born': v['born']}
                         for k, v in self.deferred.items()],
            'credit_scope': 'training_only; full ordinary held-out task-pass gain required for acceptance',
            'probe_ledger_scope': 'ordinary weak screening plus temporary pre-generation and paired probes'})

    def run(self):
        result = super().run()
        result.update(search_policy_version=4, response_guided=True,
            selection_rule='training_cell_tie_promotion_then_strict_ordinary_task_pass_selection',
            completion_spent=self.completion_spent, completion_soft_cap=self.completion_cap,
            generation_probe_count=len(self.generation_probes),
            accepted_types=[r['selection_attempts'][-1]['tag'] for r in self.history if r['accepted']])
        return result
