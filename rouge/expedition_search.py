from __future__ import annotations

from .complement import Choice
from .core import BudgetExceeded, paired_delta, rng_for, seed_for, write_json
from .demands import declared_kinds
from .frontier_search import FrontierComplementSearch
from .racing import strict_gate
from .support import make_source_item


class ExpeditionComplementSearch(FrontierComplementSearch):
    policy_version = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.confirmed_candidates = {}
        self.producer_exploration_attempts = 0
        self.producer_exploration_spent = 0.
        self.producer_exploration_cap = self.budget.limit * float(
            self.options['producer_exploration_budget_fraction'])
        self._chain_hold = 0.
        self._lean_round = False
        self._bounded_projects = set()

    def _admit(self, candidate, base, confirm, round_id):
        admitted = super()._admit(candidate, base, confirm, round_id)
        if admitted:
            self.confirmed_candidates[candidate.build.execution_id] = [t.id for t in confirm]
            self.persist()
        return admitted

    def _training_audit(self, row):
        ids = self.confirmed_candidates.get(row.build.execution_id)
        if not ids or self._anchor is None:
            return False, {'reason': 'no_disjoint_training_confirmation'}
        tasks = list(self._anchor) + [self.tasks[t] for t in ids]
        if len({t.id for t in tasks}) != len(tasks) or any(
                t.id not in self.evaluator.dataset.splits['train'] for t in tasks):
            raise PermissionError('Neutral transfer audit needs disjoint training tasks only')
        arms = [(self.champion, None), (row.build, None)]
        if not self._room(arms, tasks):
            return False, {'reason': 'insufficient_paired_training_audit_budget'}
        parent = self.evaluator.evaluate(self.champion, tasks)
        child = self.evaluator.evaluate(row.build, tasks)
        delta = paired_delta(parent, child)
        gain = self._utility(child) - self._utility(parent)
        passed = not delta.regressed and child.score >= parent.score and gain >= float(
            self.options['frontier_min_gain'])
        return passed, {'reason': 'confirmed_training_improvement' if passed else 'training_audit_failed',
            'task_ids': parent.task_ids, 'parent_score': parent.score, 'candidate_score': child.score,
            'utility_gain': gain, 'fixed': list(delta.fixed), 'regressed': list(delta.regressed),
            'scope': 'training_only_not_heldout_gain'}

    def _reserve_selection(self):
        self._release_selection()
        count = int(self.options['selection_reserve_candidates'])
        pending = [r for r in self.frontier.values() if r.build.execution_id not in self.tested_selection]
        pending.sort(key=lambda r: (self._utility(r.evaluation), r.born), reverse=True)
        validation = self.evaluator.dataset.split('validation', self.search['validation_limit'])
        needs = []
        for row in pending[:count]:
            confirm = [self.tasks[t] for t in self.confirmed_candidates.get(row.build.execution_id, [])]
            audit = list(self._anchor or []) + confirm
            needs.append(self._fresh([(row.build, None)], validation) +
                         self._fresh([(row.build, None), (self.champion, None)], audit))


        future = len(validation) + int(self.search['confirm_size'])
        fresh = sum(needs) if needs else future
        amount = max(0., min(self.selection_cap - self.selection_spent,
                            1.6 * self.evaluator.task_cost * fresh, self.budget.remaining))
        if amount:
            self.selection_hold = self.budget.reserve(amount)
        self.events.emit('selection_reserve', amount=amount, pending=len(pending),
            protected_candidates=count, estimated_fresh_tasks=fresh, scope='complete_heldout_and_paired_train_audit')

    def _check_selection(self, validation, round_id, final=False):
        self._release_selection()
        before, accepted = self.budget.spent, 0
        try:
            rows = [r for r in self.frontier.values() if r.build.execution_id not in self.tested_selection]
            rows.sort(key=lambda r: (r.evaluation.score, self._utility(r.evaluation), r.born), reverse=True)

            rows = rows[:int(self.options['selection_candidates'])]
            for row in rows:
                if self.selection_spent + self.budget.spent - before >= self.selection_cap:
                    self.events.emit('selection_deferred', round=round_id, final=final, reason='selection_allocation_spent')
                    break
                if not self._room([(row.build, None)], validation):
                    self.events.emit('selection_deferred', round=round_id, final=final, reason='insufficient_finishing_reserve')
                    break
                previous = self.champion_validation
                checked, gate = strict_gate(self.evaluator, row.build, validation, previous, 0.,
                    int(self.search['validation_chunk_size']), self.events, allow_ties=True)
                improved = bool(checked and checked.score > previous.score)
                tied = bool(checked and abs(checked.score - previous.score) <= 1e-12)
                neutral, audit = False, None
                regression = list(paired_delta(previous, checked).regressed) if checked else []
                if tied and not regression:
                    neutral, audit = self._training_audit(row)
                gained = improved or neutral
                reason = ('heldout_task_pass_gain' if improved else 'paired_nonregressing_training_tie' if neutral
                          else 'heldout_regression' if regression else 'no_qualified_gain_or_tie')
                self.tested_selection.add(row.build.execution_id)
                item = {'round': round_id, 'build': row.build.id, 'tag': row.tag, 'final': final,
                    'gate': gate, 'selection_score': checked.score if checked else None,
                    'previous_selection_score': previous.score, 'accepted': gained,
                    'validation_improved': improved, 'neutral_transfer': neutral, 'reason': reason,
                    'selection_regressed': regression, 'training_audit': audit}
                self.selection_checks.append(item)
                self.events.emit('frontier_selection', **item)
                if gained:
                    self.champion, self.champion_validation = row.build, checked
                    self.best_validation = checked
                    accepted += 1
                    self.persist()
        finally:
            self.selection_spent += self.budget.spent - before
            if not final:
                self._reserve_selection()
        return accepted

    def _combination_choices(self, base, screen, panel, round_id):
        if self._lean_round or round_id < int(self.search['warmup_rounds']):
            return []
        offers, vector = [], self._vector(base)
        for entry in self.frontier.values():
            expected = sum(max(0., b - a) for a, b in zip(vector, self._vector(entry.evaluation)))
            for item in entry.build.execution_items:
                if item.slot not in self.slots or item == self.start.get(item.key) or item == self.best.get(item.key):
                    continue
                build = self.best.with_item(item)
                if build.execution_id in self.frontier:
                    continue
                offers.append((expected, entry.born, item, build))
        rng_for(self.seed, 'productive_composition', round_id).shuffle(offers)
        offers.sort(key=lambda row: (row[0], row[1]), reverse=True)
        choices, seen = [], set()
        for expected, _, item, build in offers:
            if build.execution_id in seen:
                continue
            seen.add(build.execution_id)
            partners = [x for x in self.best.execution_items if x.slot in self.slots and x.key != item.key
                        and x != self.start.get(x.key)]
            audit = None
            if partners:
                partner = rng_for(self.seed, 'productive_audit_partner', round_id, item.id).choice(partners)
                common = self.best.with_item(self.start.get(item.key)).with_item(self.start.get(partner.key))
                if (common.execution_id, item.id, partner.id) in self.combination_rejections:
                    continue
                audit = common, item, partner
            if not self._room([(build, None)], screen):
                break
            if audit:
                self.composition_audits[build.execution_id] = audit
            result = self.evaluator.evaluate(build, screen)
            choices.append(Choice(build, result, 'frontier_composition', item))
            self.events.emit('composition_offer', round=round_id, parent=self.best.id, build=build.id,
                item=item.id, anchor_coverage_heuristic=expected, new_execution=True,
                note='mutated donor component; actual scores decide admission')
            if len(choices) >= int(self.options['combination_offers']):
                break
        return choices

    def _exploration_room(self, screen):
        minimum = 1.6 * self.evaluator.task_cost * (2 * len(screen) + 1)
        return (self.producer_exploration_attempts < int(self.options['producer_exploration_attempts'])
            and self.producer_exploration_spent + minimum <= self.producer_exploration_cap
            and self.completion_spent + minimum <= self.completion_cap
            and self._room([], [], extra=minimum))

    def _projects(self, pool, base, screen, panel, round_id):
        if self._lean_round:
            return []
        if self._exploration_room(screen):
            offers = []
            for p in pool:
                kind = next((k for k in declared_kinds(p) if k in self.options['support_kinds']), None)
                if kind is None or self.project_attempts.get(self._project_key(p, kind), 0) >= int(
                        self.options['max_project_attempts']):
                    continue
                hp = self.best.with_item(p)
                if self._room([(hp, None)], screen):
                    ep = self.evaluator.evaluate(hp, screen)
                    offers.append((self._utility(ep), p, kind))
            offers.sort(key=lambda r: r[0], reverse=True)
            if offers:
                _, p, kind = offers[0]
                arms = [(self.best.with_item(p), None), (self.best, kind), (self.best.with_item(p), kind)]
                support = ''
                if self._room(arms, screen) and self._response_room(arms, screen):
                    row = self._probe(p, kind, base, screen, round_id, 'screen')
                    support = self._paired_support_evidence(p, kind, screen, row)
                self._bounded_projects.add((round_id, self._project_key(p, kind)))
                return [(p, kind, 'bounded_producer_exploration', support)]
        return super()._projects(pool, base, screen, panel, round_id)

    def _complete_project(self, p, kind, support, base, screen, panel, round_id):
        key = self._project_key(p, kind)
        if (round_id, key) not in self._bounded_projects:
            return super()._complete_project(p, kind, support, base, screen, panel, round_id)
        before, choices, hold = self.budget.spent, [], 0.
        try:
            if not self._exploration_room(screen):
                return []
            allowance = min(self.producer_exploration_cap - self.producer_exploration_spent,
                            self.completion_cap - self.completion_spent, self.budget.remaining)

            hold = self.budget.reserve(max(0., self.budget.remaining - allowance))
            self.producer_exploration_attempts += 1
            self.project_attempts[key] = self.project_attempts.get(key, 0) + 1
            context = self._context(self.best, self.evaluator.evaluate(self.best.with_item(p), screen), round_id)
            learned = self.proposer.draft_producer(context, p,
                seed_for(self.seed, 'expedition_producer', round_id, p.id), support_evidence=support)
            if learned is not None:
                learned = self._prepare_q(learned)
                arms = [(self.best.with_item(learned), None), (self.best.compose(p, learned), None)]
                if self._room(arms, screen):
                    choices += self._normal_pair(p, learned, base, screen, round_id, 'synthesized_exploration')
            q = self._prepare_q(make_source_item(kind))
            if self._room([(self.best.with_item(q), None), (self.best.compose(p, q), None)], screen):
                choices += self._normal_pair(p, q, base, screen, round_id, 'source_adoption')
            return choices
        except BudgetExceeded as exc:
            self.events.emit('producer_exploration_aborted', round=round_id, p=p.id, reason=str(exc),
                completed_choices=len(choices), evidence_status='paid_partial_project_not_promoted')
            return choices
        finally:
            self.budget.release(hold)
            cost = self.budget.spent - before
            self.producer_exploration_spent += cost
            self.completion_spent += cost
            self.events.emit('producer_exploration_cost', round=round_id, attempts=self.producer_exploration_attempts,
                project_spent=cost, total_spent=self.producer_exploration_spent, cap=self.producer_exploration_cap)

    def _consumer_count(self, warmup):
        return 1 if self._lean_round else super()._consumer_count(warmup)

    def _generation_context(self, build, evaluation, panel, round_id):
        screen = panel[:int(self.search['screen_size'])]
        kind = self._kind(round_id)
        if not self._room([(build, None), (build, kind)], screen):
            self.events.emit('generation_probe_deferred', round=round_id,
                reason='protect_complete_training_chain', fact_kind=kind)
            return self._context(build, evaluation, round_id)
        return super()._generation_context(build, evaluation, panel, round_id)

    def _draft_consumer(self, slot, context, seed):
        try:
            return super()._draft_consumer(slot, context, seed)
        except BudgetExceeded as exc:

            if not self._chain_hold:
                raise
            self.events.emit('consumer_proposal_deferred', slot=slot, reason=str(exc),
                protected_chain=self._chain_hold)
            return None

    def _reserve_training_chain(self, panel, confirm, screen, round_id):
        self._release_training_chain()


        fresh_parent = self._fresh([(self.best, None)], confirm)
        fresh = len(panel) - len(screen) + len(confirm) + fresh_parent
        amount = 1.6 * self.evaluator.task_cost * fresh
        if amount <= self.budget.remaining:
            self._chain_hold = self.budget.reserve(amount)
        self.events.emit('training_chain_reserve', round=round_id, amount=self._chain_hold,
            fresh_tasks=fresh, protects='one_full_ordinary_anchor_and_disjoint_confirmation')

    def _release_training_chain(self):
        self.budget.release(self._chain_hold)
        self._chain_hold = 0.

    def _can_start_round(self, round_id):
        self._lean_round = False
        warm = round_id < int(self.search['warmup_rounds'])
        offers = super()._consumer_count(warm)
        per = 1.6 * self.evaluator.task_cost
        chain = int(self.search['batch_size']) - int(self.search['screen_size']) + 2 * int(self.search['confirm_size']) + 1
        regular = per * (chain + int(self.search['screen_size']) * offers)
        lean = per * (chain + int(self.search['screen_size']))
        if self.budget.remaining >= regular:
            return True
        if self.budget.remaining >= lean:
            self._lean_round = True
            self.events.emit('round_offer_reduced', round=round_id, remaining=self.budget.remaining,
                estimated_chain=lean, reason='retain_complete_candidate_chain_with_one_ordinary_offer')
            return True
        self.events.emit('round_reserve_stop', round=round_id, remaining=self.budget.remaining,
            selection_reserved=self.selection_hold, estimated_minimum_chain=lean)
        return False

    def _shortlist(self, choices, round_id):
        ordered = self._order(choices, round_id)


        if (round_id >= int(self.search['warmup_rounds']) and
                (round_id - int(self.search['warmup_rounds'])) % int(self.options['explore_every']) == 0):
            eligible = []
            for choice in ordered:
                if choice.tag == 'frontier_composition' or ':pq' in choice.tag or ':q' in choice.tag:
                    tasks = [self.tasks[t] for t in choice.evaluation.task_ids]
                    base = self.evaluator.evaluate(self.best, tasks)
                    if (not paired_delta(base, choice.evaluation).regressed and
                            self._utility(choice.evaluation) + float(self.options['frontier_min_gain']) >= self._utility(base)):
                        eligible.append(choice)
            if eligible:
                picked = rng_for(self.seed, 'expedition_full_chain_offer', round_id).choice(eligible)
                ordered = [picked] + [c for c in ordered if c is not picked]
                self.events.emit('full_chain_exploration_offer', round=round_id, build=picked.build.id, tag=picked.tag,
                    note='training test opportunity, not forced admission or interaction evidence')


        return ordered

    def persist(self):
        super().persist()
        write_json(self.output / 'expedition_state.json', {
            'search_policy_version': self.policy_version, 'confirmed_candidates': self.confirmed_candidates,
            'producer_exploration_attempts': self.producer_exploration_attempts,
            'producer_exploration_spent': self.producer_exploration_spent,
            'producer_exploration_cap': self.producer_exploration_cap,
            'training_chain_reserved': self._chain_hold, 'lean_round': self._lean_round,
            'neutral_transfer_count': sum(r.get('neutral_transfer', False) for r in self.selection_checks),
            'heldout_improvement_count': sum(r.get('validation_improved', False) for r in self.selection_checks)})

    def run(self):
        result = super().run()
        result.update(validation_improved_rounds=sum(r.get('validation_improved', False) for r in self.selection_checks),
            neutral_transfer_rounds=sum(r.get('neutral_transfer', False) for r in self.selection_checks),
            producer_exploration_attempts=self.producer_exploration_attempts,
            producer_exploration_spent=self.producer_exploration_spent,
            selection_rule='full_ordinary_heldout_pass_gain_or_paired_nonregressing_confirmed_training_tie',
            reward_scope='neutral transfers change deployment but do not count as heldout improvement')
        return result
