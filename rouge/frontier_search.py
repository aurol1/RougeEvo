from __future__ import annotations

from dataclasses import dataclass
import time

from .complement import Choice, WeakItem
from .core import BudgetExceeded, interaction, paired_delta, rng_for, seed_for, write_json
from .demands import consumer_contract, declared_kinds
from .evolution import Context, focused_evidence
from .items import Build
from .core import Evaluation
from .racing import strict_gate
from .response_search import ResponseComplementSearch, training_quality
from .support import make_source_item


@dataclass
class FrontierEntry:
    build: Build
    evaluation: Evaluation
    born: int
    tag: str
    visits: int = 0


class FrontierComplementSearch(ResponseComplementSearch):
    policy_version = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.champion = self.start
        self.champion_validation = None
        self.frontier = {}
        self.frontier_updates = []
        self.selection_checks = []
        self.tested_selection = set()
        self.selection_spent = 0.0
        self.selection_cap = self.budget.limit * float(self.options['selection_budget_fraction'])
        self.selection_hold = 0.0
        self._anchor = None
        self._confirm_pool = None
        self._responses = {}
        self.action_credit = {}
        self.kind_offers = {}
        self.composition_audits = {}
        self.combination_rejections = set()

    def _utility(self, evaluation):
        return evaluation.score + float(self.options['frontier_quality_weight']) * training_quality(evaluation)

    def _order(self, choices, round_id):


        rows = list({c.build.execution_id: c for c in choices
                     if c.build.execution_id != self.best.execution_id}.values())
        rng_for(self.seed, 'frontier_candidate_ties', round_id).shuffle(rows)
        rows.sort(key=lambda c: (self._utility(c.evaluation), c.evaluation.score,
                                 -self._changes(c.build)), reverse=True)
        return rows

    def _vector(self, evaluation):
        weight = float(self.options['frontier_quality_weight'])
        return tuple(float(r.success) + weight * (r.training_progress['score']
            if r.training_progress and r.training_progress.get('available') else float(r.success))
            for r in evaluation.results)

    def _batches(self, train, round_id):
        if self._anchor is None:

            order = rng_for(self.seed, 'frontier_anchor').sample(train, len(train))
            groups = {}
            for task in order:
                groups.setdefault(getattr(task, 'instruction_type', ''), []).append(task)
            anchor = []
            while len(anchor) < int(self.search['batch_size']):
                for group in groups.values():
                    if group and len(anchor) < int(self.search['batch_size']):
                        anchor.append(group.pop(0))
                if not any(groups.values()) and len(anchor) < int(self.search['batch_size']):
                    raise ValueError('Not enough training tasks for frontier anchor')
            self._anchor = anchor
            self._confirm_pool = [t for t in order if t not in anchor]
        size = int(self.search['confirm_size'])
        if len(self._confirm_pool) < size:
            raise ValueError('Not enough disjoint training confirmation tasks')
        block = round_id // int(self.search['panel_reuse_rounds'])
        offset = block * size % len(self._confirm_pool)
        confirm = [self._confirm_pool[(offset + i) % len(self._confirm_pool)] for i in range(size)]
        self.events.emit('frontier_panel', round=round_id, block=block,
            anchor_ids=[t.id for t in self._anchor], confirm_ids=[t.id for t in confirm],
            source='training_only_fixed_type_diverse_anchor_and_rotating_confirmation')
        return self._anchor, confirm

    def _context(self, build, evaluation, round_id):
        if any(t not in self.evaluator.dataset.splits['train'] for t in evaluation.task_ids):
            raise PermissionError('Revision context is training only')
        failed = [r for r in evaluation.results if not r.success]
        failed.sort(key=lambda r: (bool(r.diagnostic and ('Error' in r.diagnostic or 'Traceback' in r.trace)),
            r.training_progress.get('score', 0) if r.training_progress and
            r.training_progress.get('available') else 0), reverse=True)

        chosen, kinds = [], set()
        for row in failed:
            kind = getattr(self.tasks[row.task_id], 'instruction_type', '')
            if not chosen or kind not in kinds:
                chosen.append(row)
                kinds.add(kind)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            chosen += [r for r in failed if r not in chosen][:2 - len(chosen)]
        chosen += [r for r in evaluation.results if r.success][:1]
        view = type(evaluation)(evaluation.build_id, chosen or evaluation.results)
        context = Context(build, focused_evidence(self.tasks, view,
            rng_for(self.seed, 'frontier_evidence', round_id), int(self.config['proposal']['evidence_chars'])))
        if round_id >= int(self.search['warmup_rounds']):
            context.fact_kind = self._kind(round_id)
            context.chemistry = consumer_contract(context.fact_kind)
        return context

    def _draft_consumer(self, slot, context, seed):
        return self.proposer.draft_revision(slot, context, seed,
            max_chars=int(self.options['refinement_chars']))

    def _kind(self, round_id):
        if round_id in self.kind_offers:
            return self.kind_offers[round_id]
        kinds = list(self.options['support_kinds'])
        offers = rng_for(self.seed, 'frontier_potion_offers', round_id).sample(kinds, len(kinds))
        if not self.action_credit or rng_for(self.seed, 'frontier_potion_explore', round_id).random() < 0.25:
            chosen = offers[0]
        else:
            chosen = max(offers, key=lambda k: self.action_credit.get(k, [0., 0])[0] /
                         max(1, self.action_credit.get(k, [0., 0])[1]))
        self.kind_offers[round_id] = chosen
        return chosen

    def _fresh(self, arms, tasks):
        keys = set()
        for build, kind in arms:
            for task in tasks:
                key = self.evaluator._key(build, task, kind) if hasattr(self.evaluator, '_key') else (
                    build.execution_id, task.id, kind)
                keys.add(key)
        return sum(key not in self.evaluator.cache for key in keys)

    def _room(self, arms, tasks, extra=0.):


        fresh = self._fresh(arms, tasks)
        return fresh == 0 and extra == 0 or self.budget.remaining >= (
            1.6 * self.evaluator.task_cost * fresh + extra)

    def _reserve_selection(self):
        amount = max(0., min(self.selection_cap - self.selection_spent, self.budget.remaining))
        if amount:
            self.selection_hold = self.budget.reserve(amount)

    def _release_selection(self):
        self.budget.release(self.selection_hold)
        self.selection_hold = 0.

    def _parent(self, round_id):
        rows = list(self.frontier.values())
        if not rows:
            return self.start
        rng = rng_for(self.seed, 'frontier_parent', round_id)
        explore = rng.random() < float(self.options['frontier_explore_probability'])
        if explore and len(rows) > 1:
            rows = [r for r in rows if r.build.execution_id != self.start.execution_id]
            picked = rng.choice(rows)
        else:
            picked = max(rows, key=lambda r: (self._utility(r.evaluation), -r.visits, r.born))
        picked.visits += 1
        self.events.emit('frontier_parent', round=round_id, build=picked.build.id,
            execution=picked.build.execution_id, exploration=explore, utility=self._utility(picked.evaluation))
        return picked.build

    def _trim_frontier(self):
        capacity = int(self.options['frontier_capacity'])
        if len(self.frontier) <= capacity:
            return
        rows = list(self.frontier.values())
        vectors = {r.build.execution_id: self._vector(r.evaluation) for r in rows}
        def priority(row):
            vec = vectors[row.build.execution_id]
            contribution = sum(value > max(v[i] for key, v in vectors.items()
                if key != row.build.execution_id) + 1e-9 for i, value in enumerate(vec))
            return contribution, self._utility(row.evaluation), row.born
        protected = {self.start.execution_id, self.champion.execution_id}
        protected &= set(self.frontier)
        ranked = sorted((r for r in rows if r.build.execution_id not in protected), key=priority, reverse=True)
        keep = {r.build.execution_id for r in ranked[:capacity - len(protected)]} | protected
        for key in list(self.frontier):
            if key not in keep:
                row = self.frontier.pop(key)
                self.events.emit('frontier_evicted', build=row.build.id, reason='bounded_anchor_coverage_and_utility')

    def _admit(self, candidate, base, confirm, round_id):
        allowed = set(self.evaluator.dataset.splits['train'])
        if any(t not in allowed for t in base.task_ids + candidate.evaluation.task_ids) or any(
                t.id not in allowed for t in confirm):
            raise PermissionError('Frontier admission is training only')
        if self._anchor is None or base.task_ids != [t.id for t in self._anchor]:
            raise ValueError('Frontier admission requires the common ordered training anchor')
        if candidate.build.execution_id in self.frontier:
            return False
        base.paired(candidate.evaluation)
        a = self.evaluator.evaluate(self.best, confirm)
        b = self.evaluator.evaluate(candidate.build, confirm)
        old, new = base.merged(a), candidate.evaluation.merged(b)
        change = paired_delta(old, new)
        gain = self._utility(new) - self._utility(old)
        admitted = (gain >= float(self.options['frontier_min_gain']) and
                    len(change.regressed) <= int(self.options['frontier_max_regressions']))
        row = {'round': round_id, 'parent': self.best.id, 'candidate': candidate.build.id,
            'tag': candidate.tag, 'anchor_ids': base.task_ids, 'confirm_ids': a.task_ids,
            'utility_gain': gain, 'fixed': list(change.fixed), 'regressed': list(change.regressed),
            'admitted': admitted, 'scope': 'training_frontier_only_not_deployment_gain'}
        self.events.emit('frontier_admission', **row)
        if admitted:
            self.frontier[candidate.build.execution_id] = FrontierEntry(candidate.build,
                candidate.evaluation, round_id, candidate.tag)
            self.frontier_updates.append(row)
            self.save(candidate.build)
            self._trim_frontier()
            self.persist()
        return admitted

    def _combination_choices(self, base, screen, panel, round_id):
        if round_id < int(self.search['warmup_rounds']):
            return []
        offers = []
        parent_vector = self._vector(base)
        for entry in self.frontier.values():
            if entry.build.execution_id == self.best.execution_id:
                continue
            vector = self._vector(entry.evaluation)
            expected = sum(max(0., b - a) for a, b in zip(parent_vector, vector))
            for item in entry.build.execution_items:
                if item.slot not in self.slots or self.best.get(item.key) == item:
                    continue
                offers.append((expected, entry.born, item))
        rng_for(self.seed, 'frontier_composition_offers', round_id).shuffle(offers)
        offers.sort(key=lambda x: (x[0], x[1]), reverse=True)
        choices, seen = [], set()
        for expected, _, item in offers:
            build = self.best.with_item(item)
            if build.execution_id in seen:
                continue
            seen.add(build.execution_id)
            if not self._room([(build, None)], screen):
                break
            result = self.evaluator.evaluate(build, screen)
            choices.append(Choice(build, result, 'frontier_composition', item))
            partners = [x for x in self.best.execution_items if x.slot in self.slots
                        and x.key != item.key and self.start.get(x.key) != x]
            if partners:
                partner = rng_for(self.seed, 'frontier_audit_partner', round_id, item.id).choice(partners)

                common = self.best.with_item(self.start.get(item.key)).with_item(self.start.get(partner.key))
                key = (common.execution_id, item.id, partner.id)
                if key in self.combination_rejections:
                    choices.pop()
                    self.events.emit('composition_pruned', round=round_id, item=item.id,
                        partner=partner.id, reason='same_version_dominated_on_common_anchor')
                    continue
                self.composition_audits[build.execution_id] = (common, item, partner)
            self.events.emit('composition_offer', round=round_id, parent=self.best.id,
                item=item.id, build=build.id, anchor_coverage_heuristic=expected,
                note='heuristic from whole donor; actual composition score determines promotion')
            if len(choices) >= int(self.options['combination_offers']):
                break
        return choices

    def _composition_audit(self, choice, tasks, round_id):
        common, p, q = self.composition_audits[choice.build.execution_id]
        builds = [common, common.with_item(p), common.with_item(q), common.compose(p, q)]
        evaluations = [self.evaluator.evaluate(b, tasks) for b in builds]
        hard = [e.score for e in evaluations]
        credit = [training_quality(e) for e in evaluations]
        row = {'round': round_id, 'p': p.id, 'q': q.id, 'tag': 'cross_slot_removal',
            'common_base': common.id, 'task_ids': evaluations[0].task_ids,
            'temporary_support': False, 'scores': dict(zip(('H', 'Hp', 'Hq', 'Hpq'), hard)),
            'cell_scores': dict(zip(('H', 'Hp', 'Hq', 'Hpq'), credit)),
            'interaction': interaction(*evaluations),
            'cell_interaction': credit[3] - credit[1] - credit[2] + credit[0],
            'pair_gain_over_best_single': hard[3] - max(hard[1:3]),
            'pair_utility_over_best_single': self._utility(evaluations[3]) -
                max(self._utility(e) for e in evaluations[1:3]),
            'n': len(tasks), 'evidence_status': 'training_panel'}
        self.permanent_pairs.append(row)
        pair_vector = self._vector(evaluations[3])
        if any(all(a <= b + 1e-9 for a, b in zip(pair_vector, self._vector(single))) and
               any(a < b - 1e-9 for a, b in zip(pair_vector, self._vector(single)))
               for single in evaluations[1:3]):
            self.combination_rejections.add((common.execution_id, p.id, q.id))
        self.events.emit('permanent_pair', **row)
        self.events.emit('permanent_pair_credit', **row)
        choices = []
        for build, evaluation, tag in zip(builds[1:], evaluations[1:], ('p', 'q', 'pq')):
            self.save(build)
            choices.append(Choice(build, evaluation, 'cross_slot_removal:' + tag))
        return choices

    def _probe(self, p, kind, base, panel, round_id, stage):
        row = super()._probe(p, kind, base, panel, round_id, stage)
        hp = self.evaluator.evaluate(self.best.with_item(p), panel)
        hpz = self.evaluator.evaluate_supported(self.best.with_item(p), panel, kind)
        marginal = self._utility(hpz) - self._utility(hp)
        total = self._utility(hpz) - self._utility(base)

        row.update(eligible=marginal > 0.02 and total > 0.02,
                   marginal_utility=marginal, total_utility=total,
                   priority=marginal + total, eligibility_source='positive_marginal_and_total_training_utility')
        self._responses[self._project_key(p, kind)] = row
        if stage == 'screen' and row['spent'] > 0:
            credit = self.action_credit.setdefault(kind, [0., 0])
            credit[0] += marginal + total
            credit[1] += 1
        self.events.emit('frontier_response', **row)
        return row

    def _projects(self, pool, base, screen, panel, round_id):
        if self.completion_spent + 3 * len(screen) * self.evaluator.task_cost > self.completion_cap:
            return []
        offers = []
        for p in pool:
            kind = next((k for k in declared_kinds(p) if k in self.options['support_kinds']), None)
            if kind is None or self.project_attempts.get(self._project_key(p, kind), 0) >= int(self.options['max_project_attempts']):
                continue
            hp = self.best.with_item(p)
            if self._room([(hp, None)], screen) and self._response_room([(hp, None)], screen):
                ep = self.evaluator.evaluate(hp, screen)
                offers.append((p, kind, self._utility(ep)))
        offers.sort(key=lambda r: r[2], reverse=True)
        if not offers:
            return []
        if self.method == 'complement_direct':
            p, kind, _ = offers[0]
            return [(p, kind, 'ordinary_direct_completion', '')]
        rows = []
        for p, kind, _ in offers[:int(self.options['probe_projects'])]:
            hp = self.best.with_item(p)
            arms = [(hp, None), (self.best, kind), (hp, kind)]
            if self._room(arms, screen) and self._response_room(arms, screen):
                rows.append((p, kind, self._probe(p, kind, base, screen, round_id, 'screen')))
        rows.sort(key=lambda r: (r[2]['eligible'], r[2]['priority']), reverse=True)
        for p, kind, row in rows:
            if not row['eligible']:
                continue
            expanded = screen + [t for t in panel if t not in screen][:2]
            hp = self.best.with_item(p)
            arms = [(self.best, None), (hp, None), (self.best, kind), (hp, kind)]
            if self._room(arms, expanded) and self._response_room(arms, expanded):
                full = self._probe(p, kind, self.evaluator.evaluate(self.best, expanded),
                                   expanded, round_id, 'expanded')
                if full['eligible']:
                    return [(p, kind, 'confirmed_positive_marginal_support',
                             self._paired_support_evidence(p, kind, expanded, full))]


            return [(p, kind, 'screen_signal_catalog_only', '')]
        warm = int(self.search['warmup_rounds'])
        if round_id == warm or (round_id - warm + 1) % int(self.options['explore_every']) == 0:
            p, kind, _ = offers[0]
            return [(p, kind, 'bounded_catalog_exploration', '')]
        return []

    def _complete_project(self, p, kind, support, base, screen, panel, round_id):
        before = self.budget.spent
        try:
            key = self._project_key(p, kind)
            q = make_source_item(kind)
            hp, hq, hpq = self.best.with_item(p), self.best.with_item(self._prepare_q(q)), self.best.compose(p, self._prepare_q(q))
            if not self._room([(hp, None), (hq, None), (hpq, None)], screen):
                self.events.emit('completion_skipped', round=round_id, reason='protected_selection_reserve')
                return []
            self.project_attempts[key] = self.project_attempts.get(key, 0) + 1
            choices = self._normal_pair(p, q, base, screen, round_id, 'source_adoption')
            response = self._responses.get(key)
            assisted = self.evaluator.evaluate_supported(hp, screen, kind) if response else None
            source_best = max(self._utility(c.evaluation) for c in choices)


            demonstrated_gap = (response and response['stage'] == 'expanded' and response['eligible']
                and assisted and self._utility(assisted) > source_best + 0.02)
            ordinary_exploration = (self.method == 'complement_direct' and
                (round_id - int(self.search['warmup_rounds'])) % int(self.options['explore_every']) == 0)
            if ((demonstrated_gap or ordinary_exploration) and
                    self._room([], [], extra=1.6 * self.evaluator.task_cost * (len(screen) * 2 + 1))):
                context = self._context(self.best, self.evaluator.evaluate(hp, screen), round_id)
                learned = self.proposer.draft_producer(context, p,
                    seed_for(self.seed, 'frontier_producer', round_id, p.id), support_evidence=support)
                if learned is not None:
                    learned = self._prepare_q(learned)
                    if self._room([(self.best.with_item(learned), None), (self.best.compose(p, learned), None)], screen):
                        choices += self._normal_pair(p, learned, base, screen, round_id, 'synthesized')
            return choices
        finally:
            self.completion_spent += self.budget.spent - before

    def _check_selection(self, validation, round_id, final=False):
        self._release_selection()
        before = self.budget.spent
        accepted = 0
        try:
            rows = [r for r in self.frontier.values() if r.build.execution_id not in self.tested_selection]
            rows.sort(key=lambda r: (self._utility(r.evaluation), r.born), reverse=True)
            if not final:
                rows = rows[:int(self.options['selection_candidates'])]
            for row in rows:
                if not self._room([(row.build, None)], validation):
                    self.events.emit('selection_deferred', round=round_id, reason='insufficient_finishing_reserve', final=final)
                    break
                if not final and self.selection_spent + self.budget.spent - before >= self.selection_cap:
                    break
                checked, gate = strict_gate(self.evaluator, row.build, validation,
                    self.champion_validation, float(self.search['min_validation_gain']),
                    int(self.search['validation_chunk_size']), self.events)
                gained = bool(checked and checked.score > self.champion_validation.score +
                              float(self.search['min_validation_gain']))
                self.tested_selection.add(row.build.execution_id)
                item = {'round': round_id, 'build': row.build.id, 'tag': row.tag, 'final': final,
                    'gate': gate, 'selection_score': checked.score if checked else None, 'accepted': gained}
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

    def _can_start_round(self, round_id):

        estimated = 1.6 * self.evaluator.task_cost * (
            int(self.search['screen_size']) * self._consumer_count(round_id < int(self.search['warmup_rounds'])) +
            int(self.search['batch_size']) + int(self.search['confirm_size']) + 1)
        if self.budget.remaining < estimated:
            self.events.emit('round_reserve_stop', round=round_id, remaining=self.budget.remaining,
                selection_reserved=self.selection_hold, estimated_minimum_chain=estimated)
            return False
        return True

    def floor(self, round_id, train, validation):
        try:
            self._floor(round_id, train, validation)
        finally:
            self._release_training_chain()

    def _reserve_training_chain(self, panel, confirm, screen, round_id):
        return None

    def _release_training_chain(self):
        return None

    def _shortlist(self, choices, round_id):
        return self._order(choices, round_id)[:int(self.search['screen_top_k'])]

    def _floor(self, round_id, train, validation):
        before = self.budget.spent
        self.best = self._parent(round_id)
        incumbent = self.best
        panel, confirm = self._batches(train, round_id)
        size = int(self.search['screen_size'])
        offset = round_id * size % len(panel)
        screen = [panel[(offset + i) % len(panel)] for i in range(size)]
        warmup = round_id < int(self.search['warmup_rounds'])
        base = self.evaluator.evaluate(self.best, panel)
        self._remember_parent(base)
        ordinary_context = self._context(self.best, base, round_id)
        ordinary_context = Context(ordinary_context.build, ordinary_context.evidence)
        self._reserve_training_chain(panel, confirm, screen, round_id)
        choices = []
        for index in range(self._consumer_count(warmup)):
            slot = self._consumer_slot(round_id, index, warmup)
            context = ordinary_context
            if not warmup and index > 0:
                probe_panel = screen + [t for t in panel if t not in screen]
                context = self._generation_context(self.best, base, probe_panel, round_id)
            p = self._draft_consumer(slot, context, seed_for(self.seed, 'frontier_consumer', round_id, index))
            if p is None or p.id in self.seen:
                continue
            self.seen.add(p.id)
            build = self.best.with_item(p)
            if not self._room([(build, None)], screen):
                break
            result = self.evaluator.evaluate(build, screen)
            self._remember_consumer(p, result)
            self.archive[p.id] = WeakItem(p, round_id)
            self.save(build)
            choices.append(Choice(build, result, 'direct:p', p))

        for key in list(self.archive):
            if round_id - self.archive[key].born >= int(self.options['weak_ttl_rounds']):
                del self.archive[key]
        while len(self.archive) > int(self.options['weak_capacity']):
            del self.archive[min(self.archive, key=lambda k: (self.archive[k].born, k))]
        choices += self._combination_choices(base, screen, panel, round_id)
        probe_before = self.budget.spent
        try:
            weak = self._weak_pool(base, screen, round_id)
            projects = [] if warmup else self._projects(weak, self.evaluator.evaluate(self.best, screen),
                                                       screen, panel, round_id)
        finally:
            self._record_probe_cost(self.budget.spent - probe_before)
        for p, kind, reason, support in projects:

            if self.completion_spent >= self.completion_cap:
                break
            self.events.emit('completion_investment', round=round_id, p=p.id, support_kind=kind, reason=reason)
            choices += self._complete_project(p, kind, support,
                self.evaluator.evaluate(self.best, screen), screen, panel, round_id)
        self._release_training_chain()
        expanded, admitted, expanded_groups = [], 0, 0
        for choice in self._shortlist(choices, round_id):
            arms = [(choice.build, None), (self.best, None)]
            composition = self.composition_audits.get(choice.build.execution_id)
            if composition:
                common, p, q = composition
                arms += [(common, None), (common.with_item(p), None), (common.with_item(q), None)]
            elif choice.p is not None and choice.q is not None:
                arms += [(self.best.with_item(choice.p), None),
                         (self.best.with_item(self._prepare_q(choice.q)), None)]

            extra = 1.6 * self.evaluator.task_cost * self._fresh(
                [(self.best, None), (choice.build, None)], confirm)
            if not self._room(arms, panel, extra=extra):
                self.events.emit('candidate_deferred', round=round_id, build=choice.build.id,
                                 reason='protected_selection_and_confirmation_reserve')
                continue
            if composition:
                expanded += self._composition_audit(choice, panel, round_id)
            elif choice.p is not None and choice.q is not None:
                expanded += self._normal_pair(choice.p, choice.q, base, panel, round_id, 'expanded_audit')
            else:
                expanded.append(Choice(choice.build, self.evaluator.evaluate(choice.build, panel),
                                       choice.tag, choice.p, choice.q))
            expanded_groups += 1
            if expanded_groups >= int(self.search['screen_top_k']):
                break
        for choice in self._order(expanded, round_id)[:int(self.search['screen_top_k'])]:
            if not self._room([(self.best, None), (choice.build, None)], confirm):
                break
            admitted += int(self._admit(choice, base, confirm, round_id))
        previous = self.champion_validation.score
        checks = 0
        if (round_id + 1) % int(self.options['validation_every']) == 0:
            checks = self._check_selection(validation, round_id)
        total = self.budget.spent - before
        additions = [r for r in self.frontier_updates if r['round'] == round_id]
        training_gain = max((r['utility_gain'] for r in additions), default=0.)
        regressions = max((len(r['regressed']) for r in additions), default=0) / (len(panel) + len(confirm))
        cost_units = total / max(1., self.evaluator.task_cost * len(panel))
        cost_penalty = float(self.config['reward']['cost_weight']) * cost_units
        row = {'round': round_id, 'warmup': warmup, 'parent': incumbent.id,
            'harness': self.champion.id, 'training_parent': self.best.id,
            'accepted': checks > 0, 'frontier_admissions': admitted,
            'validation_score': self.champion_validation.score,
            'deployed_gain': self.champion_validation.score - previous,
            'reward_source': 'separate_training_frontier_and_ordinary_selection',
            'training_utility_gain': training_gain, 'regression_rate': regressions,
            'cost_units': cost_units, 'cost_penalty': cost_penalty,
            'reward': training_gain - cost_penalty - float(self.config['reward']['regression_weight']) * regressions,
            'candidates': [{'build': c.build.id, 'tag': c.tag, 'score': c.evaluation.score,
                            'quality': training_quality(c.evaluation)} for c in expanded],
            'panel_ids': base.task_ids, 'confirm_ids': [t.id for t in confirm],
            'round_spent': total, 'spent': self.budget.spent, 'frontier_size': len(self.frontier)}
        self.history.append(row)
        self.events.emit('complement_round_complete', **row)
        self.persist()
        print(f"round={round_id} frontier_updates={admitted} accepted={checks} "
              f"selection={self.champion_validation.score:.3f} spent={self.budget.spent:.0f}", flush=True)

    def persist(self):

        self.save(self.champion)
        write_json(self.output / 'best_build.json', self.champion.to_dict())
        write_json(self.output / 'training_parent.json', self.best.to_dict())
        write_json(self.output / 'frontier_state.json', {
            'search_policy_version': self.policy_version, 'champion': self.champion.to_dict(),
            'training_parent': self.best.to_dict(), 'frontier_updates': self.frontier_updates,
            'frontier': [{'build': r.build.to_dict(), 'anchor_score': r.evaluation.score,
                'anchor_quality': training_quality(r.evaluation), 'anchor_ids': r.evaluation.task_ids,
                'vector': self._vector(r.evaluation), 'born': r.born, 'tag': r.tag, 'visits': r.visits}
                for r in self.frontier.values()], 'selection_checks': self.selection_checks,
            'selection_spent': self.selection_spent, 'selection_cap': self.selection_cap,
            'selection_reserved': self.selection_hold, 'probe_spent': self.probe_spent,
            'completion_spent': self.completion_spent, 'action_credit': self.action_credit,
            'kind_offers': self.kind_offers,
            'project_attempts': [{'key': list(k), 'count': v} for k, v in self.project_attempts.items()],
            'combination_rejections': [list(k) for k in sorted(self.combination_rejections)],
            'history': self.history, 'probes': self.probes, 'permanent_pairs': self.permanent_pairs,
            'generation_probes': self.generation_probes,
            'note': 'Training frontier changes do not count as accepted deployment modifications'})

    def run(self):
        started, status, stop = time.monotonic(), 'completed', 'maximum rounds reached'
        train = self.evaluator.dataset.split('train', self.search['train_limit'])
        validation = self.evaluator.dataset.split('validation', self.search['validation_limit'])
        initial = None
        initial_cost = 0.
        initial_selection_cost = 0.
        fatal = False
        whole_deadline = self.budget.deadline
        if self.search['max_seconds']:
            cap = float(self.search['max_seconds'])
            finishing_seconds = min(900., cap * 0.2)
            full = min(whole_deadline, started + cap) if whole_deadline else started + cap
            self.budget.deadline = full - finishing_seconds
            whole_deadline = full
        self.persist()
        try:
            before = self.budget.spent
            self.champion_validation = self.evaluator.evaluate(self.champion, validation)
            self.best_validation = self.champion_validation
            self.selection_spent += self.budget.spent - before
            initial_selection_cost = self.budget.spent - before
            self.tested_selection.add(self.start.execution_id)
            initial = self.champion_validation.score
            anchor, _ = self._batches(train, 0)
            evaluation = self.evaluator.evaluate(self.start, anchor)
            self.frontier[self.start.execution_id] = FrontierEntry(self.start, evaluation, -1, 'starter')
            initial_cost = self.budget.spent
            self._reserve_selection()
            for round_id in range(int(self.search['rounds'])):
                if self.search['max_seconds'] and time.monotonic() - started >= self.search['max_seconds']:
                    status, stop = 'time_limit', 'wall-clock cap reached between rounds'
                    break
                if not self._can_start_round(round_id):
                    status, stop = 'insufficient_round_reserve', 'keeping held-out finishing reserve'
                    break
                self.floor(round_id, train, validation)
        except BudgetExceeded as exc:
            status, stop = 'budget_exhausted', str(exc)
            self.events.emit('complement_chain_aborted', reason=stop, spent=self.budget.spent)
        except BaseException:


            fatal = True
            raise
        finally:

            self._release_selection()
            self.budget.deadline = whole_deadline
            final_before = self.budget.spent
            try:
                if self.champion_validation is not None and not fatal:
                    self._check_selection(validation, len(self.history), final=True)
            except BudgetExceeded as exc:
                self.events.emit('final_selection_aborted', reason=str(exc), spent=self.budget.spent)
            finally:
                final_cost = self.budget.spent - final_before
                self.persist()
        unfinished = max(0., self.budget.spent - initial_cost - final_cost -
                         sum(r['round_spent'] for r in self.history))
        return {'method': self.method, 'search_policy_version': self.policy_version, 'status': status,
            'stop_reason': stop, 'partial': self.champion_validation is None,
            'test_evaluated': False, 'best_build': self.champion.to_dict(),
            'initial_validation_score': initial,
            'final_validation_score': self.champion_validation.score if self.champion_validation else None,
            'execution_changed_from_start': self.champion.execution_id != self.start.execution_id,
            'accepted_rounds': sum(r['accepted'] for r in self.selection_checks),
            'frontier_updates': len(self.frontier_updates), 'completed_rounds': len(self.history),
            'history': self.history, 'selection_checks': self.selection_checks,
            'initial_validation_spent': initial_selection_cost,
            'bootstrap_training_spent': initial_cost - initial_selection_cost,
            'final_selection_spent': final_cost,
            'selection_total_spent': self.selection_spent,
            'unfinished_round_spent': unfinished, 'unfinished_chain_gain': 0.,
            'budget': self.budget.snapshot(), 'seconds': time.monotonic() - started,
            'train_ids': [t.id for t in train], 'validation_ids': [t.id for t in validation],
            'selection_rule': 'train_frontier_accumulation_then_strict_full_ordinary_heldout_task_pass',
            'probe_count': len(self.probes), 'permanent_pair_count': len(self.permanent_pairs),
            'reward_scope': 'training progress and deployment gain logged separately; no assisted deployment reward'}
