from __future__ import annotations

import statistics

from .campaign_evolution import campaign_context
from .complement import Choice, WeakItem
from .core import BudgetExceeded, paired_delta, rng_for, seed_for, write_json
from .demands import consumer_contract, declared_kinds
from .evolution import Context, trace_excerpt
from .expedition_search import ExpeditionComplementSearch
from .frontier_search import FrontierComplementSearch
from .response_search import training_quality
from .support import make_source_item


class CampaignComplementSearch(ExpeditionComplementSearch):
    policy_version = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation = self.options.get('campaign_ablation', 'full')
        if self.ablation not in ('full', 'no_composition', 'no_potion', 'uniform_route',
                                 'no_ordinary_joint', 'within_round_partners'):
            raise ValueError('Unknown campaign ablation')
        if 'inspect' not in self.slots and 'inspect' not in self.search['disabled_slots']:
            self.slots.append('inspect')
        self.evidence_exposures = {}
        self.slot_credit, self.item_credit, self._slot_offers = {}, {}, {}
        self._round_items, self._recent_task_costs = [], []
        self.joint_screens, self.audit_outcomes = [], []
        self.audit_spent = 0.
        self.audit_cap = self.budget.limit * float(self.options.get('campaign_audit_budget_fraction', .08))
        self.unknown_producer_attempts = 0

    def _context(self, build, evaluation, round_id):
        return campaign_context(self, build, evaluation, round_id)

    def _training_tasks(self, tasks):
        allowed = set(self.evaluator.dataset.splits['train'])
        if any(t.id not in allowed for t in tasks):
            raise PermissionError('Campaign search evidence and interaction audits are training only')

    def _task_estimate(self):
        average = self.evaluator.task_cost
        recent = statistics.median(self._recent_task_costs[-12:]) if self._recent_task_costs else average
        return max(.85 * average, recent) * float(self.options.get('campaign_cost_safety', 1.25))

    def _evaluate(self, build, tasks):
        fresh, before = self._fresh([(build, None)], tasks), self.budget.spent
        result = self.evaluator.evaluate(build, tasks)
        if fresh:
            self._recent_task_costs.append((self.budget.spent - before) / fresh)
        return result

    def _room(self, arms, tasks, extra=0.):
        return self.budget.remaining >= self._task_estimate() * self._fresh(arms, tasks) + extra

    def _reserve_selection(self):
        self._release_selection()
        validation = self.evaluator.dataset.split('validation', self.search['validation_limit'])
        pending = [r for r in self.frontier.values() if r.build.execution_id not in self.tested_selection]
        pending.sort(key=lambda r: (r.evaluation.score, self._utility(r.evaluation), r.born), reverse=True)
        count = int(self.options['selection_reserve_candidates'])
        fresh = 0
        for row in pending[:count]:
            confirm = [self.tasks[t] for t in self.confirmed_candidates.get(row.build.execution_id, [])]
            audit = list(self._anchor or []) + confirm
            fresh += self._fresh([(row.build, None)], validation)
            fresh += self._fresh([(row.build, None), (self.champion, None)], audit)
        if not pending:
            fresh = len(validation) + int(self.search['confirm_size'])
        amount = max(0., min(self.selection_cap - self.selection_spent,
                             fresh * self._task_estimate(), self.budget.remaining))
        if amount:
            self.selection_hold = self.budget.reserve(amount)
        self.events.emit('selection_reserve', amount=amount, pending=len(pending),
            protected_candidates=count, estimated_fresh_tasks=fresh,
            scope='campaign_actual_cache_deficit_and_recent_paid_cost')

    def _reserve_training_chain(self, panel, confirm, screen, round_id):
        self._release_training_chain()
        fresh = len(panel) - len(screen) + len(confirm) + self._fresh([(self.best, None)], confirm)
        amount = self._task_estimate() * fresh
        if amount <= self.budget.remaining:
            self._chain_hold = self.budget.reserve(amount)
        self.events.emit('training_chain_reserve', round=round_id, amount=self._chain_hold,
            fresh_tasks=fresh, protects='ordinary_anchor_and_disjoint_confirmation')

    def _can_start_round(self, round_id):
        self._lean_round = False
        train = self.evaluator.dataset.split('train', self.search['train_limit'])
        panel, confirm = self._batches(train, round_id)


        parent_fresh = max((self._fresh([(r.build, None)], confirm) for r in self.frontier.values()),
                           default=len(confirm))
        chain = len(panel) + len(confirm) + parent_fresh + 1
        per = self._task_estimate()
        count = self._consumer_count(round_id < int(self.search['warmup_rounds']))
        regular = per * (chain + int(self.search['screen_size']) * (count - 1))
        if self.budget.remaining >= regular:
            return True
        if self.budget.remaining >= per * chain:
            self._lean_round = True
            self.events.emit('round_offer_reduced', round=round_id, remaining=self.budget.remaining,
                estimated_chain=per * chain, reason='campaign_single_complete_chain')
            return True
        self.events.emit('round_reserve_stop', round=round_id, remaining=self.budget.remaining,
            selection_reserved=self.selection_hold, estimated_minimum_chain=per * chain)
        return False

    def _consumer_count(self, warmup):
        if self._lean_round:
            return 1
        return min(len(self.slots), int(self.search['candidates']) if warmup else
                   int(self.options.get('campaign_candidates', 3)))

    def _consumer_slot(self, round_id, index, warmup):
        if warmup:
            return self.slots[(round_id + index) % len(self.slots)]
        if round_id not in self._slot_offers:
            rng = rng_for(self.seed, 'campaign_slot_offers', round_id)
            offers = rng.sample(self.slots, len(self.slots))
            if (self.ablation != 'uniform_route' and
                    rng.random() >= float(self.options.get('campaign_route_exploration', .2))):


                offers.sort(key=lambda s: (s not in self.slot_credit,
                    self.slot_credit.get(s, [0., 0])[0] / max(1, self.slot_credit.get(s, [0., 0])[1])),
                    reverse=True)
            self._slot_offers[round_id] = offers
        return self._slot_offers[round_id][index % len(self.slots)]

    def _parent(self, round_id):
        if self.ablation != 'uniform_route' or not self.frontier:
            return super()._parent(round_id)
        row = rng_for(self.seed, 'campaign_uniform_parent', round_id).choice(list(self.frontier.values()))
        row.visits += 1
        self.events.emit('frontier_parent', round=round_id, build=row.build.id,
            execution=row.build.execution_id, exploration=True, utility=self._utility(row.evaluation),
            route='uniform_ablation')
        return row.build

    def _kind(self, round_id):
        if self.ablation != 'uniform_route':
            return super()._kind(round_id)
        if round_id not in self.kind_offers:
            self.kind_offers[round_id] = rng_for(self.seed, 'campaign_uniform_kind', round_id).choice(
                list(self.options['support_kinds']))
        return self.kind_offers[round_id]

    def _generation_context(self, build, evaluation, panel, round_id):
        self._training_tasks(panel)
        context = self._context(build, evaluation, round_id)
        if self.ablation == 'no_potion' or round_id < int(self.search['warmup_rounds']):
            return context
        screen = panel[:int(self.search['screen_size'])]
        kind = self._kind(round_id)
        self._probe_stage_started = self.budget.spent
        arms = [(build, None), (build, kind)]
        if not self._room(arms, screen) or not self._response_room(arms, screen):
            return context
        before = self.budget.spent
        try:
            plain = self._evaluate(build, screen)
            assisted = self.evaluator.evaluate_supported(build, screen, kind)
            gain = self._utility(assisted) - self._utility(plain)
            credit = self.action_credit.setdefault(kind, [0., 0])
            credit[0] += gain
            credit[1] += 1
            self.generation_probes.append({'round': round_id, 'kind': kind, 'task_ids': plain.task_ids,
                'hard_scores': [plain.score, assisted.score],
                'cell_scores': [training_quality(plain), training_quality(assisted)],
                'spent': self.budget.spent - before, 'utility_gain': gain})
            blocks = [f'Paid input-only {kind} intervention: ordinary utility gain={gain:.4f}. '
                      'Useful observations may guide a reusable procedure; a losing observation '
                      'does not justify adding an observation dependency.']
            chemistry_limit = min(int(self.config['proposal']['evidence_chars']), 5000)
            if gain > float(self.options['frontier_min_gain']):
                context.fact_kind = kind
                context.chemistry = consumer_contract(kind) + '\n\n'


            limit = chemistry_limit - len(context.chemistry)
            for index, task in enumerate(screen, 1):
                a, b = plain.by_task()[task.id], assisted.by_task()[task.id]
                header = (f'Example {index}, COMPLETE INSTRUCTION:\n{task.instruction}\n'
                          f'Normal pass={a.success}; with observations pass={b.success}\n')
                room = (limit - sum(len(b) + 2 for b in blocks)) // (len(screen) - index + 1)
                if len(header) + 200 > room:
                    blocks.append('One intervention example was omitted whole because its complete instruction did not fit.')
                    continue
                each = max(1, (room - len(header) - 80) // 3)
                blocks.append(header +
                    f'Normal procedure excerpt: {trace_excerpt(a.trace, each)}\n'
                    f'Observed procedure excerpt: {trace_excerpt(b.trace, each)}\n' +
                    self.evaluator.support_evidence(build, [task], kind, limit_chars=each))
            context.chemistry += '\n\n'.join(blocks)
            return context
        except BudgetExceeded as exc:
            self.events.emit('campaign_optional_deferred', round=round_id,
                route='generation_potion', reason=str(exc))
            return context
        finally:
            self._record_probe_cost(self.budget.spent - before)

    def _projects(self, pool, base, screen, panel, round_id):
        if self.ablation == 'no_potion' or self._lean_round:
            return []


        projects = FrontierComplementSearch._projects(self, pool[:1], base, screen, panel, round_id)
        if projects:
            return projects
        cap = int(self.options.get('campaign_unknown_producer_attempts', 1))
        if self.unknown_producer_attempts >= cap or not self._exploration_room(screen):
            return []
        for p in pool[:1]:
            kind = next((k for k in declared_kinds(p) if k in self.options['support_kinds']), None)
            if kind and self._project_key(p, kind) not in self._responses:
                self._bounded_projects.add((round_id, self._project_key(p, kind)))
                return [(p, kind, 'unobserved_bounded_producer_trial', '')]
        return []

    def _complete_project(self, p, kind, support, base, screen, panel, round_id):
        if self.ablation == 'no_potion':
            return []
        self._training_tasks(screen + panel)
        key = self._project_key(p, kind)
        unknown = (round_id, key) in self._bounded_projects
        if unknown and self.unknown_producer_attempts >= int(self.options.get('campaign_unknown_producer_attempts', 1)):
            return []
        before, hold, choices = self.budget.spent, 0., []
        try:
            allowance = max(0., min(self.completion_cap - self.completion_spent, self.budget.remaining))
            hold = self.budget.reserve(max(0., self.budget.remaining - allowance))
            q = self._prepare_q(make_source_item(kind))
            hp = self.best.with_item(p)
            if not self._room([(hp, None), (self.best.with_item(q), None),
                               (self.best.compose(p, q), None)], screen):
                return []
            self.project_attempts[key] = self.project_attempts.get(key, 0) + 1
            choices = self._normal_pair(p, q, base, screen, round_id, 'source_adoption')
            response = self._responses.get(key)
            positive = bool(response and response['stage'] == 'expanded' and response['eligible'])
            if positive:
                assisted = self.evaluator.evaluate_supported(hp, screen, kind)
                positive = self._utility(assisted) > max(self._utility(c.evaluation) for c in choices) + .02
            if not (positive or unknown):
                return choices


            producer_before, producer_hold = self.budget.spent, 0.
            allowance = max(0., min(self.producer_exploration_cap - self.producer_exploration_spent,
                                   self.budget.remaining))
            try:
                producer_hold = self.budget.reserve(max(0., self.budget.remaining - allowance))
                if (self.producer_exploration_attempts >= int(self.options['producer_exploration_attempts']) or
                        not self._room([], [], extra=self._task_estimate() * (2 * len(screen) + 1))):
                    return choices
                self.producer_exploration_attempts += 1
                self.unknown_producer_attempts += int(unknown)
                context = self._context(hp, self._evaluate(hp, screen), round_id)
                producer_args = {'support_evidence': support}
                if hasattr(self.evaluator, 'check_producer_inputs'):
                    producer_args['runtime_check'] = lambda item: self.evaluator.check_producer_inputs(item, screen)
                learned = self.proposer.draft_producer(context, p,
                    seed_for(self.seed, 'campaign_producer', round_id, p.id), **producer_args)
                if learned is not None:
                    learned = self._prepare_q(learned)
                    if self._room([(self.best.with_item(learned), None), (self.best.compose(p, learned), None)], screen):
                        choices += self._normal_pair(p, learned, base, screen, round_id, 'synthesized')
            finally:
                self.budget.release(producer_hold)
                self.producer_exploration_spent += self.budget.spent - producer_before
            return choices
        except BudgetExceeded as exc:
            self.events.emit('campaign_optional_deferred', round=round_id,
                route='bounded_completion', reason=str(exc))
            return choices
        finally:
            self.budget.release(hold)
            self.completion_spent += self.budget.spent - before

    def _normal_pair(self, p, q, base, tasks, round_id, tag):
        self._training_tasks(tasks)
        q = self._prepare_q(q)
        hp = self.best.with_item(p)
        first = Choice(hp, self._evaluate(hp, tasks), tag + ':p', p)
        if self.ablation == 'no_composition':
            hq = self.best.with_item(q)
            return [first, Choice(hq, self._evaluate(hq, tasks), tag + ':q', None, q)]
        pair = self.best.compose(p, q)
        result = self._evaluate(pair, tasks)
        self.composition_audits[pair.execution_id] = self.best, p, q
        self.save(pair)
        self._record_joint(pair, result, base, p, q, round_id, tag)
        return [first, Choice(pair, result, tag + ':pq', p, q)]

    def _record_joint(self, build, evaluation, base, p, q, round_id, tag):
        row = {'round': round_id, 'build': build.id, 'p': p.id, 'q': q.id, 'tag': tag,
            'task_ids': evaluation.task_ids, 'score': evaluation.score,
            'utility_gain': self._utility(evaluation) - self._utility(base),
            'evidence_status': 'ordinary_joint_screen_not_interaction_evidence'}
        if tag == 'archive_composition':


            p_entry, q_entry = self.archive.get(p.id), self.archive.get(q.id)
            row.update(parent_build=self.best.to_dict(), p_item=p.to_dict(), q_item=q.to_dict(),
                parent_execution_id=self.best.execution_id, candidate_execution_id=build.execution_id,
                p_birth_round=p_entry.born if p_entry else (
                    round_id if any(item.id == p.id for item in self._round_items) else None),
                q_birth_round=q_entry.born if q_entry else None)
        self.joint_screens.append(row)
        self.events.emit('campaign_joint_screen', **row)

    def _combination_choices(self, base, screen, panel, round_id):
        self._training_tasks(panel + screen)
        if (self.ablation in ('no_composition', 'no_ordinary_joint') or self._lean_round or
                round_id < int(self.search['warmup_rounds'])):
            return []
        within_round = self.ablation == 'within_round_partners'


        current_ids = {item.id for item in self._round_items} if within_round else ()
        offers, seen = [], set()
        for p in self._round_items:
            if within_round:
                entry = self.archive.get(p.id)
                if entry is not None and entry.born != round_id:
                    continue
            for old in self.archive.values():
                q = old.item
                if within_round and (old.born != round_id or q.id not in current_ids):
                    continue
                if p.key == q.key or q == self.best.get(q.key) or p == self.best.get(p.key):
                    continue
                build = self.best.compose(p, q)
                if build.execution_id in seen or build.execution_id in self.frontier:
                    continue
                if ((self.best.execution_id, p.id, q.id) in self.combination_rejections or
                        (self.best.execution_id, q.id, p.id) in self.combination_rejections):
                    continue
                seen.add(build.execution_id)
                offers.append((p, q, build))
        rng = rng_for(self.seed, 'campaign_archive_offers', round_id)
        rng.shuffle(offers)
        offers = offers[:int(self.options.get('campaign_composition_pool', 8))]
        if (self.ablation != 'uniform_route' and
                rng.random() >= float(self.options.get('campaign_route_exploration', .2))):
            offers.sort(key=lambda row: sum(self.item_credit.get(x.id, [0., 0])[0] /
                max(1, self.item_credit.get(x.id, [0., 0])[1]) for x in row[:2]), reverse=True)
        choices = []
        for p, q, build in offers[:int(self.options.get('campaign_composition_offers', 1))]:
            if not self._room([(build, None)], screen):
                break
            try:
                result = self._evaluate(build, screen)
            except BudgetExceeded as exc:
                self.events.emit('campaign_optional_deferred', round=round_id,
                    route='archive_joint_screen', reason=str(exc))
                break
            self.composition_audits[build.execution_id] = self.best, p, q
            self.save(build)
            self._record_joint(build, result, self._evaluate(self.best, screen), p, q, round_id,
                               'archive_composition')
            choices.append(Choice(build, result, 'archive_composition', p, q))
        return choices

    def _shortlist(self, choices, round_id):
        rows = self._order(choices, round_id)
        if (self.ablation != 'uniform_route' and round_id >= int(self.search['warmup_rounds']) and
                (round_id - int(self.search['warmup_rounds'])) % int(self.options['explore_every']) == 0):
            eligible = []
            for c in rows:
                if c.build.execution_id not in self.composition_audits:
                    continue
                tasks = [self.tasks[t] for t in c.evaluation.task_ids]
                parent = self._evaluate(self.best, tasks)
                if (not paired_delta(parent, c.evaluation).regressed and
                        self._utility(c.evaluation) + float(self.options['frontier_min_gain']) >= self._utility(parent)):
                    eligible.append(c)
            if eligible:
                picked = rng_for(self.seed, 'campaign_joint_exploration', round_id).choice(eligible)
                rows = [picked] + [c for c in rows if c is not picked]
        return rows

    def _audit_admitted(self, choice, panel, round_id):
        self._training_tasks(panel)
        if (choice.build.execution_id not in self.composition_audits or
                choice.build.execution_id not in self.confirmed_candidates):
            return
        common, p, q = self.composition_audits[choice.build.execution_id]
        arms = [(common, None), (common.with_item(p), None), (common.with_item(q), None),
                (common.compose(p, q), None)]
        estimate = self._task_estimate() * self._fresh(arms, panel)
        allowance = max(0., min(self.audit_cap - self.audit_spent, self.budget.remaining))
        row = {'round': round_id, 'build': choice.build.id, 'task_ids': [t.id for t in panel],
               'estimated_cost': estimate}
        if estimate > allowance:
            row.update(status='deferred', reason='bounded_interaction_audit_budget',
                       evidence_status='ordinary_admission_only_no_interaction_claim')
            self.audit_outcomes.append(row)
            self.events.emit('campaign_audit', **row)
            return
        before, hold = self.budget.spent, 0.
        try:
            hold = self.budget.reserve(max(0., self.budget.remaining - allowance))
            self._composition_audit(choice, panel, round_id)
            row.update(status='completed', evidence_status='paid_real_four_arm_training_panel')
        except BudgetExceeded as exc:
            row.update(status='incomplete', reason=str(exc),
                       evidence_status='ordinary_admission_only_no_interaction_claim')
        finally:
            self.budget.release(hold)
            cost = self.budget.spent - before
            self.audit_spent += cost
            row['spent'] = cost
            self.audit_outcomes.append(row)
            self.events.emit('campaign_audit', **row)

    @staticmethod
    def _route(choice):
        return 'joint' if choice.p is not None and choice.q is not None else (
            'observer' if choice.q is not None else 'ordinary')

    def _floor(self, round_id, train, validation):
        before = self.budget.spent
        self.best = self._parent(round_id)
        incumbent = self.best
        panel, confirm = self._batches(train, round_id)
        size = int(self.search['screen_size'])
        offset = round_id * size % len(panel)
        screen = [panel[(offset + i) % len(panel)] for i in range(size)]
        warmup = round_id < int(self.search['warmup_rounds'])
        base = self._evaluate(self.best, panel)
        self._remember_parent(base)
        ordinary = self._context(self.best, base, round_id)
        ordinary = Context(ordinary.build, ordinary.evidence)
        self._reserve_training_chain(panel, confirm, screen, round_id)
        choices, self._round_items = [], []
        count = self._consumer_count(warmup)
        for index in range(count):
            slot = self._consumer_slot(round_id, index, warmup)
            context = ordinary
            if not warmup and index == count - 1 and index > 0 and slot != 'inspect':
                context = self._generation_context(self.best, base,
                    screen + [t for t in panel if t not in screen], round_id)
            item = self._draft_consumer(slot, context, seed_for(self.seed, 'campaign_consumer', round_id, index))
            if item is None or item.id in self.seen:
                continue
            self.seen.add(item.id)
            build = self.best.with_item(item)
            if not self._room([(build, None)], screen):
                break
            result = self._evaluate(build, screen)
            self._remember_consumer(item, result)
            delta = self._utility(result) - self._utility(self._evaluate(self.best, screen))
            for ledger, key in ((self.slot_credit, slot), (self.item_credit, item.id)):
                record = ledger.setdefault(key, [0., 0])
                record[0] += delta
                record[1] += 1
            self.archive[item.id] = WeakItem(item, round_id)
            self._round_items.append(item)
            self.save(build)
            choices.append(Choice(build, result, 'direct:p', item))
        for key in list(self.archive):
            if round_id - self.archive[key].born >= int(self.options['weak_ttl_rounds']):
                del self.archive[key]
        while len(self.archive) > int(self.options['weak_capacity']):
            del self.archive[min(self.archive, key=lambda k: (self.archive[k].born, k))]
        choices += self._combination_choices(base, screen, panel, round_id)
        probe_before = self.budget.spent
        try:
            weak = self._weak_pool(base, screen, round_id) if self.ablation != 'no_potion' else []
            projects = [] if warmup else self._projects(weak, self._evaluate(self.best, screen), screen, panel, round_id)
        except BudgetExceeded as exc:
            projects = []
            self.events.emit('campaign_optional_deferred', round=round_id,
                route='support_project_probe', reason=str(exc))
        finally:
            self._record_probe_cost(self.budget.spent - probe_before)
        for p, kind, reason, support in projects:
            if self.completion_spent >= self.completion_cap:
                break
            self.events.emit('completion_investment', round=round_id, p=p.id, support_kind=kind, reason=reason)
            try:
                choices += self._complete_project(p, kind, support, self._evaluate(self.best, screen), screen, panel, round_id)
            except BudgetExceeded as exc:
                self.events.emit('campaign_optional_deferred', round=round_id,
                    route='producer_completion', reason=str(exc))
        self._release_training_chain()
        expanded, admitted, attempted_routes = [], 0, set()
        for choice in self._shortlist(choices, round_id):
            if len(expanded) >= int(self.options.get('campaign_expand_candidates', 2)):
                break
            route = self._route(choice)
            if admitted and route in attempted_routes:
                continue
            arms = [(choice.build, None), (self.best, None)]
            extra = self._task_estimate() * self._fresh(arms, confirm)
            if not self._room(arms, panel, extra=extra):
                continue


            full = Choice(choice.build, self._evaluate(choice.build, panel), choice.tag, choice.p, choice.q)
            expanded.append(full)
            attempted_routes.add(route)
            ok = self._admit(full, base, confirm, round_id)
            admitted += int(ok)
            if ok:
                self._reserve_selection()
                self._audit_admitted(full, panel, round_id)
        previous = self.champion_validation.score
        checks = 0
        if admitted or (round_id + 1) % int(self.options['validation_every']) == 0:
            checks = self._check_selection(validation, round_id)
        total = self.budget.spent - before
        additions = [r for r in self.frontier_updates if r['round'] == round_id]
        gain = max((r['utility_gain'] for r in additions), default=0.)
        regressions = max((len(r['regressed']) for r in additions), default=0) / (len(panel) + len(confirm))
        units = total / max(1., self.evaluator.task_cost * len(panel))
        penalty = float(self.config['reward']['cost_weight']) * units
        row = {'round': round_id, 'warmup': warmup, 'parent': incumbent.id,
            'harness': self.champion.id, 'training_parent': self.best.id,
            'accepted': checks > 0, 'frontier_admissions': admitted,
            'validation_score': self.champion_validation.score,
            'deployed_gain': self.champion_validation.score - previous,
            'reward_source': 'separate_training_frontier_and_ordinary_selection',
            'training_utility_gain': gain, 'regression_rate': regressions,
            'cost_units': units, 'cost_penalty': penalty,
            'reward': gain - penalty - float(self.config['reward']['regression_weight']) * regressions,
            'candidates': [{'build': c.build.id, 'tag': c.tag, 'score': c.evaluation.score,
                            'quality': training_quality(c.evaluation)} for c in expanded],
            'screened_candidates': len(choices), 'panel_ids': base.task_ids,
            'confirm_ids': [t.id for t in confirm], 'round_spent': total, 'spent': self.budget.spent,
            'frontier_size': len(self.frontier)}
        self.history.append(row)
        self.events.emit('complement_round_complete', **row)
        self.persist()
        print(f'round={round_id} screened={len(choices)} expanded={len(expanded)} '
              f'frontier_updates={admitted} accepted={checks} spent={self.budget.spent:.0f}', flush=True)

    def persist(self):
        super().persist()
        write_json(self.output / 'campaign_state.json', {
            'search_policy_version': self.policy_version, 'ablation': self.ablation,
            'evidence_exposures': self.evidence_exposures, 'slot_credit': self.slot_credit,
            'item_credit': self.item_credit, 'joint_screens': self.joint_screens,
            'interaction_audits': self.audit_outcomes, 'audit_spent': self.audit_spent,
            'audit_cap': self.audit_cap, 'unknown_producer_attempts': self.unknown_producer_attempts,
            'note': 'Ordinary admission is not proof of positive component interaction.'})

    def run(self):
        result = super().run()
        result.update(campaign_ablation=self.ablation, joint_screen_count=len(self.joint_screens),
            completed_interaction_audits=sum(r['status'] == 'completed' for r in self.audit_outcomes),
            interaction_audit_spent=self.audit_spent)
        return result
