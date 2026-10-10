from __future__ import annotations

import copy
from dataclasses import replace
import unittest
from unittest import mock

from rouge.campaign_search import CampaignComplementSearch
from rouge.complement import Choice, WeakItem
from rouge.core import BudgetExceeded
from rouge.frontier_search import FrontierEntry
from rouge.items import Item
from tests.fixtures import names
import tests.fixtures as fixture


class CampaignSearchTests(unittest.TestCase):
    panel = fixture.SearchFixture.panel
    initialize_selection = fixture.SearchFixture.initialize_selection
    events = fixture.SearchFixture.events

    def engine(self, ablation='full', **kwargs):
        old, evaluator, editor = fixture.SearchFixture.engine(self, **kwargs)
        cfg = copy.deepcopy(old.config)
        cfg['complement'].update(expedition_guided=True, campaign_guided=True,
            producer_exploration_attempts=1, producer_exploration_budget_fraction=.08,
            selection_reserve_candidates=1, campaign_ablation=ablation,
            campaign_candidates=3, campaign_expand_candidates=2,
            campaign_composition_offers=1, campaign_composition_pool=8,
            campaign_audit_budget_fraction=.08, campaign_route_exploration=.2,
            campaign_cost_safety=1.25, campaign_unknown_producer_attempts=1)
        return CampaignComplementSearch(cfg, evaluator, editor, old.events, old.output, old.start), evaluator, editor

    def synergy_run(self, ablation='full', audit_fraction=.08):
        both = lambda eq: 'weak_plan' in eq and 'weak_edit' in eq
        engine, evaluator, editor = self.engine(ablation,
            hard=lambda eq, task, z: both(eq),
            credit=lambda eq, task, z: 1. if both(eq) else .1, rounds=2)
        engine.search['warmup_rounds'] = 1
        engine.options['campaign_candidates'] = 1
        engine.audit_cap = engine.budget.limit * audit_fraction
        def draft(slot, context, seed, max_chars=2200):
            engine.budget.charge_model(20, 0, 0)
            editor.contexts.append(context)
            return Item(slot, 'weak_' + slot, 'Reusable ' + slot + ' rule with a complementary interface.')
        with mock.patch.object(engine, '_consumer_slot', side_effect=lambda r, i, w: 'plan' if r == 0 else 'edit'), \
                mock.patch.object(editor, 'draft_revision', side_effect=draft):
            result = engine.run()
        return result, engine, evaluator, editor

    def test_two_individually_useless_archived_cards_make_verified_joint_progress(self):
        result, engine, evaluator, editor = self.synergy_run()
        self.assertEqual(engine.history[0]['frontier_admissions'], 0)
        self.assertEqual(engine.history[1]['frontier_admissions'], 1)
        self.assertEqual(result['final_validation_score'], 1.)
        self.assertTrue({'weak_plan', 'weak_edit'} <= set(names(engine.champion)))
        self.assertEqual(len(engine.joint_screens), 1)
        self.assertEqual(len(engine.permanent_pairs), 1)
        pair = engine.permanent_pairs[0]
        self.assertEqual(pair['scores'], {'H': 0., 'Hp': 0., 'Hq': 0., 'Hpq': 1.})
        self.assertEqual(pair['interaction'], 1.)
        self.assertEqual(pair['n'], 6)
        self.assertTrue(all(z is None for _, _, z in evaluator.calls))
        self.assertFalse(any(set(ids) & set(evaluator.dataset.splits['test']) for _, ids, _ in evaluator.calls))
        self.assertTrue(all('Validation example' not in c.evidence for c in editor.contexts))

    def test_no_composition_cannot_solve_the_same_complementary_world(self):
        result, engine, _, _ = self.synergy_run('no_composition')
        self.assertEqual(result['final_validation_score'], 0.)
        self.assertEqual(result['frontier_updates'], 0)
        self.assertFalse(engine.joint_screens)
        self.assertFalse(engine.permanent_pairs)
        self.assertEqual(engine.champion.execution_id, engine.start.execution_id)

    def test_no_audit_budget_preserves_real_joint_admission_without_inventing_synergy(self):
        result, engine, _, _ = self.synergy_run(audit_fraction=0.)
        self.assertEqual(result['final_validation_score'], 1.)
        self.assertEqual(result['completed_interaction_audits'], 0)
        self.assertEqual(engine.audit_spent, 0.)
        self.assertFalse(engine.permanent_pairs)
        self.assertEqual(engine.audit_outcomes[0]['status'], 'deferred')
        self.assertIn('no_interaction_claim', engine.audit_outcomes[0]['evidence_status'])

    def test_no_potion_disables_support_and_producer_but_keeps_inspection_revisions(self):
        engine, evaluator, editor = self.engine('no_potion')
        panel, _ = self.panel(engine, evaluator, 2)
        base = evaluator.evaluate(engine.best, panel)
        p = Item('plan', 'consumer', 'Read columns; inspect raw input if absent.', ('facts:column_profile',))
        with mock.patch.object(evaluator, 'evaluate_supported', side_effect=AssertionError('potion prohibited')), \
                mock.patch.object(editor, 'draft_producer', side_effect=AssertionError('producer prohibited')):
            context = engine._generation_context(engine.best, base, panel, 2)
            self.assertEqual(context.fact_kind, '')
            self.assertEqual(engine._projects([p], base, panel[:2], panel, 2), [])
            self.assertEqual(engine._complete_project(p, 'column_profile', '', base, panel[:2], panel, 2), [])
            inspection = engine._draft_consumer('inspect', context, 0)
            self.assertEqual(inspection.slot, 'inspect')
        self.assertIn('inspect', engine.slots)

    def test_negative_observed_potion_does_not_spend_unknown_producer_exploration(self):
        engine, evaluator, editor = self.engine(credit=lambda eq, task, z: 0. if z else .5)
        panel, _ = self.panel(engine, evaluator, 2)
        base = evaluator.evaluate(engine.best, panel[:2])
        p = Item('plan', 'consumer', 'Use column_profile if useful, otherwise inspect raw input.',
                 ('facts:column_profile',))
        engine._probe_stage_started = engine.budget.spent
        projects = engine._projects([p], base, panel[:2], panel, 2)
        for p, kind, reason, support in projects:
            engine._complete_project(p, kind, support, base, panel[:2], panel, 2)
        self.assertTrue(engine._responses)
        self.assertFalse(any(r['eligible'] for r in engine._responses.values()))
        self.assertFalse(editor.producer_calls)
        self.assertEqual(engine.unknown_producer_attempts, 0)

    def test_potion_prompt_has_independent_bound_and_never_cuts_an_instruction(self):
        engine, evaluator, _ = self.engine(credit=lambda eq, task, z: .9 if z else .1)
        engine.config['proposal']['evidence_chars'] = 10000
        panel, _ = self.panel(engine, evaluator, 2)
        complete = 'BEGIN_COMPLETE ' + 'interpret every case; ' * 45 + ' REQUIRED_END'
        oversized = 'BEGIN_OVERSIZED ' + 'do not split this instruction; ' * 400 + ' OVERSIZED_END'
        for i, instruction in enumerate((complete, oversized)):
            task = replace(panel[i], instruction=instruction)
            engine.tasks[task.id] = task
            panel[i] = task
        evaluation = evaluator.evaluate(engine.best, panel)
        for row in evaluation.results:
            row.trace = 'Executed reusable procedure; ' * 1000
        with mock.patch.object(evaluator, 'support_evidence',
                side_effect=lambda build, tasks, kind, limit_chars: 'x' * limit_chars):
            context = engine._generation_context(engine.best, evaluation, panel, 2)
        self.assertTrue(context.fact_kind)
        self.assertLessEqual(len(context.chemistry), 5000)
        self.assertIn(complete, context.chemistry)
        self.assertNotIn('BEGIN_OVERSIZED', context.chemistry)
        self.assertIn('omitted whole', context.chemistry)
        self.assertEqual(engine.config['proposal']['evidence_chars'], 10000)
        self.assertLessEqual(len(context.evidence), 10000)

    def test_optional_budget_interrupt_does_not_destroy_existing_ordinary_candidate(self):
        engine, evaluator, _ = self.engine(rounds=1,
            credit=lambda eq, task, z: .9 if any(n.startswith('stage_') for n in eq) else .1)
        engine.search['warmup_rounds'] = 0
        engine.options['campaign_candidates'] = 1
        with mock.patch.object(engine, '_projects', side_effect=BudgetExceeded('optional sub-budget exhausted')):
            result = engine.run()
        self.assertEqual(result['completed_rounds'], 1)
        self.assertEqual(result['frontier_updates'], 1)
        self.assertTrue(result['execution_changed_from_start'])
        self.assertEqual(engine.budget.reserved, 0.)
        self.assertLessEqual(engine.budget.spent, engine.budget.limit)

    def test_confirmation_budget_interrupt_cannot_promote_a_half_tested_child(self):
        engine, evaluator, _ = self.engine(rounds=1,
            credit=lambda eq, task, z: .9 if any(n.startswith('stage_') for n in eq) else .1)
        _, confirm = self.panel(engine, evaluator)
        ids = {t.id for t in confirm}
        original = evaluator.evaluate
        def stop_child_confirm(build, tasks):
            if build.execution_id != engine.start.execution_id and {t.id for t in tasks} == ids:
                engine.budget.charge_model(25, 0, 0)
                raise BudgetExceeded('incomplete confirmation')
            return original(build, tasks)
        with mock.patch.object(evaluator, 'evaluate', side_effect=stop_child_confirm):
            result = engine.run()
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['frontier_updates'], 0)
        self.assertEqual(result['accepted_rounds'], 0)
        self.assertEqual(engine.champion.execution_id, engine.start.execution_id)
        self.assertEqual(engine.budget.reserved, 0.)

    def producer_fixture(self, unknown=False):
        engine, evaluator, editor = self.engine(credit=lambda eq, task, z: .9 if z else .1)
        panel, _ = self.panel(engine, evaluator, 2)
        screen = panel[:2]
        p = Item('plan', 'consumer', 'Use column_profile; inspect raw inputs when absent.',
                 ('facts:column_profile',))
        key = engine._project_key(p, 'column_profile')
        if unknown:
            engine._bounded_projects.add((2, key))
        else:
            engine._responses[key] = {'stage': 'expanded', 'eligible': True}
        return engine, evaluator, editor, p, panel, screen

    def test_positive_producer_uses_correct_build_evidence_and_paid_runtime_callback(self):
        engine, evaluator, editor, p, panel, screen = self.producer_fixture()
        original = editor.draft_producer
        runtime_calls = []
        def checked(item, tasks):
            engine.budget.charge_tool(0)
            runtime_calls.append((item.id, [t.id for t in tasks]))
            return [{'success': True, 'truncated': False}]
        def draft(context, partner, seed, support_evidence='', runtime_check=None):
            self.assertEqual(context.build.execution_id, engine.best.with_item(p).execution_id)
            self.assertIn('Training example', context.evidence)
            result = original(context, partner, seed, support_evidence)
            self.assertIsNotNone(runtime_check)
            runtime_check(result)
            return result
        with mock.patch.object(evaluator, 'check_producer_inputs', side_effect=checked, create=True), \
                mock.patch.object(editor, 'draft_producer', side_effect=draft):
            choices = engine._complete_project(p, 'column_profile', 'Paid positive support contrast.',
                evaluator.evaluate(engine.best, screen), screen, panel, 2)
        self.assertEqual(len(runtime_calls), 1)
        self.assertEqual(runtime_calls[0][1], [t.id for t in screen])
        self.assertTrue(any('learned_producer' in names(c.build) for c in choices))
        self.assertEqual(engine.unknown_producer_attempts, 0)
        self.assertEqual(engine.producer_exploration_attempts, 1)
        self.assertGreater(engine.producer_exploration_spent, 30)
        self.assertEqual(engine.budget.reserved, 0.)

    def test_unobserved_producer_is_finite_even_when_generation_fails(self):
        engine, evaluator, editor, p, panel, screen = self.producer_fixture(unknown=True)
        def invalid(context, partner, seed, support_evidence=''):
            engine.budget.charge_model(30, 0, 0)
            editor.producer_calls.append(support_evidence)
            return None
        with mock.patch.object(editor, 'draft_producer', side_effect=invalid):
            for _ in range(2):
                engine._complete_project(p, 'column_profile', '',
                    evaluator.evaluate(engine.best, screen), screen, panel, 2)
        self.assertEqual(len(editor.producer_calls), 1)
        self.assertEqual(engine.unknown_producer_attempts, 1)
        self.assertEqual(engine.producer_exploration_attempts, 1)
        self.assertEqual(engine.budget.reserved, 0.)

    def test_second_candidate_gets_confirmation_after_screen_winner_fails(self):
        screen_ids = set()
        def credit(eq, task, z):
            return (1. if task in screen_ids else 0.) if 'deceptive' in eq else (.95 if 'robust' in eq else .9)
        engine, evaluator, editor = self.engine(rounds=1,
            hard=lambda eq, task, z: 'deceptive' in eq and task in screen_ids, credit=credit)
        panel, _ = self.panel(engine, evaluator)
        screen_ids.update(t.id for t in panel[:2])
        engine.search['candidates'] = 2
        def draft(slot, context, seed, max_chars=2200):
            engine.budget.charge_model(20, 0, 0)
            return Item(slot, 'deceptive' if slot == 'plan' else 'robust', 'Distinct reusable ' + slot + ' rule.')
        with mock.patch.object(editor, 'draft_revision', side_effect=draft):
            result = engine.run()
        self.assertEqual(len(engine.history[0]['candidates']), 2)
        self.assertEqual(result['frontier_updates'], 1)
        self.assertIn('robust', names(engine.champion))
        self.assertNotIn('deceptive', names(engine.champion))

    def test_selection_regression_still_blocks_a_large_training_gain(self):
        engine, evaluator, _ = self.engine(
            hard=lambda eq, task, z: task.startswith('va') and 'candidate' not in eq,
            credit=lambda eq, task, z: .9 if 'candidate' in eq else .1)
        validation = self.initialize_selection(engine, evaluator)
        panel, confirm = self.panel(engine, evaluator)
        p = Item('plan', 'candidate', 'A general reusable rule.')
        build = engine.best.with_item(p)
        choice = Choice(build, evaluator.evaluate(build, panel), 'direct:p', p)
        self.assertTrue(engine._admit(choice, evaluator.evaluate(engine.best, panel), confirm, 0))
        engine._check_selection(validation, 0, final=True)
        self.assertEqual(engine.champion.execution_id, engine.start.execution_id)

    def test_context_and_joint_audit_refuse_heldout_tasks(self):
        engine, evaluator, _ = self.engine()
        validation = evaluator.dataset.split('validation', 4)
        evaluation = evaluator.evaluate(engine.start, validation)
        with self.assertRaises(PermissionError):
            engine._context(engine.start, evaluation, 0)
        with self.assertRaises(PermissionError):
            engine._audit_admitted(Choice(engine.start, evaluation, 'invalid'), validation, 0)

    def test_uniform_routes_ignore_perfect_credit_without_changing_real_scores(self):
        engine, evaluator, _ = self.engine('uniform_route')
        panel, _ = self.panel(engine, evaluator)
        p = Item('plan', 'other', 'A different reusable plan.')
        other = engine.start.with_item(p)
        for build in (engine.start, other):
            engine.frontier[build.execution_id] = FrontierEntry(build, evaluator.evaluate(build, panel), 0, 'fixture')
        engine.slot_credit = {'plan': [100000., 1], 'edit': [-100000., 1],
                              'verify': [-100000., 1], 'inspect': [-100000., 1]}
        engine.action_credit = {kind: [100000. if i == 0 else -100000., 1]
                               for i, kind in enumerate(engine.options['support_kinds'])}
        parents = {engine._parent(r).execution_id for r in range(30)}
        first_slots = {engine._consumer_slot(r, 0, False) for r in range(30)}
        kinds = {engine._kind(r) for r in range(30)}
        self.assertEqual(parents, {engine.start.execution_id, other.execution_id})
        self.assertEqual(first_slots, set(engine.slots))
        self.assertEqual(kinds, set(engine.options['support_kinds']))
        for r in range(10):
            self.assertEqual(len({engine._consumer_slot(r, i, False) for i in range(3)}), 3)


if __name__ == '__main__':
    unittest.main()
