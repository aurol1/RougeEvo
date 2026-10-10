from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest import mock

from rouge.campaign_search import CampaignComplementSearch
from rouge.complement import Choice, WeakItem
from rouge.config import load_config, validate
from rouge.frontier_search import FrontierEntry
from rouge.items import Build, Item
from rouge.support import make_source_item, parse_producer
from tests.fixtures import names
import tests.test_campaign_search as fixture


HERE = Path(__file__).resolve().parents[1]
PRECISION = ('no_ordinary_joint', 'within_round_partners')


class CampaignPrecisionControlTests(unittest.TestCase):
    engine = fixture.CampaignSearchTests.engine
    panel = fixture.CampaignSearchTests.panel
    events = fixture.CampaignSearchTests.events
    initialize_selection = fixture.CampaignSearchTests.initialize_selection
    synergy_run = fixture.CampaignSearchTests.synergy_run

    def ordinary_fixture(self, variant='full', *, round_id=2, fresh_partner=True):
        engine, evaluator, editor = self.engine(variant,
            hard=lambda eq, task, z: {'new_plan', 'new_edit'} <= eq,
            credit=lambda eq, task, z: 1. if {'new_plan', 'new_edit'} <= eq else .1)
        panel, _ = self.panel(engine, evaluator, round_id)
        p = Item('plan', 'new_plan', 'Resolve the requested operation from the input.')
        q = Item('edit', 'new_edit', 'Apply the resolved operation and preserve other cells.')
        engine._round_items = [p, q] if fresh_partner else [p]
        engine.archive = {p.id: WeakItem(p, round_id), q.id: WeakItem(q, round_id if fresh_partner else 0)}
        base = evaluator.evaluate(engine.best, panel)
        return engine, evaluator, editor, p, q, panel, base

    def test_config_accepts_both_precision_controls_without_changing_other_settings(self):
        original = load_config(HERE / 'configs/rogueevo.json')
        for variant in PRECISION:
            with self.subTest(variant=variant):
                config = copy.deepcopy(original)
                config['complement']['campaign_ablation'] = variant
                validate(config)
                config['complement']['campaign_ablation'] = 'full'
                self.assertEqual(config, original)
        invalid = copy.deepcopy(original)
        invalid['complement']['campaign_ablation'] = 'within_round'
        with self.assertRaises(ValueError):
            validate(invalid)
        with self.assertRaises(ValueError):
            self.engine('within_round')

    def test_full_still_solves_an_archived_individually_useless_pair(self):
        result, engine, _, _ = self.synergy_run('full')
        self.assertEqual(result['final_validation_score'], 1.)
        self.assertEqual(result['frontier_updates'], 1)
        self.assertEqual(engine.permanent_pairs[0]['interaction'], 1.)
        self.assertEqual(engine.joint_screens[0]['tag'], 'archive_composition')

    def test_precision_controls_block_the_same_cross_round_synergy(self):
        for variant in PRECISION:
            with self.subTest(variant=variant):
                result, engine, evaluator, _ = self.synergy_run(variant)
                self.assertEqual(result['frontier_updates'], 0)
                self.assertEqual(result['final_validation_score'], 0.)
                self.assertFalse(engine.joint_screens)
                self.assertFalse(engine.permanent_pairs)
                self.assertEqual(engine.champion.execution_id, engine.start.execution_id)
                self.assertTrue(all(kind is None for _, _, kind in evaluator.calls))
                self.assertEqual(engine.budget.reserved, 0.)

    def test_within_round_retains_joint_of_two_individually_useless_fresh_cards(self):
        engine, evaluator, _, p, q, panel, base = self.ordinary_fixture('within_round_partners')
        screen = panel[:2]
        self.assertEqual(evaluator.evaluate(engine.best.with_item(p), screen).score, 0.)
        self.assertEqual(evaluator.evaluate(engine.best.with_item(q), screen).score, 0.)
        choices = engine._combination_choices(base, screen, panel, 2)
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0].evaluation.score, 1.)
        self.assertEqual(names(choices[0].build) & {'new_plan', 'new_edit'}, {'new_plan', 'new_edit'})
        self.assertEqual(engine.joint_screens[0]['p_birth_round'], 2)
        self.assertEqual(engine.joint_screens[0]['q_birth_round'], 2)

    def test_no_ordinary_joint_blocks_even_fresh_pairs_without_paying_joint_screen(self):
        engine, evaluator, _, _, _, panel, base = self.ordinary_fixture('no_ordinary_joint')
        before = engine.budget.spent
        calls = list(evaluator.calls)
        self.assertEqual(engine._combination_choices(base, panel[:2], panel, 2), [])
        self.assertEqual(engine.budget.spent, before)
        self.assertEqual(evaluator.calls, calls)
        self.assertFalse(engine.composition_audits)

    def test_fresh_offer_survives_archive_capacity_pruning_and_keeps_real_birth(self):
        engine, _, _, p, _, panel, base = self.ordinary_fixture('within_round_partners')
        del engine.archive[p.id]
        choices = engine._combination_choices(base, panel[:2], panel, 2)
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0].evaluation.score, 1.)
        self.assertEqual(engine.joint_screens[0]['p'], p.id)
        self.assertEqual(engine.joint_screens[0]['p_birth_round'], 2)
        self.assertEqual(engine.joint_screens[0]['q_birth_round'], 2)

    def test_birth_and_current_view_are_both_required_for_within_round_partners(self):

        engine, _, _, _, q, panel, base = self.ordinary_fixture(
            'within_round_partners', fresh_partner=False)
        engine.archive[q.id].born = 2
        self.assertEqual(engine._combination_choices(base, panel[:2], panel, 2), [])

        engine, _, _, p, q, panel, base = self.ordinary_fixture('within_round_partners')
        engine.archive[q.id].born = 0
        self.assertEqual(engine._combination_choices(base, panel[:2], panel, 2), [])

    def test_within_round_leaves_historical_capability_consumers_in_the_weak_pool(self):
        engine, evaluator, _, p, _, panel, base = self.ordinary_fixture('within_round_partners')
        old = Item('verify', 'historical_consumer', 'Use column facts if present; otherwise inspect inputs.',
                   ('facts:column_profile',))
        engine.archive[old.id] = WeakItem(old, 0)
        before = {key: (entry.item, entry.born) for key, entry in engine.archive.items()}
        engine._combination_choices(base, panel[:2], panel, 2)
        self.assertEqual(before, {key: (entry.item, entry.born) for key, entry in engine.archive.items()})
        self.assertEqual(engine.options['weak_ttl_rounds'], 8)
        self.assertIn(old, engine._weak_pool(evaluator.evaluate(engine.best, panel[:2]), panel[:2], 2))
        self.assertTrue(all(row['q'] != old.id and row['p'] != old.id for row in engine.joint_screens))

    def test_parent_installed_card_is_not_reacquired_as_a_fresh_partner(self):
        engine, evaluator, _, p, q, panel, _ = self.ordinary_fixture('within_round_partners')
        engine.best = engine.best.with_item(q)
        base = evaluator.evaluate(engine.best, panel)
        self.assertEqual(engine._combination_choices(base, panel[:2], panel, 2), [])
        self.assertIn(q.name, names(engine.best))
        self.assertEqual(engine._round_items, [p, q])

    def test_repeated_old_proposal_is_not_marked_as_a_new_card_by_the_floor(self):
        engine, evaluator, editor = self.engine('within_round_partners', rounds=1)
        old = Item('plan', 'old_plan', 'A previously proposed reusable plan.')
        engine.archive[old.id] = WeakItem(old, -1)
        engine.seen.add(old.id)
        with mock.patch.object(editor, 'draft_revision', return_value=old):
            result = engine.run()
        self.assertEqual(result['frontier_updates'], 0)
        self.assertEqual(engine._round_items, [])
        self.assertEqual(engine.archive[old.id].born, -1)
        self.assertFalse(engine.joint_screens)

    def test_completion_menu_remains_p_only_and_pq_for_both_precision_controls(self):
        signatures = []
        for variant in ('full',) + PRECISION:
            engine, evaluator, _ = self.engine(variant)
            panel, _ = self.panel(engine, evaluator, 2)
            screen = panel[:2]
            p = Item('plan', 'consumer', 'Use column facts; inspect inputs when absent.',
                     ('facts:column_profile',))
            q = make_source_item('column_profile')
            base = evaluator.evaluate(engine.best, screen)
            choices = engine._normal_pair(p, q, base, screen, 2, 'source_adoption')
            self.assertEqual([choice.tag for choice in choices], ['source_adoption:p', 'source_adoption:pq'])
            self.assertIsNotNone(parse_producer(choices[1].build.get(('inspect', ''))))
            signatures.append(([choice.build.execution_id for choice in choices], evaluator.calls,
                               engine.budget.spent, engine.composition_audits.keys()))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])

    def test_trials_positive_response_and_learned_completion_are_unchanged(self):
        signatures = []
        for variant in ('full',) + PRECISION:
            engine, evaluator, editor = self.engine(variant, credit=lambda eq, task, z: .9 if z else .1)
            panel, _ = self.panel(engine, evaluator, 2)
            screen = panel[:2]
            base = evaluator.evaluate(engine.best, panel)
            context = engine._generation_context(engine.best, base, panel, 2)
            self.assertTrue(context.fact_kind)
            self.assertIn('Paid input-only', context.chemistry)
            p = Item('plan', 'consumer', 'Use column facts; inspect inputs when absent.',
                     ('facts:column_profile',))
            projects = engine._projects([p], evaluator.evaluate(engine.best, screen), screen, panel, 2)
            self.assertEqual(len(projects), 1)
            self.assertEqual(projects[0][2], 'confirmed_positive_marginal_support')
            choices = engine._complete_project(p, 'column_profile', projects[0][3],
                evaluator.evaluate(engine.best, screen), screen, panel, 2)
            self.assertTrue(editor.producer_calls)
            self.assertTrue(any('learned_producer' in names(choice.build) for choice in choices))
            self.assertTrue(all(choice.tag.endswith((':p', ':pq')) for choice in choices))
            signatures.append((context.fact_kind, context.chemistry, engine.generation_probes,
                engine.action_credit, engine.probes, [choice.build.execution_id for choice in choices],
                evaluator.calls, engine.budget.spent, engine.budget.reserved))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])

    def test_feedback_routes_and_periodic_joint_opportunity_are_preserved(self):
        signatures = []
        for variant in ('full',) + PRECISION:
            engine, evaluator, _ = self.engine(variant)
            panel, _ = self.panel(engine, evaluator, 2)
            p = Item('plan', 'candidate_plan', 'A reusable plan.')
            q = Item('edit', 'candidate_edit', 'A reusable edit.')
            pair = engine.best.compose(p, q)
            for build in (engine.start, engine.start.with_item(p)):
                engine.frontier[build.execution_id] = FrontierEntry(
                    build, evaluator.evaluate(build, panel), 0, 'fixture')
            engine.slot_credit = {'plan': [100., 1], 'edit': [-100., 1]}
            engine.action_credit = {kind: [100. if i == 0 else -100., 1]
                for i, kind in enumerate(engine.options['support_kinds'])}
            routes = [(engine._parent(r).execution_id, engine._consumer_slot(r, 0, False), engine._kind(r))
                      for r in range(3, 12)]
            engine.best = engine.start
            engine.composition_audits[pair.execution_id] = engine.best, p, q
            ordinary = engine.best.with_item(p)
            ordinary_choice = Choice(ordinary, evaluator.evaluate(ordinary, panel[:2]), 'direct:p', p)
            joint_choice = Choice(pair, evaluator.evaluate(pair, panel[:2]), 'source_adoption:pq', p, q)
            shortlist = engine._shortlist([ordinary_choice, joint_choice], 2)
            self.assertEqual(shortlist[0].build.execution_id, pair.execution_id)
            signatures.append((routes, [choice.build.execution_id for choice in shortlist]))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])

    def test_joint_provenance_reconstructs_the_same_parent_and_real_births(self):
        engine, _, _, p, q, panel, base = self.ordinary_fixture('full', fresh_partner=False)
        choices = engine._combination_choices(base, panel[:2], panel, 2)
        self.assertEqual(len(choices), 1)
        row = engine.joint_screens[0]
        parent = Build.from_dict(row['parent_build'])
        factors = [Item.from_dict(row[key]) for key in ('p_item', 'q_item')]
        self.assertEqual(parent.execution_id, row['parent_execution_id'])
        self.assertEqual(parent.compose(*factors).execution_id, row['candidate_execution_id'])
        self.assertEqual(row['candidate_execution_id'], choices[0].build.execution_id)
        self.assertEqual((row['p_birth_round'], row['q_birth_round']), (2, 0))
        self.assertEqual((row['p'], row['q']), (p.id, q.id))


if __name__ == '__main__':
    unittest.main()
