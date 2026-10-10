import json
import unittest

from rouge.core import Budget, ContextTooLong
from rouge.deck import starter_build
from rouge.evolution import Context, Proposer, REFINEMENT_MARKER
from rouge.items import Build, CARD_CHARS, Item


class RecordingEvents:
    def __init__(self):
        self.rows = []

    def emit(self, event, **fields):
        self.rows.append({'event': event, **fields})


class ScriptedRevisionModel:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def chat(self, messages, *, budget, **kwargs):
        self.calls.append((list(messages), kwargs))
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        budget.charge_model(50, 40, 0)
        return result


def reply(slot, content, needs=(), name='revised_card'):
    header = {'slot': slot, 'name': name, 'needs': list(needs)}
    return '```json\n' + json.dumps(header) + '\n```\n```text\n' + content + '\n```'


class RevisionEditorTests(unittest.TestCase):
    def setUp(self):
        self.budget = Budget(1000)
        self.events = RecordingEvents()
        self.config = {'repair_attempts': 1, 'max_tokens': 400, 'temperature': 0.8}
        self.context = Context(starter_build(), 'Synthetic training failure: tuple unpacking error.')

    def proposer(self, replies):
        model = ScriptedRevisionModel(replies)
        return Proposer(model, self.budget, self.config, self.events), model

    def test_existing_ordinary_dependencies_are_accepted_without_repair(self):
        for needs in [(), ('inspect',), ('inspect', 'plan'), ('edit', 'verify')]:
            with self.subTest(needs=needs):
                proposer, model = self.proposer([
                    reply('edit', 'Iterate over worksheet rows, unpack their cells, then write each case.', needs)])
                item = proposer.draft_revision('edit', self.context, 21)
                self.assertIsNotNone(item)
                self.assertEqual(item.needs, needs)
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(model.calls[0][1]['purpose'], 'propose_revision')
                self.assertEqual(self.events.rows[-1]['mode'], 'revision')

    def test_ordinary_fact_dependency_and_missing_card_dependency_are_rejected(self):
        contexts = [(self.context, ('facts:numeric_profile',)),
                    (Context(Build((Item('plan', 'only_card', 'Plan every case.'),)), 'Failure.'),
                     ('inspect',))]
        for context, needs in contexts:
            with self.subTest(needs=needs):
                proposer, model = self.proposer([reply('plan', 'Use a verified operation mapping.', needs)] * 2)
                self.assertIsNone(proposer.draft_revision('plan', context, 1))
                self.assertEqual(len(model.calls), 2)
                self.assertIn('existing ordinary card', self.events.rows[-1]['error'])

    def test_fact_consumer_without_declaration_is_rejected(self):
        self.context.fact_kind = 'numeric_profile'
        content = 'Use numeric_profile to inspect types; inspect raw original inputs if unavailable.'
        proposer, model = self.proposer([reply('plan', content, ('inspect',))] * 2)
        self.assertIsNone(proposer.draft_revision('plan', self.context, 9))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.budget.spent, 180)
        self.assertIn('facts:numeric_profile', self.events.rows[-1]['error'])

    def test_valid_fact_consumer_preserves_interface_and_raw_fallback_prompt(self):
        self.context.fact_kind = 'column_profile'
        self.context.chemistry = 'Synthetic original-input observation contrast.'
        content = ('Diagnose ambiguous columns. Use column_profile header candidates to resolve '
                   'the mapping, confirm against each original workbook, and inspect raw inputs '
                   'if the observations are missing or incomplete.')
        proposer, model = self.proposer([reply('plan', content, ('facts:column_profile',))])
        item = proposer.draft_revision('plan', self.context, 9)
        self.assertEqual(item.needs, ('facts:column_profile',))
        system = model.calls[0][0][0]['content']
        self.assertIn('Retain this consumer interface', system)
        self.assertIn('facts are absent, fail or are incomplete', system)
        self.assertIn('ORIGINAL input', system)
        self.assertIn(self.context.chemistry, model.calls[0][0][1]['content'])

    def test_overlong_revision_is_repaired_and_every_attempt_is_paid(self):
        content = 'Derive a separate source-to-output mapping for every input case.'
        proposer, model = self.proposer([reply('plan', 'x' * 2210), reply('plan', content)])
        item = proposer.draft_revision('plan', self.context, 3, max_chars=2200)
        self.assertEqual(item.content, content)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.budget.spent, 180)
        self.assertEqual([c[1]['seed'] for c in model.calls], [3, 4])
        self.assertIn('at most 2200', model.calls[1][0][-1]['content'])
        self.assertEqual(self.events.rows[-1]['attempt'], 1)

    def test_absolute_card_limit_is_enforced_even_with_larger_requested_limit(self):
        proposer, model = self.proposer([reply('edit', 'x' * (CARD_CHARS + 1)),
                                         reply('edit', 'Use actual input data to compute all requested rows.')])
        item = proposer.draft_revision('edit', self.context, 3, max_chars=4000)
        self.assertLessEqual(len(item.content), CARD_CHARS)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.budget.spent, 180)
        self.assertIn('at most 2500', model.calls[0][0][0]['content'])

    def test_full_revision_can_remove_bad_core_and_does_not_accumulate_old_patch(self):
        old = Item('plan', 'bad_plan', 'Always copy a guessed value.' + REFINEMENT_MARKER +
                   'Repeat the same unsuccessful code.')
        self.context.build = self.context.build.with_item(old)
        content = ('Diagnose repeated guesses. Derive the requested operation from original inputs; '
                   'when a code error occurs, revise the failed decision before executing again.')
        proposer, model = self.proposer([reply('plan', content, ('inspect',))])
        item = proposer.draft_revision('plan', self.context, 8)
        self.assertEqual(item.content, content)
        self.assertNotIn('guessed value', item.content)
        self.assertNotIn(REFINEMENT_MARKER, item.content)
        self.assertIn(old.content, model.calls[0][0][1]['content'])

    def test_name_only_change_is_discarded_without_execution_cost(self):
        old = Item('edit', 'original_name', 'Write every requested case.', ('inspect', 'plan'))
        self.context.build = self.context.build.with_item(old)
        proposer, model = self.proposer([
            reply('edit', old.content, ('plan', 'inspect'), name='new_name')])
        item = proposer.draft_revision('edit', self.context, 4)
        self.assertIsNone(item)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(self.budget.spent, 90)
        self.assertEqual(self.budget.tool_calls, 0)
        self.assertEqual(self.events.rows[-1]['event'], 'unchanged_revision')
        self.assertFalse(any(r['event'] == 'proposal' for r in self.events.rows))

    def test_inspect_is_not_a_revision_target_and_model_is_never_called(self):
        proposer, model = self.proposer([])
        with self.assertRaisesRegex(ValueError, 'plan, edit or verify'):
            proposer.draft_revision('inspect', self.context, 4)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.budget.spent, 0)

    def test_missing_target_and_invalid_limit_do_not_call_the_model(self):
        proposer, model = self.proposer([])
        with self.assertRaisesRegex(ValueError, 'existing instruction card'):
            proposer.draft_revision('plan', Context(Build(), 'Failure.'), 4)
        for limit in [0, -1, 2.5, True]:
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'positive integer'):
                proposer.draft_revision('plan', self.context, 4, max_chars=limit)
        self.assertEqual(model.calls, [])

    def test_context_overflow_is_recorded_without_execution(self):
        proposer, model = self.proposer([ContextTooLong('Synthetic context overflow')])
        self.assertIsNone(proposer.draft_revision('verify', self.context, 4))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(self.budget.spent, 0)
        self.assertEqual(self.events.rows[-1]['event'], 'invalid_revision')
        self.assertEqual(self.events.rows[-1]['mode'], 'revision')


if __name__ == '__main__':
    unittest.main()
