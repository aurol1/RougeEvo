import json
import unittest
from types import SimpleNamespace

from rouge.campaign_evolution import (CampaignProposer, CoverageEvidence, campaign_context,
    execution_evidence, unresolved_error, validate_producer_observation)
from rouge.core import Budget, BudgetExceeded, Evaluation, TaskResult
from rouge.data import Task
from rouge.deck import starter_build
from rouge.evolution import Context
from tests.test_revision_editor import RecordingEvents, ScriptedRevisionModel, reply


def result(tid, success=False, steps=None):
    return TaskResult(tid, success, [success], 'all_cases_passed' if success else 'wrong_or_missing_cells',
        json.dumps(steps or [{'step': 0, 'response': '```python\nwb.save(output)\n```',
                             'returncode': 0, 'feedback': 'exit code 0'}]),
        '' if success else 'Synthetic mismatching cells', {'available': True, 'score': float(success)})


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.build = starter_build()
        self.tasks = {str(i): Task(str(i), f'Complete synthetic instruction {i}.',
                                  'Cell-Level Manipulation', 'A1:A2', ()) for i in range(8)}
        self.selector = CoverageEvidence(self.tasks, self.tasks, 0)

    def test_long_instruction_tail_survives_whole(self):
        instruction = 'Specification. ' * 70 + 'TAIL: Preserve blanks when no matching source exists.'
        self.tasks['0'] = Task('0', instruction, 'Cell-Level Manipulation', 'A1:A2', ())
        evaluation = Evaluation(self.build.id, [result('0')])
        context = self.selector.context(self.build, evaluation, [], 0, limit_chars=2200)
        self.assertIn(instruction, context.evidence)
        self.assertIn('TAIL:', context.evidence)
        self.assertLessEqual(len(context.evidence), 2200)
        self.assertTrue(self.selector.last_selection[0]['instruction_complete'])

    def test_oversized_instruction_is_omitted_whole_not_cut(self):
        self.tasks['0'] = Task('0', 'x' * 9000 + 'END_OF_INSTRUCTION', 'Cell', 'A1', ())
        context = self.selector.context(self.build, Evaluation(self.build.id, [result('0')]), [], 0, 1000)
        self.assertIn('omitted whole', context.evidence)
        self.assertNotIn('xxx', context.evidence)
        self.assertEqual(self.selector.last_task_ids, [])
        self.assertEqual(self.selector.last_omissions[0]['task_id'], '0')
        self.assertEqual(self.selector.exposures, {})

    def test_failure_coverage_rotates_and_retains_a_success(self):
        rows = [result(str(i)) for i in range(6)] + [result('6', True)]
        evaluation = Evaluation(self.build.id, rows)
        covered = set()
        for round_id in range(3):
            context = self.selector.context(self.build, evaluation, [], round_id, 6500)
            self.assertIn('6', self.selector.last_task_ids)
            covered.update(t for t in self.selector.last_task_ids if t != '6')
            self.assertIn('success=True', context.evidence)
        self.assertEqual(covered, {str(i) for i in range(6)})

    def test_repaired_exception_does_not_count_as_current_error(self):
        repaired = result('0', steps=[
            {'step': 0, 'response': '```python\nbad()\n```', 'returncode': 1, 'feedback': 'Traceback old'},
            {'step': 1, 'response': '```python\nwb.save(output)\n```', 'returncode': 0, 'feedback': 'exit code 0'}])
        unresolved = result('1', steps=[
            {'step': 0, 'response': '```python\nbad()\n```', 'returncode': 1, 'feedback': 'Traceback current'}])
        self.assertFalse(unresolved_error(repaired))
        self.assertTrue(unresolved_error(unresolved))
        evaluation = Evaluation(self.build.id, [repaired, unresolved])
        self.selector.context(self.build, evaluation, [], 0, 2500, max_examples=1)
        self.assertEqual(self.selector.last_task_ids, ['1'])


        self.selector.context(self.build, evaluation, [], 1, 2500, max_examples=1)
        self.assertEqual(self.selector.last_task_ids, ['0'])
        text = execution_evidence(repaired, 1000)
        self.assertIn('wb.save', text)
        self.assertNotIn('Traceback old', text)

    def test_selector_rejects_nontraining_or_wrong_build(self):
        selector = CoverageEvidence(self.tasks, ['0'], 0)
        with self.assertRaises(PermissionError):
            selector.context(self.build, Evaluation(self.build.id, [result('0')]), [result('1')], 0)
        with self.assertRaises(ValueError):
            selector.context(self.build, Evaluation('another_build', [result('0')]), [], 0)

    def test_wrapper_excludes_support_other_build_validation_and_train_beyond_limit(self):
        dataset = SimpleNamespace(tasks=self.tasks,
            split=lambda name, limit: [self.tasks[str(i)] for i in range(limit)])
        execution_id = self.build.execution_id
        evaluator = SimpleNamespace(dataset=dataset,
            _key=lambda build, task, kind=None: (build.execution_id, task.id, 9),
            cache={(execution_id, '0', 9): result('0'),
                   (execution_id, '1', 9): result('1', True),
                   (execution_id, '2', 9, 'input_support', 'layout_profile'): result('2'),
                   ('other_execution', '3', 9): result('3'),
                   (execution_id, '7', 9): result('7')})
        engine = SimpleNamespace(evaluator=evaluator, tasks=self.tasks, seed=0,
            search={'train_limit': 4}, options={}, config={'proposal': {'evidence_chars': 6500}},
            events=RecordingEvents())
        context = campaign_context(engine, self.build, Evaluation(self.build.id, [result('0')]), 0)
        self.assertEqual(set(engine.evidence_exposures), {'0', '1'})
        self.assertEqual(engine.events.rows[-1]['pool_size'], 2)
        self.assertEqual(context.fact_kind, '')
        with self.assertRaises(PermissionError):
            campaign_context(engine, self.build, Evaluation(self.build.id, [result('7')]), 1)


class CampaignProposerTests(unittest.TestCase):
    def setUp(self):
        self.budget, self.events = Budget(1000), RecordingEvents()
        self.config = {'repair_attempts': 1, 'max_tokens': 400, 'temperature': .8}
        self.context = Context(starter_build(), 'Synthetic training evidence with a conditional blank output.')

    def proposer(self, replies):
        model = ScriptedRevisionModel(replies)
        return CampaignProposer(model, self.budget, self.config, self.events), model

    def test_inspect_revision_supported_without_duplicate_target_prompt(self):
        proposer, model = self.proposer([reply('inspect', 'Inspect the original inputs around unresolved mappings.', ['plan'])])
        item = proposer.draft_revision('inspect', self.context, 5)
        self.assertEqual(item.slot, 'inspect')
        target = self.context.build.get(('inspect', '')).content
        self.assertEqual(model.calls[0][0][1]['content'].count(target), 1)
        self.assertEqual(self.budget.spent, 90)

    def test_conditional_blank_constraint_is_explicit_in_prompt(self):
        proposer, model = self.proposer([reply('verify', 'Check the requested conditions before deciding whether blank cells are wrong.')])
        self.assertIsNotNone(proposer.draft_revision('verify', self.context, 5))
        system = model.calls[0][0][0]['content']
        self.assertIn('a blank output can be correct', system)
        self.assertIn('never require every target cell to be nonempty', system)
        self.assertIn('never', system.lower())
        self.assertIn(self.context.evidence, model.calls[0][0][1]['content'])

    def test_inspect_cannot_declare_fact_dependency_even_when_context_has_interface(self):
        self.context.fact_kind = 'numeric_profile'
        proposer, model = self.proposer([reply('inspect', 'Use numeric_profile.', ['facts:numeric_profile'])] * 2)
        self.assertIsNone(proposer.draft_revision('inspect', self.context, 5))
        self.assertEqual(self.budget.spent, 180)
        self.assertEqual(len(model.calls), 2)

    def test_fact_consumer_enforced_and_overlong_revision_repaired(self):
        self.context.fact_kind = 'numeric_profile'
        text = 'Use numeric_profile only when present; otherwise inspect raw original inputs.'
        proposer, model = self.proposer([reply('plan', 'x' * 2210, ['facts:numeric_profile']),
                                         reply('plan', text, ['facts:numeric_profile'])])
        item = proposer.draft_revision('plan', self.context, 11)
        self.assertEqual(item.content, text)
        self.assertEqual([c[1]['seed'] for c in model.calls], [11, 12])
        self.assertEqual(self.budget.spent, 180)
        self.assertEqual(self.events.rows[-1]['attempt'], 1)

    def test_budget_error_propagates_and_name_only_change_is_not_trial(self):
        proposer, model = self.proposer([BudgetExceeded('budget exhausted')])
        with self.assertRaises(BudgetExceeded):
            proposer.draft_revision('plan', self.context, 1)
        old = self.context.build.get(('plan', ''))
        proposer, model = self.proposer([reply('plan', old.content, old.needs, 'new_name')])
        self.assertIsNone(proposer.draft_revision('plan', self.context, 1))
        self.assertEqual(self.events.rows[-1]['event'], 'unchanged_revision')

    def test_producer_observation_rejects_truncation_but_preserves_legitimate_empty(self):
        empty = {'success': True, 'truncated': False,
                 'facts': '[{"layout_profile":[{"header_signature":[]}]}]'}
        validate_producer_observation(empty, fact_kind='layout_profile', input_has_text=False)
        with self.assertRaisesRegex(ValueError, 'header is empty'):
            validate_producer_observation(empty, fact_kind='layout_profile', input_has_text=True)
        with self.assertRaisesRegex(ValueError, 'truncated'):
            validate_producer_observation({**empty, 'truncated': True})
        with self.assertRaisesRegex(ValueError, 'failed'):
            validate_producer_observation({'success': False, 'error': 'Synthetic invalid program'})

    def test_producer_clarification_and_paid_runtime_repair_use_parent_contract(self):
        header = '```json\n{"slot":"inspect","name":"input_observer","needs":[]}\n```\n'
        program = header + '```python\ndef observe(books):\n    return [{"files": len(books)}]\n```'
        proposer, model = self.proposer([program, program])
        attempts = []
        def runtime(item):
            attempts.append(item.id)
            return [{'success': True, 'truncated': len(attempts) == 1, 'facts': '[]'}]
        item = proposer.draft_producer(self.context, self.context.build.get(('plan', '')), 3,
                                        runtime_check=runtime)
        self.assertIsNotNone(item)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(self.budget.spent, 180)
        prompt = model.calls[0][0][1]['content']
        self.assertIn('"s" for text', prompt)
        self.assertIn('Never compare kind with "string"', prompt)
        self.assertIn('truncated', model.calls[1][0][1]['content'])
        self.assertTrue(self.events.rows[-1]['runtime_checked'])


if __name__ == '__main__':
    unittest.main()
