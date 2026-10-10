from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import evaluate as report
import report_budget as ledger
from rouge.core import BudgetExceeded, InfrastructureError, digest, write_json
from rouge.data import Case, Task
from rouge.deck import starter_build
from rouge.execution import SpreadsheetEvaluator


HERE = Path(__file__).resolve().parents[1]
TEST_ROOT = HERE / '.test_tmp/report'


class WorkspaceTest(unittest.TestCase):
    def setUp(self):

        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.directory = TEST_ROOT / uuid.uuid4().hex
        self.directory.mkdir()

    def tearDown(self):
        self.directory.resolve().relative_to(TEST_ROOT.resolve())
        shutil.rmtree(self.directory)


class ReportBudgetTests(WorkspaceTest):
    def budget(self, *, resume=False):
        return ledger.ReportBudget(self.directory / 'budget.json', 'same-report', 10000, 256, resume=resume)

    def test_reserve_is_durable_before_execution_and_settle_matches_actual_usage(self):
        budget = self.budget()
        hold = budget.reserve(100)
        self.assertEqual(json.loads(budget.path.read_text())['budget']['reserved'], 100)
        budget.settle_model(hold, 30, 20, .5)
        budget.charge_tool(.1)
        recovered = self.budget(resume=True)
        self.assertEqual(recovered.spent, 306)
        self.assertEqual(recovered.reserved, 0)
        self.assertEqual(recovered.prompt_tokens, 30)
        self.assertEqual(recovered.generated_tokens, 20)
        self.assertEqual(recovered.tool_calls, 1)
        self.assertTrue(recovered.snapshot()['exact_usage'])

    def test_unresolved_holds_are_charged_once_and_never_open_a_new_pool(self):
        budget = self.budget()
        budget.charge_model(20, 10, .1)
        budget.reserve(100)
        budget.reserve(256)
        recovered = self.budget(resume=True)
        self.assertEqual(recovered.spent, 386)
        self.assertEqual(recovered.reserved, 0)
        self.assertEqual(recovered.uncertain_recovery_cost, 356)
        self.assertFalse(recovered.snapshot()['exact_usage'])
        recovered_again = self.budget(resume=True)
        self.assertEqual(recovered_again.spent, 386)
        self.assertEqual(recovered_again.snapshot()['recovery_count'], 1)
        self.assertEqual(recovered_again.remaining, 9614)

    def test_concurrent_reserve_settle_has_no_lost_updates(self):
        budget = self.budget()
        def request(_):
            hold = budget.reserve(100)
            budget.settle_model(hold, 5, 7, .1)
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(request, range(40)))
        recovered = self.budget(resume=True)
        self.assertEqual(recovered.spent, 480)
        self.assertEqual(recovered.prompt_tokens, 200)
        self.assertEqual(recovered.generated_tokens, 280)
        self.assertEqual(recovered.reserved, 0)

    def test_persistence_failure_aborts_before_an_operation_can_begin(self):
        budget = self.budget()
        with mock.patch.object(ledger, 'durable_json', side_effect=OSError('disk full')):
            with self.assertRaises(InfrastructureError):
                budget.reserve(100)
        with self.assertRaises(InfrastructureError):
            budget.reserve(1)

        self.assertEqual(self.budget(resume=True).spent, 0)

    def test_failed_settle_keeps_last_reservation_for_conservative_recovery(self):
        budget = self.budget()
        hold = budget.reserve(100)
        with mock.patch.object(ledger, 'durable_json', side_effect=OSError('disk full')):
            with self.assertRaises(InfrastructureError):
                budget.settle_model(hold, 30, 20, .1)
        recovered = self.budget(resume=True)
        self.assertEqual(recovered.spent, 100)
        self.assertEqual(recovered.uncertain_recovery_cost, 100)

    def test_identity_limit_and_counter_tampering_are_rejected(self):
        budget = self.budget()
        for identity, limit in [('other-report', 10000), ('same-report', 11000)]:
            with self.subTest(identity=identity, limit=limit), self.assertRaises(ValueError):
                ledger.ReportBudget(budget.path, identity, limit, 256, resume=True)
        data = json.loads(budget.path.read_text())
        data['budget']['spent'] = 17
        write_json(budget.path, data)
        with self.assertRaises(ValueError):
            self.budget(resume=True)

    def test_directory_lock_prevents_two_writers_and_releases_on_exit(self):
        with ledger.ReportLock(self.directory):
            with self.assertRaises(InfrastructureError):
                with ledger.ReportLock(self.directory):
                    self.fail('Second writer acquired the report directory')
        with ledger.ReportLock(self.directory):
            pass


class StubDataset:
    def __init__(self, root: Path, count=199):
        self.fingerprint = 'fixed-cpu-dataset'
        self.metadata = {'dataset': 'CPU-fixture', 'protocol': 'CPU-fixture',
                         'scorer_compatibility_verified': True}
        self.scorer_path = root / 'scorer.py'
        input_path, answer = root / 'input.xlsx', root / 'answer.xlsx'
        input_path.write_bytes(b'CPU fixture; no real spreadsheet grading')
        answer.write_bytes(b'CPU fixture')
        self.tasks = [Task(f'test{i:03}', 'Do the synthetic task.', 'Cell-Level Manipulation',
                           "'S'!A1", (Case(input_path, answer),)) for i in range(count)]
        self.splits = {'test': [t.id for t in self.tasks], 'train': ['train'], 'validation': ['validation']}

    def split(self, split):
        if split != 'test':
            raise AssertionError('Report opened a non-test split')
        return self.tasks


class StubModel:
    name = 'CPU-STUB-NOT-PERFORMANCE-EVIDENCE'
    def __init__(self):
        self.calls = []

    def chat(self, messages, *, budget, **kwargs):
        hold = budget.reserve(10)
        self.calls.append(kwargs['seed'])
        budget.settle_model(hold, 4, 3, .001)
        return 'DONE'


class StubRunner:
    def check(self):
        pass

    def describe(self):
        return {'image': 'CPU stub', 'image_id': 'fixed-image', 'libreoffice': 'CPU stub'}


class CPUReportEvaluator(SpreadsheetEvaluator):
    calls = []
    def evaluate(self, build, tasks):
        self.calls.append([task.id for task in tasks])
        return super().evaluate(build, tasks)

    def _score(self, task, workspace, training_progress=None):
        self.budget.charge_tool(.001)
        return [int(task.id[4:]) % 3 == 0], ''


class FrozenReportTests(WorkspaceTest):
    def setUp(self):
        super().setUp()
        self.build = starter_build()
        self.dataset = StubDataset(self.directory)
        self.model = StubModel()
        self.config = {'seed': 0, 'model': {'backend': 'hf', 'path': 'no-model'},
                       'dataset_manifest': 'no-manifest', 'docker': {'image': 'no-docker'},
                       'budget': {'development': 1500000, 'test': 6000000, 'tool_price': 256},
                       'execution': {'max_steps': 1, 'max_tokens': 10, 'temperature': 0,
                                     'observation_chars': 100, 'parallel_tasks': 1}}
        self.args = SimpleNamespace(out_dir=self.directory / 'out', build=self.directory / 'build.json',
            origin=self.directory / 'origin.json', development_evidence=self.directory / 'development.json',
            development_source=None, resume=False, budget=6000000, deadline_unix=None,
            max_seconds=None, label='cpu_protocol_test')
        write_json(self.args.build, self.build.to_dict())
        write_json(self.args.origin, {'search_hardware_cohort': 'V100_preexisting',
                                     'original_model': {'name': 'actual-source-model', 'placement': 'V100'},
                                     'label': 'base'})
        write_json(self.args.development_evidence, {'source': 'none', 'method': 'base',
                                                  'spent': 0, 'limit': 0, 'tool_price': 256})
        CPUReportEvaluator.calls = []

    def run_report(self):
        runtime = {'host': 'CPU test', 'gpus': ['CPU stub; never claimed as actual hardware'],
                   'source_hashes': {'rouge/execution.py': 'same-executor', 'evaluate.py': 'same-helper'}}
        with mock.patch.object(report, 'Dataset', return_value=self.dataset), \
             mock.patch.object(report, 'DockerRunner', return_value=StubRunner()), \
             mock.patch.object(report, 'provenance', return_value=runtime), \
             mock.patch.object(report, 'execution_protocol', return_value={'fingerprint': 'same-protocol'}), \
             mock.patch.object(report, 'make_model', return_value=self.model) as factory, \
             mock.patch('rouge.execution.OfficialScorer'), \
             mock.patch.object(report, 'SpreadsheetEvaluator', CPUReportEvaluator):
            result = report.run(self.config, self.args)
            self.assertEqual(factory.call_count, 1)
        return result

    def test_one_model_four_fixed_batches_full_score_and_true_origin(self):
        result = self.run_report()
        self.assertEqual([len(tasks) for tasks in CPUReportEvaluator.calls], [50, 50, 50, 49])
        self.assertEqual(result['n_tasks'], 199)
        self.assertEqual(result['n_pending'], 0)
        self.assertAlmostEqual(result['score'], 67 / 199)
        self.assertEqual(result['budget']['spent'], 199 * 263)
        self.assertEqual(result['development']['source'], 'none')
        self.assertEqual(result['origin']['data']['original_model']['placement'], 'V100')
        self.assertEqual(result['model']['served_name'], self.model.name)
        self.assertEqual(result['runtime']['host'], 'CPU test')
        self.assertEqual(len(self.model.calls), 199)

    def test_completed_resume_reuses_result_json_without_model_calls_or_budget_reset(self):
        before = self.run_report()
        self.args.resume = True
        CPUReportEvaluator.calls = []
        self.model.calls = []
        after = self.run_report()
        self.assertEqual(CPUReportEvaluator.calls, [])
        self.assertEqual(self.model.calls, [])
        self.assertEqual(after['results'], before['results'])
        self.assertEqual(after['budget']['spent'], before['budget']['spent'])
        self.assertEqual(len(after['restored_task_ids']), 199)
        self.assertEqual(after['invocation'], 2)


    def test_budget_abort_preserves_completed_and_pending_without_inventing_failures(self):
        self.args.budget = 550
        result = self.run_report()
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['n_tasks'], 2)
        self.assertEqual(result['n_failures_completed'], 1)
        self.assertEqual(result['n_pending'], 197)
        self.assertIsNone(result['score'])
        self.assertEqual(result['budget']['spent'], 547)
        self.assertTrue(result['partial'])
        self.args.resume = True
        self.model.calls = []
        after = self.run_report()
        self.assertEqual(after['n_tasks'], 2)
        self.assertEqual(after['budget']['spent'], 547)
        self.assertEqual(self.model.calls, [])

    def test_elapsed_deadline_stops_before_any_task_operation(self):
        self.args.deadline_unix = 1
        result = self.run_report()
        self.assertEqual(result['status'], 'deadline_reached')
        self.assertEqual(result['n_tasks'], 0)
        self.assertEqual(result['n_pending'], 199)
        self.assertEqual(result['n_failures_completed'], 0)
        self.assertEqual(result['budget']['spent'], 0)
        self.assertEqual(self.model.calls, [])
        self.assertIsNone(result['score'])

    def test_interrupted_batch_resume_completes_remaining_tasks_from_durable_files(self):
        normal_chat = self.model.chat
        def interrupted(*args, **kwargs):
            if len(self.model.calls) >= 2:
                raise BudgetExceeded('CPU interruption after two completed tasks')
            return normal_chat(*args, **kwargs)
        with mock.patch.object(self.model, 'chat', side_effect=interrupted):
            before = self.run_report()
        self.assertEqual(before['n_tasks'], 2)
        self.assertEqual(before['n_pending'], 197)
        self.assertIsNone(before['score'])
        self.assertEqual(before['budget']['spent'], 526)

        self.args.resume = True
        self.model.calls = []
        CPUReportEvaluator.calls = []
        after = self.run_report()
        self.assertEqual([len(tasks) for tasks in CPUReportEvaluator.calls], [48, 50, 50, 49])
        self.assertEqual(len(self.model.calls), 197)
        self.assertEqual(after['n_tasks'], 199)
        self.assertEqual(after['n_pending'], 0)
        self.assertEqual(after['budget']['spent'], 52337)
        self.assertEqual(after['restored_task_ids'], ['test000', 'test001'])
        self.assertAlmostEqual(after['score'], 67 / 199)

    def test_fake_completed_rows_never_become_a_valid_report(self):
        self.model.name = report.FAKE_MODEL
        result = self.run_report()
        self.assertEqual(result['n_tasks'], 199)
        self.assertEqual(result['n_pending'], 0)
        self.assertTrue(result['complete_task_coverage'])
        self.assertFalse(result['valid_report'])
        self.assertTrue(result['partial'])
        self.assertIsNone(result['score'])
        self.assertEqual(result['evidence_type'], 'fake_test_server')

    def test_origin_from_another_build_is_rejected(self):
        write_json(self.args.origin, {'execution_id': 'wrong-execution'})
        with mock.patch.object(report, 'make_model') as factory:
            with self.assertRaises(ValueError):
                report.run(self.config, self.args)
            factory.assert_not_called()

    def test_resume_missing_budget_refuses_to_make_completed_work_free(self):
        self.run_report()
        (self.args.out_dir / 'budget_checkpoint.json').unlink()
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'without its durable budget'):
            self.run_report()

    def test_resume_before_any_execution_can_initialize_the_first_budget(self):
        self.args.resume = True
        self.args.out_dir.mkdir()
        result = self.run_report()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['budget']['spent'], 52337)

    def test_resume_rejects_changed_plan_before_model_loading(self):
        self.run_report()
        self.args.resume = True
        self.config['seed'] = 1
        with mock.patch.object(report, 'Dataset', return_value=self.dataset), \
             mock.patch.object(report, 'provenance', return_value={'source_hashes': {}}), \
             mock.patch.object(report, 'make_model') as factory:
            with self.assertRaises(ValueError):
                report.run(self.config, self.args)
            factory.assert_not_called()

    def test_invalid_or_temporary_cache_is_rejected_instead_of_counted_or_rerun(self):
        self.run_report()
        path = self.args.out_dir / 'executions' / digest([self.build.execution_id, 'test000', 0]) / 'result.json'
        original = json.loads(path.read_text())
        for mutate in ('inconsistent', 'temporary', 'training', 'task_identity'):
            data = copy.deepcopy(original)
            if mutate == 'inconsistent':
                data['result']['success'] = False
            elif mutate == 'temporary':
                data['support_kind'] = 'structure'
            elif mutate == 'training':
                data['result']['diagnostic'] = 'grader detail'
            else:
                data['task']['id'] = 'wrong-task'
            write_json(path, data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                report.cached_result(path, self.build, self.dataset.tasks[0])

    def test_manifest_must_have_all199_and_audited_scorer(self):
        self.dataset.tasks.pop()
        with self.assertRaises(ValueError):
            report.fixed_batches(self.dataset)
        self.dataset = StubDataset(self.directory)
        self.dataset.metadata['scorer_compatibility_verified'] = False
        with self.assertRaises(ValueError):
            report.fixed_batches(self.dataset)

    def test_true_source_search_preserves_model_and_cost_and_checks_build(self):
        source = self.directory / 'search/final_info.json'
        data = {'mode': 'search', 'method': 'complement', 'partial': False, 'test_evaluated': False,
                'best_build': self.build.to_dict(), 'label': 'full', 'status': 'completed',
                'evidence_type': 'local_model_run', 'model': {'placement': 'V100-original'},
                'dataset_fingerprint': self.dataset.fingerprint,
                'budget': {'spent': 1234, 'limit': 1500000, 'tool_price': 256, 'reserved': 0}}
        write_json(source, data)
        dev = report.development_record(self.config, self.build, source=source.parent)
        self.assertEqual(dev['source'], 'search')
        self.assertEqual(dev['spent'], 1234)
        self.assertEqual(dev['model'], {'placement': 'V100-original'})
        for field, value in [('best_build', {'items': []}), ('partial', True), ('test_evaluated', True)]:
            bad = copy.deepcopy(data)
            bad[field] = value
            write_json(source, bad)
            with self.subTest(field=field), self.assertRaises(ValueError):
                report.development_record(self.config, self.build, source=source)

    def test_declared_cost_is_not_promoted_to_source_search_and_base_is_verified(self):
        data = {'source': 'declared', 'method': 'external_method', 'spent': 1200,
                'limit': 1500000, 'tool_price': 256, 'notes': 'Original V100 receipt'}
        write_json(self.args.development_evidence, data)
        dev = report.development_record(self.config, self.build, declared=self.args.development_evidence)
        self.assertEqual(dev['source'], 'declared')
        self.assertEqual(dev['spent'], 1200)
        write_json(self.args.development_evidence, {'source': 'none', 'method': 'base',
                                                  'spent': 0, 'limit': 0, 'tool_price': 256})
        from rouge.items import Build
        with self.assertRaises(ValueError):
            report.development_record(self.config, Build(), declared=self.args.development_evidence)


if __name__ == '__main__':
    unittest.main()
