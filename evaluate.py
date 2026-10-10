from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from experiment import FAKE_MODEL, execution_protocol, model_identity, provenance, sha256
from rouge.config import load_config, resolve_path
from rouge.core import BudgetExceeded, Events, InfrastructureError, TaskResult, digest, write_json
from rouge.data import Dataset
from rouge.deck import starter_build
from rouge.execution import SpreadsheetEvaluator
from rouge.items import Build
from rouge.model import make_model
from rouge.sandbox import DockerRunner
from report_budget import ReportBudget, ReportLock, durable_json


SCHEMA = 1
BATCH_SIZES = (50, 50, 50, 49)
TEST_BUDGET = 6_000_000.0


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def frozen_build(path: Path) -> Build:
    value = json_object(path)
    record = value.get('build') or value.get('best_build') or value
    build = Build.from_dict(record)
    for key, actual in (('id', build.id), ('execution_id', build.execution_id)):
        if key in record and record[key] != actual:
            raise ValueError(f'Frozen build has a noncanonical {key}')
    return build


def development_record(config: dict, build: Build, *, declared: Path | None = None,
                       source: Path | None = None) -> dict:

    if bool(declared) == bool(source):
        raise ValueError('Specify exactly one of --development-evidence and --development-source')
    path = Path(declared or source).resolve()
    if source and path.is_dir():
        path = path / 'final_info.json'
    data = json_object(path)
    if source:
        if (path.parent / 'INVALID_INFRASTRUCTURE.json').exists():
            raise ValueError('Development search is marked invalid infrastructure')
        if data.get('mode') != 'search' or 'best_build' not in data:
            raise ValueError('Development source must be a search final_info.json')
        if data.get('partial') is not False or data.get('test_evaluated') is not False:
            raise ValueError('Development search must be finished and have never evaluated test')
        source_build = Build.from_dict(data['best_build'])
        if (source_build.to_dict() != build.to_dict() or
                data['best_build'].get('id') != source_build.id or
                data['best_build'].get('execution_id') != source_build.execution_id):
            raise ValueError('Development search best_build differs from the frozen build')
        if data.get('evidence_type') == 'fake_test_server' or data.get('model', {}).get('served_name') == FAKE_MODEL:
            raise ValueError('Fake development search is not experimental evidence')
        pool = data['budget']
        if pool.get('reserved') != 0:
            raise ValueError('Development search has unresolved budget reservations')
        record = {'source': 'search', 'method': data.get('method'), 'label': data.get('label'),
                  'spent': pool['spent'], 'limit': pool['limit'], 'tool_price': pool['tool_price'],
                  'status': data.get('status'), 'model': data.get('model'),
                  'dataset_fingerprint': data.get('dataset_fingerprint'),
                  'source_execution_protocol': data.get('execution_protocol')}
    else:
        kind = data.get('source', 'declared')
        if kind not in ('none', 'declared'):
            raise ValueError('Declared evidence must use source:declared or the true Base source:none')
        record = {**data, 'source': kind}
        if kind == 'none':
            if (data.get('method') != 'base' or data.get('spent') != 0 or
                    data.get('limit') not in (None, 0) or
                    build.execution_id != starter_build().execution_id):
                raise ValueError('Only the unchanged Base build may claim zero no-search development')
    for key in ('spent', 'tool_price'):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid development cost: {key}')
    if record['tool_price'] != config['budget']['tool_price']:
        raise ValueError('Development evidence uses a different tool price')
    if record['source'] != 'none':
        value = record.get('limit')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Development evidence requires its actual positive budget limit')
    if not isinstance(record.get('method'), str) or not record['method'].strip():
        raise ValueError('Development evidence requires its true method')
    return {**record, 'file': str(path), 'sha256': sha256(path)}


def panel_spec(panel):
    if panel == 'test199':
        return 'test', 0, 199, (50, 50, 50, 49)
    if panel == 'dev80':
        return 'validation', 8, 80, (40, 40)
    raise ValueError('Unknown frozen panel: ' + str(panel))


def fixed_batches(dataset, panel='test199') -> list[list]:
    split, skip, count, sizes = panel_spec(panel)
    tasks = dataset.split(split)[skip:skip + count]
    if len(tasks) != count or len({t.id for t in tasks}) != count:
        raise ValueError('Incomplete or duplicate frozen panel')
    if not dataset.metadata.get('scorer_compatibility_verified'):
        raise ValueError('Dataset has not passed the pinned scorer audit')
    if panel == 'dev80':
        ids = {t.id for t in tasks}
        excluded = set(dataset.splits['train'] + dataset.splits['test'] + dataset.splits['validation'][:8])
        if ids & excluded:
            raise ValueError('Development report overlaps optimizer or test panels')
    batches, start = [], 0
    for size in sizes:
        batches.append(tasks[start:start + size])
        start += size
    return batches


def cached_result(path: Path, build: Build, task) -> TaskResult:

    data = json_object(path)
    if data.get('build') != build.id or data.get('task') != task.public():
        raise ValueError(f'Cached result identity mismatch: {path}')
    if 'support_kind' not in data or data['support_kind'] is not None:
        raise ValueError(f'Temporary capability result cannot enter frozen reporting: {path}')
    observations = data.get('observations')
    if (not isinstance(observations, list) or any(not isinstance(x, dict) or
            x.get('temporary') is not False for x in observations)):
        raise ValueError(f'Cached result has incomplete or temporary observations: {path}')
    row = data.get('result')
    if not isinstance(row, dict) or row.get('task_id') != task.id:
        raise ValueError(f'Cached result has a different task: {path}')
    cases = row.get('case_results')
    if (not isinstance(cases, list) or len(cases) != len(task.cases) or
            any(type(x) is not bool for x in cases) or type(row.get('success')) is not bool or
            row['success'] != all(cases)):
        raise ValueError(f'Cached result has incomplete or inconsistent verdicts: {path}')
    if row.get('diagnostic', '') or row.get('training_progress'):
        raise ValueError(f'Training-only grader details found in a frozen test result: {path}')
    for key in ('feedback', 'trace'):
        if not isinstance(row.get(key), str):
            raise ValueError(f'Cached result is missing {key}: {path}')
    return TaskResult(task.id, row['success'], cases, row['feedback'], row['trace'])


def restore_results(evaluator, build: Build, tasks: list, output: Path) -> list[str]:
    restored = []
    for task in tasks:
        key = (build.execution_id, task.id, evaluator.seed)
        path = output / 'executions' / digest(list(key)) / 'result.json'
        if path.is_file():
            result = cached_result(path, build, task)
            evaluator.cache[key] = result
            restored.append(task.id)
    evaluator.stats['fresh'] = len(restored)
    evaluator.stats['errors'] = sum(result.feedback.startswith('task_error:')
                                    for result in evaluator.cache.values())
    evaluator.stats['fresh_cost'] = evaluator.budget.spent
    return restored


def report_progress(common: dict, evaluator, build: Build, tasks: list, *, status: str,
                    error: str | None, started: float, previous_wall: float,
                    invocation: int, restored: list[str]) -> dict:
    done = {key[1]: value for key, value in evaluator.cache.items()
            if key[0] == build.execution_id and len(key) == 3 and key[2] == evaluator.seed}
    results = [done[t.id] for t in tasks if t.id in done]
    complete = len(results) == len(tasks)
    budget = evaluator.budget.snapshot()
    valid = (complete and status == 'completed' and budget['spent'] <= budget['limit'] + 1e-9
             and common['evidence_type'] == 'local_model_run')
    return {**common, 'status': status, 'error': error, 'partial': not valid,
            'complete_task_coverage': complete,
            'valid_report': valid, 'split': common['split'], 'limit': common['limit'], 'skip_first': common['skip_first'],
            'build': build.to_dict(), 'build_id': build.id, 'execution_id': build.execution_id,
            'score': sum(r.success for r in results) / len(tasks) if valid else None,
            'n_tasks': len(results), 'n_planned': len(tasks),
            'n_successes_completed': sum(r.success for r in results),
            'n_failures_completed': sum(not r.success for r in results),
            'n_pending': len(tasks) - len(results),
            'pending_task_ids': [t.id for t in tasks if t.id not in done],
            'budget': budget, 'evaluator': dict(evaluator.stats),
            'invocation': invocation, 'restored_task_ids': restored,
            'wall_seconds': previous_wall + time.monotonic() - started,
            'results': [{'task_id': r.task_id, 'success': r.success,
                         'case_results': r.case_results, 'feedback': r.feedback[:300]} for r in results]}


def run(config: dict, args) -> dict:
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ReportLock(output):
        return _run(config, args, output)


def _run(config: dict, args, output: Path) -> dict:
    started = time.monotonic()
    if not math.isfinite(args.budget) or args.budget <= 0:
        raise ValueError('Report budget must be finite and positive')
    deadline_unix = getattr(args, 'deadline_unix', None)
    max_seconds = getattr(args, 'max_seconds', None)
    if deadline_unix is not None and max_seconds is not None:
        raise ValueError('Use only one deadline argument')
    for value in (deadline_unix, max_seconds):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Deadline/max-seconds must be finite and positive')
    if max_seconds is not None:
        deadline_unix = time.time() + max_seconds
    build_path = args.build.resolve()
    build = frozen_build(build_path)
    origin_path = args.origin.resolve()
    origin = {'file': str(origin_path), 'sha256': sha256(origin_path), 'data': json_object(origin_path)}
    origin_id = origin['data'].get('execution_id')
    if origin_id is not None and origin_id != build.execution_id:
        raise ValueError('Origin execution_id differs from the frozen build')
    for key, build_key in (('original_report', 'build'), ('original_search', 'best_build')):
        source_record = origin['data'].get(key)
        if source_record is not None:
            if not isinstance(source_record, dict) or build_key not in source_record:
                raise ValueError(f'Origin {key} is missing its frozen build')
            if Build.from_dict(source_record[build_key]).to_dict() != build.to_dict():
                raise ValueError(f'Origin {key} differs from the frozen build')
    development = development_record(config, build, declared=args.development_evidence,
                                     source=args.development_source)
    dataset = Dataset(Path(config['dataset_manifest']))
    panel = getattr(args, 'panel', 'test199')
    split, skip, count, batch_sizes = panel_spec(panel)
    batches = fixed_batches(dataset, panel)
    tasks = [task for batch in batches for task in batch]
    if (development.get('dataset_fingerprint') is not None and
            development['dataset_fingerprint'] != dataset.fingerprint):
        raise ValueError('Development evidence and report use different dataset manifests')
    runtime = provenance(args)

    relevant_sources = {name: value for name, value in runtime['source_hashes'].items()
                        if name.startswith('rouge/') or name in
                        ('experiment.py', 'evaluate.py', 'report_budget.py')}
    plan = {'schema_version': SCHEMA, 'mode': 'frozen_' + panel, 'split': split,
            'task_ids': [t.id for t in tasks], 'batch_sizes': list(batch_sizes),
            'budget_limit': args.budget, 'tool_price': config['budget']['tool_price'],
            'seed': config['seed'], 'build': build.to_dict(),
            'build_source_sha256': sha256(build_path), 'origin_sha256': origin['sha256'],
            'development': development, 'dataset_fingerprint': dataset.fingerprint,
            'config': config, 'source_hashes': relevant_sources,
            'test_score_requires_all_tasks': True, 'temporary_support': False}
    plan_path = output / 'report_plan.json'
    if args.resume and plan_path.exists():
        previous = json_object(plan_path)
        if previous.get('plan') != plan or previous.get('fingerprint') != fingerprint(plan):
            raise ValueError('Frozen report plan changed; cannot reuse this cache or budget')
    elif any(path.exists() for path in (plan_path, output / 'budget_checkpoint.json',
                                        output / 'executions', output / 'final_info.json',
                                        output / 'report_runtime.json', output / 'events.jsonl')):
        raise ValueError('Report output contains evidence without a resumable matching plan')
    else:
        durable_json(plan_path, {'plan': plan, 'fingerprint': fingerprint(plan)})
        write_json(output / 'build.json', build.to_dict())
        write_json(output / 'config.json', config)
        write_json(output / 'origin.json', origin)
        write_json(output / 'development.json', development)
    budget_path = output / 'budget_checkpoint.json'
    if args.resume and not budget_path.exists() and any(path.exists() for path in
            (output / 'executions', output / 'final_info.json', output / 'report_runtime.json', output / 'events.jsonl')):
        raise ValueError('Report execution evidence exists without its durable budget; refusing a free restart')


    budget = ReportBudget(budget_path, fingerprint(plan), args.budget,
                          float(config['budget']['tool_price']), resume=args.resume and budget_path.exists())
    if deadline_unix is not None:
        budget.deadline = time.monotonic() + max(0.0, deadline_unix - time.time())
    events = Events(output / 'events.jsonl')
    runner = DockerRunner(config['docker'], events)
    runner.check()
    sandbox = runner.describe()
    runtime.update(docker=sandbox, dataset_fingerprint=dataset.fingerprint)
    protocol = execution_protocol(config, runner, sandbox, dataset)

    model = make_model(config['model'], events)
    identity = model_identity(config, model)
    runtime.update(model=identity, execution_protocol=protocol)
    runtime_file = output / 'report_runtime.json'
    history = json_object(runtime_file) if args.resume and runtime_file.exists() else {'invocations': []}
    if history['invocations']:
        first = history['invocations'][0]
        if first['model'] != identity or first['execution_protocol'] != protocol:
            raise ValueError('Runtime model or sandbox protocol changed on resume')
    history['invocations'].append(runtime)
    write_json(runtime_file, history)
    write_json(output / 'provenance.json', runtime)
    evaluator = SpreadsheetEvaluator(dataset, model, runner, config['execution'], output, events,
                                     int(config['seed']), budget, splits=(split,))
    restored = restore_results(evaluator, build, tasks, output) if args.resume else []
    previous_wall = 0.0
    if args.resume and (output / 'final_info.json').is_file():
        previous_wall = float(json_object(output / 'final_info.json').get('wall_seconds', 0))
    common = {'schema_version': SCHEMA, 'label': args.label or origin['data'].get('label') or build_path.parent.name,
              'mode': 'evaluate', 'seed': int(config['seed']), 'model': identity,
              'split': split, 'skip_first': skip, 'limit': count, 'panel': panel,
              'evidence_type': 'fake_test_server' if identity.get('served_name') == FAKE_MODEL else 'local_model_run',
              'dataset_name': dataset.metadata['dataset'], 'protocol': dataset.metadata['protocol'],
              'dataset_fingerprint': dataset.fingerprint, 'execution_protocol': protocol,
              'development': development, 'origin': origin, 'runtime': runtime,
              'source_hashes': runtime['source_hashes'], 'report_plan_fingerprint': fingerprint(plan),
              'report_protocol': {'name': 'frozen_' + panel, 'batch_sizes': list(batch_sizes),
                                  'independent_budget': True, 'score_requires_all_tasks': True,
                                  'deadline_unix': deadline_unix, 'temporary_support': False},
              'build_source': str(build_path), 'build_source_sha256': sha256(build_path)}
    status, error = 'running', None

    def save():
        report = report_progress(common, evaluator, build, tasks, status=status, error=error,
                                 started=started, previous_wall=previous_wall,
                                 invocation=len(history['invocations']), restored=restored)
        write_json(output / 'final_info.json', report)
        write_json(output / 'progress.json', {key: report[key] for key in
                   ('status', 'n_tasks', 'n_planned', 'n_pending', 'pending_task_ids', 'budget', 'score')})
        return report

    save()
    try:
        for index, batch in enumerate(batches):
            pending = [task for task in batch if (build.execution_id, task.id, evaluator.seed) not in evaluator.cache]
            events.emit('frozen_report_batch', batch_index=index, planned=len(batch),
                        pending=len(pending), task_ids=[t.id for t in batch])
            if pending:
                budget.ensure(0.0)
                evaluator.evaluate(build, pending)
            save()
        evaluator._guard()
        status = 'completed' if budget.spent <= budget.limit + 1e-9 else 'budget_overrun'
    except BudgetExceeded as exc:
        status, error = 'deadline_reached' if budget.deadline and time.monotonic() >= budget.deadline else 'budget_exhausted', str(exc)
    except InfrastructureError as exc:
        status, error = 'interrupted_infrastructure', str(exc)
    except KeyboardInterrupt:
        status, error = 'interrupted', 'KeyboardInterrupt'
    except BaseException as exc:
        status, error = 'interrupted_error', f'{type(exc).__name__}: {exc}'
        save()
        raise
    report = save()
    events.emit('frozen_report_status', status=status, n_tasks=report['n_tasks'],
                n_pending=report['n_pending'], score=report['score'], spent=budget.spent)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description='Evaluate a frozen build on dev80 or test199, with resumable accounting.')
    result.add_argument('--config', required=True, type=Path)
    result.add_argument('--build', required=True, type=Path, help='Frozen build JSON; no symbolic build or search')
    result.add_argument('--origin', required=True, type=Path, help='Original provenance JSON, preserved verbatim')
    development = result.add_mutually_exclusive_group(required=True)
    development.add_argument('--development-evidence', type=Path, help='True declared costs, or Base source:none')
    development.add_argument('--development-source', type=Path, help='Source search final_info.json or its directory')
    result.add_argument('--out-dir', required=True, type=Path)
    result.add_argument('--label')
    result.add_argument('--panel', choices=('test199', 'dev80'), default='test199')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--budget', type=float, default=TEST_BUDGET)
    deadline = result.add_mutually_exclusive_group()
    deadline.add_argument('--deadline-unix', type=float)
    deadline.add_argument('--max-seconds', type=float)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    config['dataset_manifest'] = resolve_path(config['dataset_manifest'])
    config['model']['path'] = resolve_path(config['model']['path'])
    if config['model'].get('tokenizer'):
        config['model']['tokenizer'] = resolve_path(config['model']['tokenizer'])
    report = run(config, args)
    print(f"{report['label']}: {report['status']}, {report['n_tasks']}/{report['n_planned']} complete, "
          f"{report['n_pending']} pending, score={report['score']}, spent={report['budget']['spent']:.0f}", flush=True)
    return 0 if report['valid_report'] else 2


if __name__ == '__main__':
    sys.exit(main())
