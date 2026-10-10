from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .core import (Budget, BudgetExceeded, Evaluation, Events, InfrastructureError, TaskResult,
                   digest, seed_for, write_json)
from .data import Dataset, OfficialScorer, Task
from .items import EXECUTION_SLOTS, Build, Item
from .parsing import parse_action
from .sandbox import DockerRunner
from .preview import workbook_preview
from .support import SUPPORT_KINDS, ObservationFailure, SupportProvider, parse_producer

SYSTEM = """You solve spreadsheet tasks by writing Python that runs in a Linux sandbox (no internet).
Each turn, reply with exactly one fenced block:
```python
# your code
```
Files persist in /work between turns and you see the printed output. Use openpyxl or pandas.
For EACH case_N_input.xlsx write case_N_output.xlsx with the SAME general solution; keep unrelated
content unchanged. Answers are checked by cell VALUE inside the answer position; formulas are
recalculated by LibreOffice before checking, so write plain values when unsure a function exists.
When every output exists and you have checked it, reply with the single word DONE (you may put
DONE after a final code block). Do not create step.py or a directory named recalculated.
"""


class SpreadsheetEvaluator:


    def __init__(self, dataset: Dataset, model, runner: DockerRunner, config: dict,
                 output: Path, events: Events, seed: int, budget: Budget,
                 splits: tuple[str, ...] = ("train", "validation")):
        self.dataset, self.model, self.runner, self.config = dataset, model, runner, config
        self.output, self.events, self.seed, self.budget = output, events, seed, budget
        self.scorer = OfficialScorer(dataset.scorer_path)
        self.cache: dict[tuple, TaskResult] = {}

        self.splits = tuple(splits)
        self.allowed_ids = {x for name in self.splits for x in dataset.splits[name]}

        self.explain_ids = set(dataset.splits["train"])
        self.workers = max(1, int(config.get("parallel_tasks", 1)))
        self.stats = {"fresh": 0, "errors": 0, "fresh_cost": 0.0}
        self._lock = threading.Lock()
        self.previews: dict[str, str] = {}
        self.support = SupportProvider(runner, budget, events, config)
        self.support_records: dict[tuple, list[dict]] = {}

    @property
    def task_cost(self) -> float:

        fallback = float(self.config.get("max_steps", 5)) * float(self.config.get("max_tokens", 1536)) * 2
        return self.stats["fresh_cost"] / self.stats["fresh"] if self.stats["fresh"] else fallback

    def evaluate(self, build: Build, tasks: list[Task]) -> Evaluation:
        return self._evaluate(build, tasks)

    def evaluate_supported(self, build: Build, tasks: list[Task], support_kind: str) -> Evaluation:

        if support_kind not in SUPPORT_KINDS:
            raise ValueError(f"Unknown support kind: {support_kind!r}")
        if any(t.id not in self.explain_ids for t in tasks):
            raise PermissionError("Temporary support probes may only open training tasks")
        return self._evaluate(build, tasks, support_kind)

    def check_producer_inputs(self, item: Item, tasks: list[Task]) -> list[dict]:

        if not tasks or len({t.id for t in tasks}) != len(tasks):
            raise ValueError('Producer checks require nonempty unique training tasks')
        if any(t.id not in self.explain_ids or t.id not in self.allowed_ids for t in tasks):
            raise PermissionError('Producer execution checks may only open allowed training tasks')
        producer = parse_producer(item)
        if producer is None:
            raise ValueError('Producer execution checks require an input producer')
        rows = []
        for task in tasks:
            record = self.output / 'producer_checks' / digest([item.id, task.id, time.time_ns()])
            workspace = record / 'work'
            workspace.mkdir(parents=True, exist_ok=False)
            workspace.chmod(0o777)
            for index, case in enumerate(task.cases):
                shutil.copyfile(case.input, workspace / f'case_{index}_input.xlsx')
            observation = self.support.observe('permanent', workspace, producer=producer,
                                               temporary=False)
            row = {'task_id': task.id, 'item_id': item.id, **observation.to_dict(),
                   'purpose': 'functional_producer_check_only', 'actor_executed': False}
            write_json(record / 'check.json', row)
            event = dict(row)
            event['support_kind'] = event.pop('kind')
            self.events.emit('producer_runtime_check', **event)
            rows.append(row)
        return rows

    def _key(self, build: Build, task: Task, support_kind: str | None = None) -> tuple:
        key = (build.execution_id, task.id, self.seed)
        return key if support_kind is None else key + ("input_support", support_kind)

    def support_evidence(self, build: Build, tasks: list[Task], support_kind: str,
                         limit_chars: int = 6000) -> str:

        if support_kind not in SUPPORT_KINDS:
            raise ValueError(f"Unknown support kind: {support_kind!r}")
        if any(t.id not in self.explain_ids for t in tasks):
            raise PermissionError("Support evidence may only expose training tasks")
        blocks = []
        for index, task in enumerate(tasks, 1):
            observations = self.support_records.get(self._key(build, task, support_kind), [])
            if not observations:
                blocks.append(f"Task {index}: support probe not executed.")
            for observation in observations:
                if observation["temporary"]:
                    blocks.append(f"Task {index}: {json.dumps(observation, ensure_ascii=False)}")
        return "\n".join(blocks)[:max(0, limit_chars)]

    def _evaluate(self, build: Build, tasks: list[Task], support_kind: str | None = None) -> Evaluation:
        for item in build.items:
            parse_producer(item)
        if any(t.id not in self.allowed_ids for t in tasks):
            raise PermissionError(f"This evaluator only opens {self.splits} (frozen split protection)")
        keys = [self._key(build, t, support_kind) for t in tasks]
        fresh = list({k: t for k, t in zip(keys, tasks) if k not in self.cache}.items())
        before = self.budget.spent
        failure = None
        if fresh:
            progress = len(fresh) >= int(self.config.get("progress_every", 10))
            with ThreadPoolExecutor(max_workers=min(self.workers, len(fresh))) as pool:
                futures = [(k, pool.submit(self.run_task, build, t, support_kind)) for k, t in fresh]
                for done, (key, future) in enumerate(futures, 1):
                    try:
                        self.cache[key] = future.result()
                    except (BudgetExceeded, InfrastructureError) as exc:
                        failure = failure or exc
                    if progress and (done % 10 == 0 or done == len(futures)):
                        print(f"  evaluated {done}/{len(futures)} tasks, spent {self.budget.spent:.0f}",
                              flush=True)
            with self._lock:
                self.stats["fresh"] += sum(k in self.cache for k, _ in fresh)
                self.stats["fresh_cost"] += self.budget.spent - before
        if failure is not None:
            raise failure
        self._guard()
        evaluation = Evaluation(build.id, [self.cache[k] for k in keys])
        self.events.emit("evaluation", build=build.id, task_ids=[t.id for t in tasks],
                         fresh=len(fresh), score=evaluation.score, spent=self.budget.spent,
                         support_kind=support_kind)
        return evaluation

    def _guard(self) -> None:

        fresh, errors = self.stats["fresh"], self.stats["errors"]
        if fresh >= 10 and errors / fresh > float(self.config.get("max_error_rate", 0.5)):
            raise InfrastructureError(f"{errors}/{fresh} task runs crashed inside the harness")

    def run_task(self, build: Build, task: Task, support_kind: str | None = None) -> TaskResult:
        if support_kind is not None:
            if support_kind not in SUPPORT_KINDS:
                raise ValueError(f"Unknown support kind: {support_kind!r}")
            if task.id not in self.explain_ids or task.id not in self.allowed_ids:
                raise PermissionError("Temporary support probes may only open allowed training tasks")
        key = self._key(build, task, support_kind)
        record = self.output / "executions" / digest(list(key))
        workspace = record / "work"
        if record.exists():
            try:
                shutil.rmtree(record)
            except OSError:
                record.rename(record.with_name(f"{record.name}.stale-{time.time_ns()}"))
        workspace.mkdir(parents=True)
        workspace.chmod(0o777)
        for i, case in enumerate(task.cases):
            shutil.copyfile(case.input, workspace / f"case_{i}_input.xlsx")
        for name, source in build.relic_files().items():
            (workspace / name).write_text(source, encoding="utf-8")
        trace: list[dict] = []
        observations: list[dict] = []
        training_progress = {}
        try:


            for item in build.execution_items:
                producer = parse_producer(item)
                if producer is not None:
                    observation = self.support.observe("permanent", workspace, producer=producer,
                                                       temporary=False)
                    observations.append({**observation.to_dict(), "temporary": False})
            if support_kind is not None:
                observation = self.support.observe(support_kind, workspace, temporary=True)
                observations.append({**observation.to_dict(), "temporary": True})
            with self._lock:
                self.support_records[key] = observations
            if any(not x["success"] for x in observations):
                trace.extend({"step": -1, "stage": "input_observation", **x} for x in observations)
                raise ObservationFailure("; ".join(x["error"] for x in observations if not x["success"]))
            self._play(build, task, workspace, trace, observations)
            cases, diagnostic = self._score(task, workspace, training_progress)
            feedback = "all_cases_passed" if all(cases) else "wrong_or_missing_cells"
        except (BudgetExceeded, InfrastructureError):
            write_json(record / "partial.json", {"status": "aborted", "trace": trace})
            raise
        except ObservationFailure as exc:
            cases, feedback = [False] * len(task.cases), "input_producer_failed"
            diagnostic = str(exc)[:1200]
        except Exception as exc:
            cases, feedback = [False] * len(task.cases), f"task_error:{type(exc).__name__}"
            diagnostic = f"{type(exc).__name__}: {str(exc)[:300]}"
            with self._lock:
                self.stats["errors"] += 1
            self.events.emit("task_error", task=task.id, build=build.id, error=diagnostic)
        if task.id not in self.explain_ids:
            diagnostic = ""
        trace_chars = int(self.config.get('training_trace_chars', 0))
        if task.id in self.explain_ids and trace_chars:
            from .evolution import bounded_training_trace
            short = bounded_training_trace(trace, trace_chars)
        else:
            short = json.dumps(trace[-2:], ensure_ascii=False)[-6000:]
        result = TaskResult(task.id, all(cases), cases, feedback, short, diagnostic,
                            training_progress or None)
        write_json(record / "result.json", {"build": build.id, "task": task.public(),
                                            "result": asdict(result), "trace": trace,
                                            "support_kind": support_kind, "observations": observations})
        return result

    def _play(self, build: Build, task: Task, workspace: Path, trace: list[dict],
              observations: list[dict] | None = None) -> None:
        cfg, keep = self.config, int(self.config["observation_chars"])


        rendered = []
        for item in build.items:
            producer = parse_producer(item)
            rendered.append(Item(item.slot, item.name, producer.instructions or
                                 "Use the producer's original input observations.", item.needs)
                            if producer is not None else item)
        system = SYSTEM + "\n" + Build(tuple(rendered)).instructions(EXECUTION_SLOTS)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(task.agent_view(), ensure_ascii=False)}]
        preview_chars = int(cfg.get("preview_chars", 0))
        if preview_chars:
            with self._lock:
                if task.id not in self.previews:
                    self.previews[task.id] = workbook_preview([c.input for c in task.cases], preview_chars)
                preview = self.previews[task.id]
            messages[1]["content"] += ("\nInput-only workbook preview (values are abbreviated, not answers):\n"
                                       + preview + "\nUse this survey; inspect more cells only if necessary.")
        for observation in observations or []:
            text = observation["facts"] if observation["success"] else observation["error"]
            messages[1]["content"] += ("\nInput-only producer observations (before first actor call):\n"
                                       f"{text}\nThese are original input observations, not checked answers.")
            trace.append({"step": -1, "stage": "input_observation", **observation})
        last_code = None
        for step in range(int(cfg["max_steps"])):
            response = self.model.chat(messages, budget=self.budget,
                                       seed=seed_for(self.seed, "task", task.id, step),
                                       max_tokens=int(cfg["max_tokens"]),
                                       temperature=float(cfg["temperature"]),
                                       top_p=float(cfg.get("top_p", 0.95)), purpose="task_execution")
            try:
                source, done = parse_action(response)
                feedback, code = "No code executed.", last_code
                if source.strip():
                    script = workspace / "step.py"
                    if script.is_symlink():
                        raise ValueError("step.py cannot be a symbolic link")
                    script.write_text(source, encoding="utf-8")
                    code, output = self.runner.run(workspace, ["python", "/work/step.py"],
                                                   self.budget, "task_python")
                    feedback, last_code = f"exit code {code}\n{output}", code
                trace.append({"step": step, "response": response, "returncode": code,
                              "feedback": feedback[-keep:]})
                if done and code in (None, 0):
                    return
                if done:
                    feedback += "\nThe last code failed; fix it before replying DONE."
            except ValueError as exc:
                feedback = f"Invalid reply: {exc}"
                trace.append({"step": step, "response": response, "feedback": feedback})
            messages += [{"role": "assistant", "content": response},
                         {"role": "user", "content": feedback[-keep:]}]
            messages = messages[:2] + messages[2:][-4:]

    def _score(self, task: Task, workspace: Path, progress_sink=None) -> tuple[list[bool], str]:
        self.runner.recalculate(workspace, self.budget)
        hold, start = self.budget.reserve(self.budget.tool_price), time.perf_counter()
        explain = task.id in self.explain_ids
        reveal = bool(self.config.get("reveal_expected", True))
        try:
            checks = [self.scorer.check(task, case, workspace / "recalculated" / f"case_{i}_output.xlsx",
                                        explain=explain, reveal_expected=reveal)
                      for i, case in enumerate(task.cases)]
            if explain and self.config.get('training_cell_credit', False) and progress_sink is not None:
                credits = [self.scorer.training_progress(task, case, workspace / 'recalculated' /
                            f'case_{i}_output.xlsx') for i, case in enumerate(task.cases)]
                available = bool(credits) and all(c.get('available') for c in credits)
                progress_sink.update(available=available, cases=credits,
                    score=sum(c['fraction'] for c in credits) / len(credits) if available else None,
                    scope='training_only_official_target_cell_equality_macro_case_fraction')
        finally:
            self.budget.settle_tool(hold, time.perf_counter() - start)
        notes = [f"case {i}: {msg}" for i, (ok, msg) in enumerate(checks) if not ok and msg]
        return [ok for ok, _ in checks], " | ".join(notes)[:1200]
