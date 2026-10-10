from __future__ import annotations
import copy
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from rouge.config import load_config, validate
from rouge.core import Budget, Evaluation, Events, TaskResult
from rouge.data import Task
from rouge.deck import starter_build
from rouge.frontier_search import FrontierComplementSearch
from rouge.items import Item
from rouge.support import format_producer_item
HERE = Path(__file__).resolve().parents[1]

def names(build: Build) -> set[str]:
    return {x.name for x in build.execution_items}


class FakeDataset:
    def __init__(self, train=60, validation=24, test=20):
        self.tasks, self.splits = {}, {}
        for split, count in (("train", train), ("validation", validation), ("test", test)):
            ids = [f"{split[:2]}{i:03d}" for i in range(count)]
            self.splits[split] = ids
            for x in ids:
                self.tasks[x] = Task(x, f"synthetic instruction {x}", "Cell-Level Manipulation", "'S'!A1", ())
        self.metadata = {"dataset": "synthetic", "protocol": "unit test world"}
        self.fingerprint = "synthetic"

    def split(self, name, limit=0):
        ids = self.splits[name][:limit] if limit else self.splits[name]
        return [self.tasks[x] for x in ids]


class ProbeWorld:
    def __init__(self, world, budget):
        self.world, self.budget = world, budget
        self.dataset = FakeDataset(train=24, validation=8, test=8)
        self.cache, self.calls = {}, []
        self.task_cost = 100.0

    def evaluate(self, build, tasks):
        return self._evaluate(build, tasks, None)

    def evaluate_supported(self, build, tasks, kind):
        if any(t.id not in self.dataset.splits["train"] for t in tasks):
            raise PermissionError("temporary support is training only")
        return self._evaluate(build, tasks, kind)

    def support_evidence(self, build, tasks, kind, limit_chars=6000):
        return "Input-only observed structure; no golden data."

    def _evaluate(self, build, tasks, support):
        self.calls.append((build.id, [t.id for t in tasks], support))
        out = []
        for task in tasks:
            if task.id in self.dataset.splits["test"]:
                raise PermissionError("sealed test")
            key = (build.execution_id, task.id, support)
            if key not in self.cache:
                self.budget.charge_model(100, 0, 0)
                if support:
                    self.budget.charge_tool(0)
                ok = self.world(names(build), task.id, support)
                self.cache[key] = TaskResult(task.id, ok, [ok], "pass" if ok else "wrong", "trace")
            out.append(self.cache[key])
        return Evaluation(build.id, out)


class Editor:
    def __init__(self, budget, invalid=False):
        self.budget, self.invalid, self.n = budget, invalid, 0
        self.producer_calls = []

    def draft(self, mode, slot, context, seed):
        self.budget.charge_model(20, 0, 0)
        self.n += 1
        return Item(slot, "consumer_" + str(self.n), "Use input observations to make general edits.",
                    ("input structure",))

    def draft_producer(self, context, partner, seed, support_evidence=""):
        self.budget.charge_model(30, 0, 0)
        self.producer_calls.append(support_evidence)
        if self.invalid:
            return None
        return format_producer_item("learned_producer", "def observe(books):\n    return books\n")


class CreditWorld(ProbeWorld):
    def __init__(self, world, budget, credit=None):
        super().__init__(world, budget)
        self.credit = credit

    def _evaluate(self, build, tasks, support):
        result = super()._evaluate(build, tasks, support)
        if self.credit:
            for task, row in zip(tasks, result.results):
                if task.id in self.dataset.splits['train']:
                    row.training_progress = {'available': True,
                        'score': self.credit(names(build), task.id, support)}
        return result


class RevisionEditor(Editor):


    def __init__(self, budget):
        super().__init__(budget)
        self.contexts = []

    def draft_revision(self, slot, context, seed, max_chars=2200):
        self.budget.charge_model(20, 0, 0)
        self.n += 1
        self.contexts.append(context)
        kind = context.fact_kind
        content = "Derive the requested transformation; preserve useful previous procedures."
        if kind:
            content += f" Use {kind} observations; inspect raw inputs if absent."
        return Item(slot, "stage_" + slot, content, ("facts:" + kind,) if kind else ())


class SearchFixture:

    def engine(self, hard=None, credit=None, rounds=2, limit=100000, exploration=0, max_regressions=1):
        cfg = copy.deepcopy(load_config(HERE / 'tests/fixtures/synthetic_search.json'))
        cfg['budget']['development'] = limit
        cfg['search'].update(rounds=rounds, warmup_rounds=2, candidates=1, batch_size=6, confirm_size=4, validation_limit=4, validation_chunk_size=2, panel_reuse_rounds=2)
        cfg['complement'].update(frontier_guided=True, frontier_capacity=4, frontier_quality_weight=0.5, frontier_min_gain=0.02, frontier_max_regressions=max_regressions, frontier_explore_probability=exploration, validation_every=2, selection_budget_fraction=0.25, combination_offers=2, refinement_chars=900, post_warmup_candidates=2, selection_candidates=2, probe_budget_fraction=0.15, completion_budget_fraction=0.5)
        validate(cfg)
        output = Path(tempfile.gettempdir()) / ('rouge_frontier_test_' + uuid.uuid4().hex)
        output.mkdir()
        self.addCleanup(shutil.rmtree, output)
        budget = Budget(limit, tool_price=10)
        evaluator = CreditWorld(hard or (lambda equipped, task, support: False), budget, credit or (lambda equipped, task, support: 0.1))
        editor = RevisionEditor(budget)
        engine = FrontierComplementSearch(cfg, evaluator, editor, Events(output / 'events.jsonl'), output, starter_build())
        return (engine, evaluator, editor)

    def panel(self, engine, evaluator, round_id=0):
        return engine._batches(evaluator.dataset.split('train', 24), round_id)

    def initialize_selection(self, engine, evaluator):
        validation = evaluator.dataset.split('validation', 4)
        engine.champion_validation = evaluator.evaluate(engine.champion, validation)
        return validation

    def events(self, engine):
        return [json.loads(line) for line in (engine.output / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
