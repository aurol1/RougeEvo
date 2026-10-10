from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()[:20]


def seed_for(seed: int, *keys: Any) -> int:

    return int(digest([seed, *keys])[:8], 16) % (2**31 - 1)


def rng_for(seed: int, *keys: Any) -> random.Random:
    return random.Random(seed_for(seed, *keys))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str),
                    encoding="utf-8")
    temp.replace(path)


class Events:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, kind: str, **data: Any) -> None:
        line = json.dumps({"event": kind, **data}, ensure_ascii=False, allow_nan=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class BudgetExceeded(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    pass


class ProposalError(ValueError):
    pass


class ContextTooLong(ValueError):
    pass


@dataclass
class Budget:

    limit: float
    tool_price: float = 256.0
    spent: float = 0.0
    reserved: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    tool_calls: int = 0
    model_seconds: float = 0.0
    tool_seconds: float = 0.0
    failed_calls: int = 0
    overrun: float = 0.0
    deadline: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        if (not math.isfinite(self.limit) or self.limit <= 0 or
                not math.isfinite(self.tool_price) or self.tool_price <= 0):
            raise ValueError("Budget and tool price must be finite and positive")

    @property
    def remaining(self) -> float:
        return self.limit - self.spent - self.reserved

    def ensure(self, cost: float) -> None:
        if self.deadline and time.monotonic() >= self.deadline:
            raise BudgetExceeded("Wall-time limit reached; keeping completed evidence")
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid operation cost")
        if cost > self.remaining + 1e-9:
            raise BudgetExceeded(f"Need {cost:.0f} budget units; {self.remaining:.0f} remain")

    def reserve(self, cost: float) -> float:
        with self._lock:
            self.ensure(cost)
            self.reserved += cost
        return cost

    def release(self, hold: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - hold)

    def settle_model(self, hold: float, prompt: int, generated: int, seconds: float,
                     failed: bool = False) -> None:
        if prompt < 0 or generated < 0:
            raise ValueError("Token usage cannot be negative")
        with self._lock:
            self.reserved = max(0.0, self.reserved - hold)
            cost = prompt + generated

            self.overrun += max(0.0, cost - hold)
            self.spent += cost
            self.prompt_tokens += prompt
            self.generated_tokens += generated
            self.model_seconds += seconds
            self.failed_calls += int(failed)

    def settle_tool(self, hold: float, seconds: float, failed: bool = False) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - hold)
            self.spent += self.tool_price
            self.tool_calls += 1
            self.tool_seconds += seconds
            self.failed_calls += int(failed)

    def charge_model(self, prompt: int, generated: int, seconds: float, failed=False) -> None:
        self.settle_model(self.reserve(prompt + generated), prompt, generated, seconds, failed)

    def charge_tool(self, seconds: float, failed=False) -> None:
        self.settle_tool(self.reserve(self.tool_price), seconds, failed)

    def snapshot(self) -> dict:
        keys = ("limit", "tool_price", "spent", "reserved", "prompt_tokens", "generated_tokens",
                "tool_calls", "model_seconds", "tool_seconds", "failed_calls", "overrun")
        with self._lock:
            return {k: getattr(self, k) for k in keys}


@dataclass
class TaskResult:
    task_id: str
    success: bool
    case_results: list[bool]
    feedback: str
    trace: str
    diagnostic: str = ""
    training_progress: dict | None = None


@dataclass
class Evaluation:
    build_id: str
    results: list[TaskResult]

    @property
    def score(self) -> float:
        if not self.results:
            raise ValueError("Cannot score an empty or incomplete evaluation")
        return sum(x.success for x in self.results) / len(self.results)

    @property
    def task_ids(self) -> list[str]:
        return [x.task_id for x in self.results]

    def by_task(self) -> dict[str, TaskResult]:
        return {x.task_id: x for x in self.results}

    def paired(self, other: Evaluation) -> list[tuple[TaskResult, TaskResult]]:
        left, right = self.by_task(), other.by_task()
        if len(left) != len(self.results) or len(right) != len(other.results) or left.keys() != right.keys():
            raise ValueError("Paired comparisons require the same unique task IDs")
        return [(left[k], right[k]) for k in sorted(left)]

    def merged(self, other: Evaluation) -> Evaluation:
        if self.build_id != other.build_id or set(self.task_ids) & set(other.task_ids):
            raise ValueError("Merge needs the same build on disjoint tasks")
        return Evaluation(self.build_id, self.results + other.results)


@dataclass(frozen=True)
class Delta:

    fixed: tuple[str, ...]
    regressed: tuple[str, ...]
    n: int

    @property
    def net(self) -> int:
        return len(self.fixed) - len(self.regressed)

    @property
    def gain(self) -> float:
        return self.net / self.n


def paired_delta(parent: Evaluation, child: Evaluation) -> Delta:
    pairs = parent.paired(child)
    return Delta(tuple(a.task_id for a, b in pairs if b.success and not a.success),
                 tuple(a.task_id for a, b in pairs if a.success and not b.success), len(pairs))


def reward(before: Evaluation, after: Evaluation, cost_units: float, cost_weight: float,
           regression_weight: float = 0.0) -> dict:

    delta = paired_delta(before, after)
    regressions = len(delta.regressed) / delta.n
    penalty = cost_weight * cost_units
    return {"gain": delta.gain, "regression_rate": regressions, "cost_units": cost_units,
            "cost_penalty": penalty, "reward": delta.gain - penalty - regression_weight * regressions}


def interaction(base: Evaluation, p: Evaluation, q: Evaluation, pq: Evaluation) -> float:

    for value in (p, q, pq):
        base.paired(value)
    return pq.score - p.score - q.score + base.score
