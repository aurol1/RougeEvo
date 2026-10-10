from __future__ import annotations

from .core import Evaluation


def strict_gate(evaluator, build, tasks, incumbent: Evaluation, min_gain: float,
                chunk_size: int, events=None, *, allow_ties=False):

    if not tasks or set(incumbent.task_ids) != {t.id for t in tasks}:
        raise ValueError("Validation gate needs the same nonempty task set as the incumbent")
    if chunk_size <= 0:
        result = evaluator.evaluate(build, tasks)
        if set(result.task_ids) != {t.id for t in tasks} or len(result.task_ids) != len(tasks):
            raise ValueError('Validation evaluator returned incomplete or duplicate task results')
        return result, {"evaluated": len(tasks), "planned": len(tasks), "pruned": False}
    parents = incumbent.by_task()
    ordered = sorted(tasks, key=lambda t: (parents[t.id].success, t.id))
    required = incumbent.score + min_gain
    results = []
    for offset in range(0, len(ordered), chunk_size):
        batch = ordered[offset:offset + chunk_size]
        partial = evaluator.evaluate(build, batch)
        if set(partial.task_ids) != {t.id for t in batch} or len(partial.task_ids) != len(batch):
            raise ValueError('Validation evaluator returned incomplete or duplicate task results')
        results.extend(partial.results)
        upper = (sum(r.success for r in results) + len(tasks) - len(results)) / len(tasks)
        impossible = upper < required if allow_ties else upper <= required
        if len(results) < len(tasks) and impossible:
            details = {"evaluated": len(results), "planned": len(tasks), "pruned": True,
                       "score_upper_bound": upper, "required_strictly_above": required,
                       "allow_ties": allow_ties}
            if events:
                events.emit("validation_pruned", build=build.id, **details)
            return None, details
    by_id = {r.task_id: r for r in results}
    return Evaluation(build.id, [by_id[t.id] for t in tasks]), {
        "evaluated": len(tasks), "planned": len(tasks), "pruned": False}
