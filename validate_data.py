import argparse
import copy
import json
import os
from pathlib import Path

from rouge.core import write_json
from rouge.data import Dataset, OfficialScorer
from rouge.positions import parse_targets


def self_check(dataset: Dataset) -> list[dict]:
    scorer = OfficialScorer(dataset.scorer_path)
    failures = []
    for index, task in enumerate(dataset.tasks.values()):
        try:
            if not all(scorer.compare(task, case, case.answer) for case in task.cases):
                failures.append({"task_id": task.id, "reason": "reference_self_comparison_false"})
        except Exception as exc:
            failures.append({"task_id": task.id, "reason": type(exc).__name__, "detail": str(exc)})
        if (index + 1) % 50 == 0:
            print(f"Scorer compatibility: inspected {index+1}/{len(dataset.tasks)} tasks", flush=True)
    return failures


def normalized(dataset: Dataset, output: Path) -> None:
    import openpyxl

    records = list(dataset.root.rglob("dataset.json"))
    if len(records) != 1:
        raise SystemExit(f"Expected one official dataset.json under {dataset.root}, found {len(records)}")
    source = {str(r["id"]): r for r in json.loads(records[0].read_text(encoding="utf-8-sig"))}
    manifest, unresolved = copy.deepcopy(dataset.metadata), []
    for row in manifest["tasks"]:
        record = source[row["id"]]
        try:
            per_case = []
            for case in dataset.tasks[row["id"]].cases:
                with case.answer.open("rb") as stream:
                    names = openpyxl.load_workbook(stream, read_only=True).sheetnames
                per_case.append(parse_targets(record["answer_position"], record.get("answer_sheet"), names))
            if any(t != per_case[0] for t in per_case):
                raise ValueError("cases resolve to different targets")
            row["targets"] = [list(t) for t in per_case[0]]
        except Exception as exc:
            unresolved.append({"task_id": row["id"], "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
    if unresolved:
        raise SystemExit(f"{len(unresolved)} positions could not be resolved: {unresolved[:5]}")
    manifest["dataset"] += " (normalized answer positions)"
    manifest["protocol"] += ("; answer positions resolved against golden sheet names; "
                             "official cell comparison unchanged (supplementary, disclose as patched)")
    manifest["source_manifest_fingerprint"] = dataset.fingerprint
    staging = output.with_name(output.name + ".staging")
    write_json(staging, manifest)
    failures = self_check(Dataset(staging))
    if failures:
        staging.unlink()
        raise SystemExit(f"Normalized manifest fails the reference self-check on {len(failures)} tasks: "
                         f"{failures[:5]}")
    manifest["scorer_compatibility_verified"] = True
    write_json(staging, manifest)
    os.replace(staging, output)
    print(f"Normalized manifest: {len(manifest['tasks'])}/{len(manifest['tasks'])} reference self-checks pass "
          f"(NOT agent accuracy) -> {output}")


def main():
    parser = argparse.ArgumentParser(description='Audit scorer compatibility and create the compatible manifest.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/scorer_audit.json"))
    parser.add_argument("--compatible-manifest", type=Path)
    parser.add_argument("--normalized-manifest", type=Path)
    args = parser.parse_args()
    for target in (args.compatible_manifest, args.normalized_manifest):
        if target and target.exists():
            raise SystemExit(f"{target} already exists; use a new name to preserve split provenance")
        if target and target.resolve().parent != args.manifest.resolve().parent:
            raise SystemExit("Derived manifests must stay next to the original for relative paths")
    dataset = Dataset(args.manifest)
    if args.normalized_manifest:
        normalized(dataset, args.normalized_manifest)
        return
    failures = self_check(dataset)
    report = {"evidence_type": "data_scorer_self_check", "agent_performance": False,
              "manifest_fingerprint": dataset.fingerprint, "total_tasks": len(dataset.tasks),
              "compatible_tasks": len(dataset.tasks) - len(failures), "failures": failures,
              "note": "A reference compared with itself is an interface check, not a model success score."}
    write_json(args.output, report)
    if args.compatible_manifest:
        excluded = {x["task_id"] for x in failures}
        manifest = copy.deepcopy(dataset.metadata)
        manifest["dataset"] += " (scorer-compatible preliminary subset)"
        manifest["protocol"] += "; incompatible reference self-check tasks quarantined before model experiments"
        manifest["source_manifest_fingerprint"] = dataset.fingerprint
        manifest["scorer_compatibility_verified"] = True
        manifest["quarantine"] = failures
        manifest["tasks"] = [r for r in manifest["tasks"] if r["id"] not in excluded]
        manifest["splits"] = {k: [x for x in ids if x not in excluded] for k, ids in dataset.splits.items()}
        write_json(args.compatible_manifest, manifest)
    print(f"Scorer-compatible tasks: {report['compatible_tasks']}/{report['total_tasks']}; "
          f"this is NOT agent accuracy. Audit: {args.output}")


if __name__ == "__main__":
    main()
