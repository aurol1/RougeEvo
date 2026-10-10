from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from rouge.core import rng_for, write_json

# Find the public SpreadsheetBench Verified 400 release and its official evaluator.
# Obtain spreadsheetbench_verified_400.tar.gz and evaluation/evaluation.py at this revision.
SCORER_COMMIT = "49b73a94775fb489063f60ca1865e3a650079a79"
SCORER_SHA256 = "4ae77cee8df01d1f34684fceab972810d696886533d33be2e89373de6b4d3de3"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 1024**3:
            raise ValueError("Unexpectedly large dataset archive")
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isfile() and member.size > 128 * 1024**2:
                raise ValueError(f"Unexpectedly large archive member: {member.name}")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot extract {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def make_manifest(root: Path, scorer: Path, train: int, val: int, seed: int) -> dict:
    metadata = list(root.rglob("dataset.json"))
    if len(metadata) != 1:
        raise ValueError(f"Expected one dataset.json, found {len(metadata)}")
    records = json.loads(metadata[0].read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("Official dataset.json must contain a task list")
    tasks = []
    for row in records:
        task_id = str(row["id"])
        task_dir = (metadata[0].parent / row["spreadsheet_path"]).resolve()
        if not task_dir.is_relative_to(root.resolve()):
            raise ValueError("Task directory escapes data root")
        inputs = sorted(task_dir.glob("*_init.xlsx"))
        input_suffix, answer_suffix = "_init.xlsx", "_golden.xlsx"
        if not inputs:
            inputs = sorted(task_dir.glob("*_input.xlsx"))
            input_suffix, answer_suffix = "_input.xlsx", "_answer.xlsx"
        if not inputs and (task_dir / "initial.xlsx").is_file():
            inputs = [task_dir / "initial.xlsx"]
            input_suffix, answer_suffix = "initial.xlsx", "golden.xlsx"
        if not inputs:
            raise ValueError(f"No workbook inputs for task {task_id}")
        cases = []
        for source in inputs:
            answer = source.with_name(source.name.removesuffix(input_suffix) + answer_suffix)
            if not answer.is_file() and len(inputs) == 1:
                possible = [p for p in task_dir.glob("*.xlsx")
                            if "golden" in p.stem.lower() or p.stem.lower().endswith("answer")]
                if len(possible) == 1:
                    answer = possible[0]
            if not answer.is_file():
                raise ValueError(f"Missing answer workbook for {source}")
            cases.append({"input": source.relative_to(root).as_posix(),
                          "answer": answer.relative_to(root).as_posix(),
                          "input_sha256": sha256(source), "answer_sha256": sha256(answer)})
        position = row["answer_position"]
        sheet = row.get("answer_sheet")
        if sheet:
            position = ",".join(part if "!" in part else f"'{sheet}'!{part.strip()}"
                                for part in position.split(","))
        tasks.append({"id": task_id, "instruction": row["instruction"],
                      "instruction_type": row["instruction_type"],
                      "answer_position": position, "source_answer_position": row["answer_position"],
                      "source_exclude_note": row.get("exclude"), "cases": cases})
    ids = sorted(t["id"] for t in tasks)
    if len(set(ids)) != len(ids) or min(train, val) < 1 or train + val >= len(ids):
        raise ValueError("Require unique IDs and nonempty train/validation/test splits")
    rng_for(seed, "split").shuffle(ids)
    return {"schema_version": 1, "dataset": "SpreadsheetBench Verified 400",
            "protocol": "task-level custom split; not an official leaderboard split",
            "split_seed": seed, "scorer": scorer.relative_to(root).as_posix(),
            "scorer_sha256": sha256(scorer), "scorer_commit": SCORER_COMMIT,
            "tasks": tasks, "splits": {"train": ids[:train], "validation": ids[train:train+val],
                                        "test": ids[train+val:]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare SpreadsheetBench data and the pinned evaluator.')
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--archive", type=Path, help="Local SpreadsheetBench Verified 400 archive")
    parser.add_argument("--scorer", type=Path, help="Local official evaluation.py at the pinned revision")
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise SystemExit("manifest.json already exists; use a new output directory to change splits")
    archive = args.archive.resolve() if args.archive else root / "verified_400.tar.gz"
    scorer = root / "official_evaluator.py"
    supplied_scorer = args.scorer.resolve() if args.scorer else scorer
    if not archive.is_file() or not supplied_scorer.is_file():
        raise SystemExit("Obtain the official Verified 400 archive and pinned evaluation.py first, "
                         "then pass --archive and --scorer. See THIRD_PARTY.md for search terms.")
    if sha256(supplied_scorer) != SCORER_SHA256:
        raise SystemExit(f"Evaluator checksum mismatch. Obtain evaluation/evaluation.py at revision {SCORER_COMMIT}.")
    if supplied_scorer != scorer:
        if scorer.exists() and sha256(scorer) != SCORER_SHA256:
            raise SystemExit("A different official_evaluator.py already exists; use a new output directory")
        if not scorer.exists():
            shutil.copyfile(supplied_scorer, scorer)
    safe_extract(archive, root)
    manifest = make_manifest(root, scorer, args.train, args.val, args.seed)
    manifest["archive_sha256"] = sha256(archive)
    manifest["source_reference"] = "SpreadsheetBench Verified 400, user-supplied official release"
    write_json(root / "manifest.json", manifest)
    (root / "SOURCE.txt").write_text(
        "SpreadsheetBench data and official evaluator: CC BY-SA 4.0.\n"
        "Source: official SpreadsheetBench Verified 400 release and evaluator.\n"
        "Consult the upstream distribution for its full attribution and license notices.\n"
        f"Evaluator revision: {SCORER_COMMIT}\nData archive SHA256: {manifest['archive_sha256']}\n",
        encoding="utf-8")
    print(json.dumps({"tasks": len(manifest["tasks"]),
                      "splits": {k: len(v) for k, v in manifest["splits"].items()},
                      "manifest": str(root / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
