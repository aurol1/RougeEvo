from __future__ import annotations

from .items import Build, Item

STARTER_CARDS = {
    "inspect": (
        "survey",
        "Before editing, print a compact survey of every input workbook with openpyxl: sheet names, "
        "max_row/max_column, merged ranges, the first ~8 rows of each sheet, and the cells around the "
        "answer position (values and formulas, load once with data_only=False). Identify the header "
        "row, the data rows and which columns the instruction refers to. Do not print whole sheets."),
    "plan": (
        "cell_plan",
        "Restate the goal as concrete cell changes: which sheet, which cells inside the answer "
        "position, and the rule that produces each value. Derive the rule from the instruction and "
        "from existing example rows; it must work for every case_N_input.xlsx, not only one file. "
        "Note edge cases you saw in the survey (blank cells, text numbers, dates, trailing spaces)."),
    "edit": (
        "value_writer",
        "Load each input with openpyxl (keep formatting; never recreate the workbook), compute the "
        "answers in Python and write plain values into the answer position. Only write a formula "
        "when the task explicitly asks for one and the function exists in LibreOffice. Match the "
        "expected type: numbers as int/float, dates as datetime, text without extra spaces. Leave "
        "cells outside the answer position unchanged and save as case_N_output.xlsx."),
    "verify": (
        "reopen_check",
        "Before replying DONE, re-open every case_N_output.xlsx with openpyxl and print the values in "
        "the answer position for each case. Check that every output exists, that no expected cell is "
        "empty, that types look right, and that the same rule was applied to all cases. Fix problems "
        "first; reply DONE only after a clean check."),
}

STARTER_META = {
    "diagnose": (
        "root_cause",
        "Name the single most common root cause across the failures (e.g. wrong sheet or range, "
        "misread header, type mismatch, formula left uncomputed, rule that only fits one case, "
        "output missing). Point to the evidence line that shows it. Then state one reusable change "
        "to the harness that would prevent it; never quote task-specific answers."),
    "propose": (
        "general_fix",
        "Fix the failure pattern that appears in the most traces with the smallest general change. "
        "Prefer concrete, checkable procedures (what to print, what to compare) over vague advice. "
        "Keep what already works in the current card. Never include task IDs, file-specific values "
        "or expected answers."),
}


def starter_build(include_meta: bool = True) -> Build:
    items = [Item(slot, name, text) for slot, (name, text) in STARTER_CARDS.items()]
    if include_meta:
        items += [Item(slot, name, text) for slot, (name, text) in STARTER_META.items()]
    return Build(tuple(items))


def resolve_build(spec: str) -> Build:

    import json
    from pathlib import Path

    if spec == "starter":
        return starter_build()
    if spec == "starter-no-meta":
        return starter_build(include_meta=False)
    if spec == "empty":
        return Build()
    path = Path(spec)
    if not path.is_file():
        raise ValueError(f"Build must be starter, starter-no-meta, empty, or a JSON file: {spec}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("build") or data.get("best_build") or data
    return Build.from_dict(data)
