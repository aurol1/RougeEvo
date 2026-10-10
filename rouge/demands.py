from __future__ import annotations

from .items import Item

FACT_KINDS = ('column_profile', 'numeric_profile', 'layout_profile')
FACT_SCHEMAS = {
    'column_profile': 'list[{file_index, sheets:[{name, column_profile:[{column, headers:[{row,text}], kinds:{type:count}}]}]}]',
    'numeric_profile': 'list[{file_index, sheets:[{name, numeric_profile:[{coordinate, raw, numeric_candidate, number_format}]}]}]',
    'layout_profile': 'list[{file_index, layout_profile:[{name, rows, columns, header_signature:[{row,column,text}]}]}]',
}
FACT_USE = {
    'column_profile': 'Use column numbers and header candidates to resolve semantic columns before building cell/formula mappings. Headers are candidates, not guaranteed header rows; confirm against the task and workbook.',
    'numeric_profile': 'Use raw values and conservative numeric candidates to plan type-aware edits; validate locale and percent semantics. Preserve formulas and raw values unless the task requires conversion.',
    'layout_profile': 'Compare case dimensions and header signatures to detect changes in layout; derive mappings separately for each case rather than copying fixed coordinates.',
}
KEYWORDS = {
    'column_profile': ('column', 'header', 'table', 'lookup', 'transpose', 'match'),
    'numeric_profile': ('number', 'numeric', 'sum', 'average', 'percent', 'decimal', 'currency', 'convert'),
    'layout_profile': ('sheet', 'range', 'row', 'copy', 'transpose', 'dynamic', 'case'),
}


def declared_kinds(item: Item) -> tuple[str, ...]:
    return tuple(dict.fromkeys(n.removeprefix('facts:') for n in item.needs
                              if n.startswith('facts:') and n.removeprefix('facts:') in FACT_KINDS))


def consumer_contract(kind: str) -> str:
    if kind not in FACT_KINDS:
        raise ValueError('Unknown input-fact interface')
    return (f'For this offer, declare needs=["facts:{kind}"] in the JSON header. '
            f'Write a reusable consumer that explicitly reads the {kind} fields when observations '
            f'are present. Fact schema: {FACT_SCHEMAS[kind]}. {FACT_USE[kind]} '
            'When facts are absent, insufficient or truncated, inspect the raw workbook normally. '
            'Do not disable ordinary solving, invent a dependency, or assume observations are answers. '
            'The card must also improve a concrete procedure suggested by the training failures.')


def validate_consumer(item: Item, kind: str) -> None:
    if declared_kinds(item) != (kind,) or kind not in item.content:
        raise ValueError(f'Consumer must declare exactly facts:{kind} and describe use of {kind} in its content')


def task_affinity(kind: str, instruction: str) -> int:
    text = instruction.casefold()
    return sum(word in text for word in KEYWORDS[kind])
