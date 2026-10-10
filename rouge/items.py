from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

from .core import digest

CARD_SLOTS = ("inspect", "plan", "edit", "verify")
META_SLOTS = ("diagnose", "propose")
RELIC = "relic"
DOCUMENT = "document"
SLOTS = CARD_SLOTS + META_SLOTS + (RELIC, DOCUMENT)
EXECUTION_SLOTS = CARD_SLOTS + (RELIC, DOCUMENT)
SLOT_ORDER = {slot: i for i, slot in enumerate(SLOTS)}
SLOT_MEANINGS = {
    "inspect": "how the agent explores the workbook before acting",
    "plan": "how it turns the instruction and answer position into concrete cell changes",
    "edit": "how it writes values/formulas with openpyxl or pandas",
    "verify": "how it re-opens outputs and checks them before replying DONE",
    "diagnose": "how the evolver explains a failure (meta: affects evolution, not execution)",
    "propose": "how the evolver writes new items (meta: affects evolution, not execution)",
    RELIC: "an importable Python helper module (a tool) available in every sandbox",
}
CARD_CHARS = 2500
RELIC_CHARS = 6000
NAME = re.compile(r"[a-z][a-z0-9_]{1,30}")


_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def relic_functions(source: str) -> list[ast.FunctionDef]:

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Relic is not valid Python: {exc.msg} (line {exc.lineno})") from None
    for node in tree.body:
        docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not isinstance(node, _TOP_LEVEL) and not docstring:
            raise ValueError(f"Relic top level may only hold imports, functions, classes and assignments "
                             f"(line {node.lineno}); do the work inside functions")
    public = [x for x in tree.body if isinstance(x, ast.FunctionDef) and not x.name.startswith("_")]
    if not public:
        raise ValueError("Relic must define at least one public function")
    if any(not ast.get_docstring(x) for x in public):
        raise ValueError("Every public relic function needs a docstring")
    return public


def relic_api(module: str, source: str) -> str:
    lines = []
    for fn in relic_functions(source):
        doc = (ast.get_docstring(fn) or "").strip().splitlines()
        lines.append(f"  {module}.{fn.name}({ast.unparse(fn.args)}): {doc[0] if doc else ''}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Item:
    slot: str
    name: str
    content: str
    needs: tuple[str, ...] = ()

    def __post_init__(self):
        if self.slot not in SLOTS:
            raise ValueError(f"Unknown slot: {self.slot}")
        limit = (CARD_CHARS * len(CARD_SLOTS) if self.slot == DOCUMENT else
                 RELIC_CHARS if self.slot == RELIC else CARD_CHARS)
        if self.slot == "inspect" and self.content.startswith("ROUGE_INPUT_PRODUCER_V1"):


            from .support import parse_producer
            producer = parse_producer(self)
            if len(producer.instructions) > CARD_CHARS:
                raise ValueError(f"Producer instructions exceed {CARD_CHARS} characters")
            limit = 12 * CARD_CHARS + 128
        if not self.name.strip() or not self.content.strip() or len(self.content) > limit:
            raise ValueError(f"{self.slot} item needs a name and 1..{limit} characters of content")
        if len(self.name) > 80 or len(self.needs) > 8 or any(
                not isinstance(x, str) or len(x) > 300 for x in self.needs):
            raise ValueError("name <= 80 chars; needs = at most eight short requirements")
        if self.slot == RELIC:
            if not NAME.fullmatch(self.name):
                raise ValueError("Relic name must match [a-z][a-z0-9_]{1,30} (it becomes a module name)")
            relic_functions(self.content)

    @property
    def id(self) -> str:
        return digest(asdict(self))

    @property
    def key(self) -> tuple[str, str]:

        return (self.slot, self.name if self.slot == RELIC else "")

    @property
    def module(self) -> str:
        return f"relic_{self.name}"

    @property
    def label(self) -> str:
        return f"{self.slot}:{self.name}#{self.id[:6]}"

    def to_dict(self) -> dict:
        return {**asdict(self), "needs": list(self.needs)}

    @classmethod
    def from_dict(cls, value: dict) -> Item:
        needs = value.get("needs", [])
        if not isinstance(needs, list):
            raise ValueError("needs must be a JSON list")
        return cls(slot=str(value["slot"]).strip().lower(), name=str(value["name"]).strip(),
                   content=str(value["content"]), needs=tuple(str(x) for x in needs))


@dataclass(frozen=True)
class Build:

    items: tuple[Item, ...] = ()

    def __post_init__(self):
        keys = [x.key for x in self.items]
        if any(x.slot == DOCUMENT for x in self.items) and any(
                x.slot in CARD_SLOTS for x in self.items):
            raise ValueError("A monolithic document cannot be mixed with instruction cards")
        if len(set(keys)) != len(keys):
            raise ValueError("A build holds one card per slot and one relic per name")
        ordered = tuple(sorted(self.items, key=lambda x: (SLOT_ORDER[x.slot], x.name)))
        object.__setattr__(self, "items", ordered)

    def with_item(self, item: Item) -> Build:
        return Build(tuple(x for x in self.items if x.key != item.key) + (item,))

    def without(self, item: Item) -> Build:
        return Build(tuple(x for x in self.items if x.key != item.key))

    def compose(self, first: Item, second: Item) -> Build:
        if first.key == second.key:
            raise ValueError("A completion must occupy a different slot than the retained item")
        return self.with_item(first).with_item(second)

    def get(self, key: tuple[str, str]) -> Item | None:
        return next((x for x in self.items if x.key == key), None)

    def has(self, item: Item) -> bool:
        return any(x.id == item.id for x in self.items)

    @property
    def relics(self) -> tuple[Item, ...]:
        return tuple(x for x in self.items if x.slot == RELIC)

    @property
    def execution_items(self) -> tuple[Item, ...]:
        return tuple(x for x in self.items if x.slot in EXECUTION_SLOTS)

    def instructions(self, slots=CARD_SLOTS) -> str:

        parts = [f"[{x.slot}] {x.content}" for x in self.items if x.slot in slots and x.slot != RELIC]
        if DOCUMENT in slots:
            parts = [x.content if x.slot == DOCUMENT else f"[{x.slot}] {x.content}"
                     for x in self.items if x.slot in slots and x.slot != RELIC]
        if RELIC in slots and self.relics:
            api = "\n".join(relic_api(x.module, x.content) for x in self.relics)
            parts.append("[relics] Helper modules already in /work (import them; do not overwrite):\n" + api)
        return "\n\n".join(parts)

    def relic_files(self) -> dict[str, str]:
        return {f"{x.module}.py": x.content for x in self.relics}

    def to_dict(self) -> dict:
        return {"id": self.id, "execution_id": self.execution_id,
                "items": [x.to_dict() for x in self.items]}

    @classmethod
    def from_dict(cls, data: dict) -> Build:
        return cls(tuple(Item.from_dict(x) for x in data["items"]))

    @property
    def id(self) -> str:
        return digest([asdict(x) for x in self.items])

    @property
    def execution_id(self) -> str:

        return digest([self.instructions(EXECUTION_SLOTS), self.relic_files()])
