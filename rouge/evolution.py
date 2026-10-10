from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass

from .core import Budget, ContextTooLong, Events, Evaluation
from .items import CARD_CHARS, CARD_SLOTS, RELIC, RELIC_CHARS, SLOT_MEANINGS, Build, Item, relic_api
from .parsing import parse_item, strip_thinking

RELIC_PACKAGES = {"openpyxl", "pandas", "numpy"}

SYSTEM = f"""You improve the harness of a frozen spreadsheet agent. The agent is the same language model
as you. It solves each task by writing Python (openpyxl/pandas) in an offline sandbox over a few turns,
guided by harness cards (one per slot) and relic modules (Python helpers it can import).
A change is kept only if it raises the pass rate on held-out tasks, so write general, checkable
procedures. Never include task IDs, file names, concrete cell values or expected answers.

Slots:
{chr(10).join(f"  {s}: {SLOT_MEANINGS[s]}" for s in CARD_SLOTS + (RELIC,))}

Reply format: first a JSON header in a ```json block, then the content in ONE more fenced block.
```json
{{"slot": "<slot>", "name": "<short_name>", "needs": ["<what this item relies on, if anything>"]}}
```
Card content: plain instructions, at most {CARD_CHARS} characters, in a ```text block.
Relic content: a ```python module of at most {RELIC_CHARS} characters. Top level may only contain imports,
constants, functions and classes; every public function needs a docstring; import only the Python
standard library, openpyxl, pandas or numpy. The agent imports it as relic_<name>."""

GOALS = {
    "fresh": "Write a NEW item for the target slot that fixes the most common failure in the evidence.",
    "forge": "REVISE the item below. Keep what works, fix the weaknesses the evidence shows. Keep its slot"
             " (and its name if it is a relic).",
    "complete": "The PARTNER item below looks weak on its own; it may be missing a prerequisite. Write an"
                " item for the target slot that supplies what the partner needs so the two work together.",
}


@dataclass
class Context:

    build: Build
    evidence: str
    chemistry: str = ""
    diagnosis: str = ""
    fact_kind: str = ""


REFINEMENT_MARKER = "\n\n[RogueEvo refinement]\n"


def bounded_training_trace(trace: list[dict], limit: int) -> str:
    steps = [s for s in trace if s.get('step', -1) >= 0]
    if not steps:
        return '[]'
    room = max(20, (limit - 100 * len(steps)) // len(steps))
    out = []
    for step in steps:
        response = str(step.get('response', ''))
        take = room * 3 // 4
        if len(response) > take:
            response = response[:take // 2] + '\n[...]\n' + response[-take // 2:]
        out.append({'step': step.get('step'), 'returncode': step.get('returncode'),
                    'response': response, 'feedback': str(step.get('feedback', ''))[-room // 4:]})
    encoded = json.dumps(out, ensure_ascii=False)

    while len(encoded) > limit:
        for step in out:
            step['response'] = step['response'][:len(step['response']) // 2]
            step['feedback'] = step['feedback'][-len(step['feedback']) // 2:] if step['feedback'] else ''
        encoded = json.dumps(out, ensure_ascii=False)
        if all(len(s['response']) < 2 and len(s['feedback']) < 2 for s in out):
            return '[]' if len(encoded) > limit else encoded
    return encoded


def trace_excerpt(trace: str, limit: int = 900) -> str:

    try:
        steps = json.loads(trace)
    except (ValueError, TypeError):
        return trace[-limit:]
    if not isinstance(steps, list):
        return trace[-limit:]
    steps = [s for s in steps if isinstance(s, dict) and s.get('step', -1) >= 0]
    if not steps:
        return trace[-limit:]
    coded = [s for s in steps if '```' in str(s.get('response', ''))]
    errors = [s for s in steps if s.get('returncode') not in (None, 0) or
              'Traceback' in str(s.get('feedback', ''))]
    chosen = (coded[:1] + errors[:1] + coded[-1:]) or steps[-1:]
    chosen = list({s.get('step'): s for s in chosen}.values())
    each = max(1, limit // len(chosen))
    blocks = []
    for step in chosen:
        response = str(step.get('response', ''))
        room = max(1, each * 2 // 3)
        code = response if len(response) <= room else response[:room // 2] + '\n[...]\n' + response[-room // 2:]
        blocks.append(f"step={step.get('step')} response:\n{code}\nfeedback:\n" +
                      str(step.get('feedback', ''))[-max(1, each // 3):])
    return '\n'.join(blocks)[:limit]


def focused_evidence(tasks: dict, evaluation: Evaluation, rng, limit_chars: int) -> str:

    failed = [r for r in evaluation.results if not r.success]
    passed = [r for r in evaluation.results if r.success]
    rng.shuffle(failed)
    rng.shuffle(passed)
    selected = failed[:2] + passed[:1]
    if not selected:
        return '(no evidence)'
    each = max(1, limit_chars // len(selected))
    blocks = []
    for index, result in enumerate(selected, 1):
        task = tasks[result.task_id]
        text = (f"Training example {index}: success={result.success}; {task.instruction_type}; "
                f"answer position={task.answer_position}\nInstruction: {task.instruction[:500]}\n"
                f"Outcome: {result.feedback}\nTraining diagnostic: {result.diagnostic[:300]}\n"
                f"Training cell credit: {result.training_progress}\n"
                f"Executed procedure:\n{trace_excerpt(result.trace, max(200, each // 2))}")
        blocks.append(text[:each])
    return '\n\n'.join(blocks)[:limit_chars]


def render_build(build: Build) -> str:
    lines = []
    for item in build.execution_items:
        if item.slot == RELIC:
            lines.append(f"[relic {item.name}] API:\n{relic_api(item.module, item.content)}")
        else:
            lines.append(f"[{item.slot}] {item.name}: {item.content}")
    return "\n".join(lines) or "(empty harness: the agent has only the base system prompt)"


def evidence(tasks: dict, evaluation: Evaluation, rng, limit_chars: int) -> str:

    failed = [x for x in evaluation.results if not x.success]
    passed = [x for x in evaluation.results if x.success]
    rng.shuffle(failed)
    rng.shuffle(passed)
    blocks, used = [], 0
    for index, result in enumerate(failed + passed[:1], 1):
        task = tasks[result.task_id]
        text = (f"### Task {index} ({'PASSED' if result.success else 'FAILED'}; {task.instruction_type}; "
                f"answer at {task.answer_position})\nInstruction: {task.instruction[:700]}\n"
                f"Outcome: {result.feedback}" + (f"\nGrader: {result.diagnostic[:500]}" if result.diagnostic else "")
                + f"\nLast steps (tail): {result.trace[-700:]}")
        if blocks and used + len(text) > limit_chars:
            break
        blocks.append(text[:limit_chars])
        used += len(text)
    return "\n\n".join(blocks) or "(no evidence)"


def check_relic_imports(source: str) -> None:
    allowed = set(getattr(sys, "stdlib_module_names", ())) | RELIC_PACKAGES
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Relics cannot use relative imports")
            names = [node.module or ""]
        for name in names:
            root = name.split(".")[0]
            if root.startswith("relic_"):
                raise ValueError("A relic cannot import another relic (builds must stay composable)")
            if root not in allowed:
                raise ValueError(f"Relic imports {root!r}, which the sandbox does not provide")


class Proposer:
    def __init__(self, model, budget: Budget, config: dict, events: Events):
        self.model, self.budget, self.config, self.events = model, budget, config, events

    def _chat(self, messages, seed: int, max_tokens: int, purpose: str) -> str:
        return self.model.chat(messages, budget=self.budget, seed=seed, max_tokens=max_tokens,
                               temperature=float(self.config["temperature"]), purpose=purpose)

    def draft(self, mode: str, slot: str, context: Context, seed: int, *, target: Item | None = None,
              partner: Item | None = None) -> Item | None:

        if mode not in GOALS:
            raise ValueError(f"Unknown proposal mode {mode}")
        system = SYSTEM
        guide = context.build.get(("propose", ""))
        if guide is not None:
            system += f"\n\nImprover guidance:\n{guide.content}"
        parts = [f"GOAL: {GOALS[mode]}", f"TARGET SLOT: {slot}",
                 f"CURRENT HARNESS:\n{render_build(context.build)}"]
        if target is not None:
            parts.append(f"ITEM TO REVISE ({target.slot} {target.name}):\n{target.content}")
        if partner is not None:
            needs = f"\nIt says it needs: {'; '.join(partner.needs)}" if partner.needs else ""
            parts.append(f"PARTNER ({partner.slot} {partner.name}):\n{partner.content[:2000]}{needs}")
        if context.diagnosis:
            parts.append(f"DIAGNOSIS OF THE FAILURES:\n{context.diagnosis}")
        if context.chemistry:
            parts.append(f"INTERFACE GUIDANCE / OBSERVED TRAINING EVIDENCE:\n{context.chemistry}")
        parts.append(f"EVIDENCE (training tasks, current harness):\n{context.evidence}")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": "\n\n".join(parts)}]
        for attempt in range(1 + int(self.config["repair_attempts"])):
            try:
                raw = self._chat(messages, seed + attempt, int(self.config["max_tokens"]), f"propose_{mode}")
            except ContextTooLong as exc:
                self.events.emit("invalid_proposal", mode=mode, slot=slot, error=str(exc))
                return None
            try:
                item = self._validate(parse_item(raw), mode, slot, target, partner)
                if context.fact_kind and slot in ("plan", "edit", "verify"):
                    from .demands import validate_consumer
                    validate_consumer(item, context.fact_kind)
                self.events.emit("proposal", mode=mode, slot=slot, item=item.to_dict(), attempt=attempt)
                return item
            except (ValueError, KeyError, TypeError) as exc:
                self.events.emit("invalid_proposal", mode=mode, slot=slot, attempt=attempt,
                                 error=str(exc)[:500], response=raw[-2000:])
                messages = messages + [
                    {"role": "assistant", "content": strip_thinking(raw)[-3000:]},
                    {"role": "user", "content": f"That reply could not be used: {str(exc)[:300]}. Reply "
                     f"again with the ```json header (slot {slot!r}) and exactly one fenced content block."}]
        return None

    @staticmethod
    def _validate(item: Item, mode: str, slot: str, target: Item | None, partner: Item | None) -> Item:
        if item.slot != slot:
            raise ValueError(f"The header slot must be {slot!r}, got {item.slot!r}")
        if mode == "forge" and target is not None and slot == RELIC and item.name != target.name:
            item = Item(item.slot, target.name, item.content, item.needs)
        if mode == "complete" and partner is not None and item.key == partner.key:
            raise ValueError("The completion must occupy a different slot (or relic name) than the partner")
        if item.slot == RELIC:
            check_relic_imports(item.content)
        return item

    def draft_refinement(self, slot: str, context: Context, seed: int,
                         max_chars: int = 900) -> Item | None:

        target = context.build.get((slot, ''))
        if target is None:
            raise ValueError('Refinement requires an existing instruction card')
        core = target.content.split(REFINEMENT_MARKER, 1)[0]
        limit = min(max_chars, CARD_CHARS - len(core) - len(REFINEMENT_MARKER))
        if limit < 100:
            self.events.emit('invalid_refinement', slot=slot, error='No room for a bounded refinement')
            return None
        system = SYSTEM + (f"\nReturn only the NEW refinement, at most {limit} characters, "
                           "as the content block. It is appended after the retained core card; "
                           "do not repeat or contradict that core. Replace the previous refinement "
                           "while preserving its useful procedures. Improve the actual failed "
                           "decision, not generic advice or a scorer-matching output rule. "
                           "Never equate output values with input numeric candidates. Input "
                           "facts are observations, not target values. Avoid task-specific coordinates.")
        if not context.fact_kind:
            system += '\nThis is ordinary warmup: needs=[]; do not invent an input-fact dependency.'
        user = (f"TARGET SLOT: {slot}\nCURRENT HARNESS:\n{render_build(context.build)}\n\n"
                f"TRAINING EVIDENCE:\n{context.evidence}\n\n"
                f"INTERFACE GUIDANCE / OBSERVED TRAINING CONTRAST:\n{context.chemistry}")
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        for attempt in range(1 + int(self.config['repair_attempts'])):
            try:
                raw = self._chat(messages, seed + attempt, int(self.config['max_tokens']), 'propose_refinement')
            except ContextTooLong as exc:
                self.events.emit('invalid_refinement', slot=slot, error=str(exc))
                return None
            try:
                patch = self._validate(parse_item(raw), 'forge', slot, target, None)
                if len(patch.content) > limit or REFINEMENT_MARKER.strip() in patch.content:
                    raise ValueError(f'Return only a refinement of at most {limit} characters')
                if context.fact_kind:
                    from .demands import validate_consumer
                    validate_consumer(patch, context.fact_kind)
                elif patch.needs:
                    raise ValueError('Ordinary warmup refinement must have needs=[]')
                item = Item(slot, patch.name, core + REFINEMENT_MARKER + patch.content, patch.needs)
                self.events.emit('proposal', mode='refinement', slot=slot, item=item.to_dict(), attempt=attempt)
                return item
            except (ValueError, KeyError, TypeError) as exc:
                self.events.emit('invalid_refinement', slot=slot, attempt=attempt,
                                 error=str(exc)[:500], response=raw[-2000:])
                messages += [{'role': 'assistant', 'content': strip_thinking(raw)[-2200:]},
                             {'role': 'user', 'content': f'Rejected: {str(exc)[:300]}. Return the '
                              f'JSON header and one text refinement, at most {limit} characters.'}]
        return None

    def draft_revision(self, slot: str, context: Context, seed: int,
                       max_chars: int = 2200) -> Item | None:

        if slot not in ('plan', 'edit', 'verify'):
            raise ValueError('Revision target must be plan, edit or verify')
        target = context.build.get((slot, ''))
        if target is None:
            raise ValueError('Revision requires an existing instruction card')
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
            raise ValueError('Revision max_chars must be a positive integer')
        limit = min(max_chars, CARD_CHARS)
        examples = {
            'plan': ('Diagnose a misread operation, then specify how to resolve the sheet, '
                     'source-to-output mapping and edge cases before choosing an algorithm. '
                     'For example, distinguish a per-row running count from a grouped summary '
                     'by checking the instruction and workbook structure.'),
            'edit': ('Diagnose a faulty transformation or code API, then specify a concrete '
                     'writing procedure. For example, when the operation deletes rows, first '
                     'collect the rows that satisfy its conditions, preserve protected rows, '
                     'and delete in reverse order; use this only when deletion is requested.'),
            'verify': ('Diagnose why a successful code exit can still leave a wrong workbook. '
                       'For example, check output coverage and the instruction-derived '
                       'invariants against the original inputs, and repair the specific '
                       'failure rather than repeat the same code or merely say DONE.'),
        }
        ordinary_needs = tuple(s for s in ('inspect', 'plan', 'edit', 'verify')
                               if context.build.get((s, '')) is not None)
        system = SYSTEM + (
            f'\nWrite the COMPLETE replacement for the {slot} card, at most {limit} '
            'characters. Preserve useful procedures and the interfaces the other cards '
            'rely on, but you may revise the original core and remove duplicate, harmful '
            'or unsupported rules. Do not append a patch or repeat an old patch marker. '
            'A concise, substantive revision is preferable to padding. State a brief '
            'failure diagnosis and a concrete decision procedure, with conditional '
            'routines only when the evidence justifies them. '
            f'Target-slot example of a procedure, not an answer template: {examples[slot]} '
            'Read ORIGINAL input workbooks, derive rules separately for every case, '
            'and save all requested output workbooks. Never use expected target values, '
            'fixed task coordinates, task IDs or copied example answers. Input facts '
            'are observations, not targets: validate their meaning against the task; '
            'when facts are absent, fail or are incomplete, inspect the raw original '
            'inputs and continue ordinary solving. Do not equate numeric candidates '
            'with output answers or unconditionally convert numbers, text or percentages.')
        if context.fact_kind:
            from .demands import consumer_contract
            system += '\nRetain this consumer interface:\n' + consumer_contract(context.fact_kind)
        else:
            system += (f'\nThis is an ordinary revision. needs may be [] or reference only '
                       f'these existing ordinary card slots: {list(ordinary_needs)}. '
                       'Do not declare a facts:* dependency or invent another dependency.')
        guide = context.build.get(('propose', ''))
        if guide is not None:
            system += f'\n\nImprover guidance:\n{guide.content}'
        parts = [f'TARGET SLOT: {slot}', f'CURRENT HARNESS:\n{render_build(context.build)}',
                 f'COMPLETE CARD TO REVISE:\n{target.content}',
                 f'TRAINING EVIDENCE:\n{context.evidence}']
        if context.diagnosis:
            parts.append(f'FAILURE DIAGNOSIS:\n{context.diagnosis}')
        if context.chemistry:
            parts.append(f'INTERFACE GUIDANCE / OBSERVED TRAINING CONTRAST:\n{context.chemistry}')
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': '\n\n'.join(parts)}]
        for attempt in range(1 + int(self.config['repair_attempts'])):
            try:
                raw = self._chat(messages, seed + attempt, int(self.config['max_tokens']),
                                 'propose_revision')
            except ContextTooLong as exc:
                self.events.emit('invalid_revision', mode='revision', slot=slot, error=str(exc))
                return None
            try:
                item = self._validate(parse_item(raw), 'forge', slot, target, None)
                if len(item.content) > limit:
                    raise ValueError(f'The complete revision must be at most {limit} characters')
                if context.fact_kind:
                    from .demands import validate_consumer
                    validate_consumer(item, context.fact_kind)
                elif any(need not in ordinary_needs for need in item.needs):
                    raise ValueError('Ordinary revision dependencies must be existing ordinary card '
                                     'slots, never facts:* or undeclared interfaces')


                if (item.content.strip() == target.content.strip() and
                        set(item.needs) == set(target.needs)):
                    self.events.emit('unchanged_revision', mode='revision', slot=slot,
                                     attempt=attempt, reason='same_content_and_dependencies')
                    return None
                self.events.emit('proposal', mode='revision', slot=slot,
                                 item=item.to_dict(), attempt=attempt)
                return item
            except (ValueError, KeyError, TypeError) as exc:
                self.events.emit('invalid_revision', mode='revision', slot=slot, attempt=attempt,
                                 error=str(exc)[:500], response=raw[-2000:])
                messages += [{'role': 'assistant', 'content': strip_thinking(raw)[-2200:]},
                             {'role': 'user', 'content': f'Rejected: {str(exc)[:300]}. Return '
                              f'the JSON header for {slot} and ONE text block containing the '
                              f'complete replacement, at most {limit} characters.'}]
        return None

    def diagnose(self, context: Context, seed: int) -> str:

        system = ("You analyse failures of a spreadsheet agent (the same model as you) to guide a harness "
                  "change. Be concrete and brief (at most 150 words). Never quote expected answers.")
        guide = context.build.get(("diagnose", ""))
        if guide is not None:
            system += f"\n\nDiagnosis guidance:\n{guide.content}"
        user = f"CURRENT HARNESS:\n{render_build(context.build)}\n\nEVIDENCE:\n{context.evidence}"
        try:
            text = self._chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                              seed, int(self.config["diagnosis_tokens"]), "diagnose")
        except ContextTooLong as exc:
            self.events.emit("diagnosis_failed", error=str(exc))
            return ""
        text = strip_thinking(text)[:1500]
        self.events.emit("diagnosis", text=text)
        return text

    def draft_producer(self, context: Context, partner: Item, seed: int,
                       support_evidence: str = "", runtime_check=None,
                       max_attempts: int | None = None) -> Item | None:

        if max_attempts is not None and (type(max_attempts) is not int or not 1 <= max_attempts <= 4):
            raise ValueError('Producer max_attempts must be an integer within 1..4')
        attempts = 1 + int(self.config['repair_attempts']) if max_attempts is None else max_attempts
        from .support import PRODUCER_SCHEMA, format_producer_item
        system = ("Write a reusable input-observation producer for a frozen spreadsheet agent. "
                  "It must help the partner interpret fresh workbook inputs. Return one ```json "
                  "header {\"slot\":\"inspect\",\"name\":\"short_name\",\"needs\":[]} "
                  "and one ```python block defining observe(books). Return JSON-serializable "
                  "facts, never executable edits or answers. No imports, file access, writes to "
                  "workbooks, task IDs, fixed file names, fixed sheet names, target cell constants, "
                  "or concrete answer values. Use only the provided input snapshot; functions, "
                  "lists/dicts, bounded loops, numeric/string conversion and aggregation are "
                  "available. Keep the source below 1500 characters so the contract can retain "
                  "the incumbent inspection guidance. No try/except, f-strings, next(), chr(), "
                  "lambda or extra helper functions. Use cell['coordinate'] directly; the "
                  "snapshot already supplies coordinates. Initialize out = []; for each "
                  "book initialize sheets = [], append the computed sheet records, then "
                  "append {'file_index': book['file_index'], 'sheets': sheets} to out and "
                  "return out. Fill the declared fact records using bounded loops and .get().\n" +
                  PRODUCER_SCHEMA)
        system += ("\nPrefer simple imperative code: initialize out/sheets/records lists, "
                   "use separate for loops and append one dictionary at each level, then "
                   "return out. Do not nest multiple list comprehensions inside a returned "
                   "dictionary; avoid starred list unpacking. Keep the header needs empty: "
                   "this observer reads books directly and supplies the partner's facts.")
        system += ("\nInput shape reminder: sheet['cells'] is a FLAT list of cell dictionaries, "
                   "not rows or columns. Iterate for cell in sheet['cells']; use cell['row'] "
                   "and cell['column'] to filter/group records. Iterating a cell dictionary "
                   "itself produces string keys, never further cell dictionaries.")
        user = (f"CURRENT HARNESS:\n{render_build(context.build)}\n\n"
                f"RETAINED CONSUMER ({partner.slot}, {partner.name}):\n{partner.content}\n"
                f"DECLARED NEEDS: {list(partner.needs)}\n\n"
                f"ORDINARY TRAINING EVIDENCE:\n{context.evidence}\n\n"
                f"TEMPORARY SUPPORT EVIDENCE (empty for direct control):\n{support_evidence}")
        from .demands import FACT_SCHEMAS, FACT_USE, declared_kinds
        kinds = declared_kinds(partner)
        if kinds:
            kind = kinds[0]
            system += (f"\nImplement this exact consumer interface: {FACT_SCHEMAS[kind]}. "
                       f"{FACT_USE[kind]} Select concise useful records, not the complete books snapshot. "
                       "Do not copy concrete facts from the examples. The actor must still be able to inspect raw inputs.")
            system += '\nPrefer source under 1200 characters to leave room for retained inspection instructions.'
            user += f"\nRequired fact field: {kind}; preserve this key in your returned structure."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(attempts):
            try:
                raw = self._chat(messages, seed + attempt, int(self.config["max_tokens"]),
                                 "propose_producer")
            except ContextTooLong as exc:
                self.events.emit("invalid_producer", error=str(exc), partner=partner.id)
                return None
            try:
                validation_stage = 'static'
                parsed = parse_item(raw)
                if parsed.slot != "inspect":
                    raise ValueError("A producer must use slot inspect")
                item = format_producer_item(parsed.name, parsed.content, needs=parsed.needs)
                if kinds:
                    producer = ast.parse(parsed.content)
                    constants = {n.value for n in ast.walk(producer)
                                 if isinstance(n, ast.Constant) and isinstance(n.value, str)}
                    if kinds[0] not in constants:
                        raise ValueError(f"Producer must return the consumer's {kinds[0]} field")

                    guidance = 'Fact interface: ' + kinds[0] + '. ' + FACT_USE[kinds[0]]
                    item = format_producer_item(parsed.name, parsed.content, guidance, parsed.needs)
                    from .support import retain_inspection
                    item = retain_inspection(item, context.build)
                if runtime_check is not None:
                    validation_stage = 'runtime'
                    runtime_check(item)
                self.events.emit("producer_proposal", item=item.to_dict(), partner=partner.id,
                                 attempt=attempt, used_support_evidence=bool(support_evidence),
                                 runtime_checked=runtime_check is not None)
                return item
            except (ValueError, KeyError, TypeError) as exc:
                if validation_stage == 'runtime' and isinstance(exc, (KeyError, TypeError)):
                    raise
                self.events.emit("invalid_producer", partner=partner.id, attempt=attempt,
                                 error=str(exc)[-500:], response=raw[-2000:],
                                 validation_stage=validation_stage)


                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": user +
                             "\n\nLATEST REJECTED PROGRAM:\n" + strip_thinking(raw)[-4000:] +
                             "\nVALIDATION ERROR:\nRejected: " + str(exc)[-1200:] +
                             "\nRewrite the observer from scratch to fix this error. "
                             "Do not repeat the rejected program. Use simple separate "
                             "for loops and append calls, balanced brackets, and the "
                             "same declared fact field. Return the JSON header with "
                             "needs: [] and one complete Python observe(books) block."}]
        return None
