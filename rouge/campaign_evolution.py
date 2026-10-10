from __future__ import annotations

import json

from .core import ContextTooLong, rng_for
from .evolution import Context, Proposer, render_build
from .items import CARD_CHARS
from .parsing import parse_item, strip_thinking


SLOTS = ('inspect', 'plan', 'edit', 'verify')


def _steps(result):
    try:
        rows = json.loads(result.trace)
    except (ValueError, TypeError):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get('step', -1) >= 0] if isinstance(rows, list) else []


def unresolved_error(result):

    if result.success:
        return False
    rows = _steps(result)
    if not rows:
        return str(result.feedback).startswith(('task_error:', 'input_producer_failed'))
    final = rows[-1]
    feedback = str(final.get('feedback', ''))
    return final.get('returncode') not in (None, 0) or feedback.startswith('Invalid reply:')


def _bounded(text, limit):
    if len(text) <= limit:
        return text
    marker = '\n[execution excerpt omitted]\n'
    if limit < len(marker) + 2:
        return '[omitted]'[:max(0, limit)]
    room = limit - len(marker)
    return text[:room * 2 // 3] + marker + text[-(room - room * 2 // 3):]


def execution_evidence(result, limit):

    if limit <= 0:
        return ''
    rows = _steps(result)
    if not rows:
        return _bounded(str(result.trace), limit)
    code = [r for r in rows if '```' in str(r.get('response', ''))]
    writes = [r for r in code if '.save(' in str(r.get('response', ''))]
    chosen = (writes[-1:] or code[-1:] or rows[-1:])
    if unresolved_error(result) and rows[-1] not in chosen:
        chosen += rows[-1:]
    each = max(1, limit // len(chosen))
    blocks = []
    for row in chosen:
        prefix = f"step={row.get('step')}; returncode={row.get('returncode')}\n"
        feedback = str(row.get('feedback', ''))
        error_room = min(len(feedback), max(0, each // 3))
        response_room = max(0, each - len(prefix) - error_room - 18)
        block = prefix + _bounded(str(row.get('response', '')), response_room)
        if error_room:
            block += '\nfeedback:\n' + feedback[-error_room:]
        blocks.append(block)
    return '\n'.join(blocks)[:limit]


class CoverageEvidence:

    def __init__(self, tasks, train_ids, seed, exposures=None):
        self.tasks, self.train_ids, self.seed = tasks, set(train_ids), seed
        self.exposures = exposures if exposures is not None else {}
        self.last_task_ids, self.last_selection, self.last_omissions = [], [], []

    def context(self, build, evaluation, cached_results, round_id, limit_chars=6500, max_examples=3):
        if type(limit_chars) is not int or limit_chars < 100:
            raise ValueError('Evidence limit must be an integer of at least 100 characters')
        if type(max_examples) is not int or not 1 <= max_examples <= 4:
            raise ValueError('Evidence examples must be within 1..4')
        if evaluation.build_id != build.id:
            raise ValueError('Evidence evaluation must belong to the current build')
        rows = {r.task_id: r for r in list(cached_results) + list(evaluation.results)}
        if any(t not in self.train_ids or t not in self.tasks for t in rows):
            raise PermissionError('Campaign evidence may only use configured training tasks')
        self.last_task_ids, self.last_selection, self.last_omissions = [], [], []
        rng = rng_for(self.seed, 'campaign_evidence', round_id, build.execution_id)
        failed = [r for r in rows.values() if not r.success]
        passed = [r for r in rows.values() if r.success]
        rng.shuffle(failed)
        rng.shuffle(passed)
        failed.sort(key=lambda r: (self.exposures.get(r.task_id, 0), not unresolved_error(r)))
        passed.sort(key=lambda r: self.exposures.get(r.task_id, 0))


        ordered, seen_types = [], set()
        while failed:
            least = self.exposures.get(failed[0].task_id, 0)
            choices = [r for r in failed if self.exposures.get(r.task_id, 0) == least]
            diverse = [r for r in choices if self.tasks[r.task_id].instruction_type not in seen_types]
            row = (diverse or choices)[0]
            failed.remove(row)
            ordered.append(row)
            seen_types.add(self.tasks[row.task_id].instruction_type)


        wanted = []
        if ordered:
            wanted.append(ordered.pop(0))
        if passed and max_examples > 1:
            wanted.append(passed.pop(0))
        wanted += ordered + passed
        selected, headers, used = [], [], 0
        reserve_note = 110
        for row in wanted:
            if len(selected) >= max_examples:
                break
            task = self.tasks[row.task_id]
            header = (f'Training example {len(selected)+1}: success={row.success}; '
                      f'type={task.instruction_type}; answer_position={task.answer_position}\n'
                      f'COMPLETE INSTRUCTION:\n{task.instruction}\n'
                      f'Outcome: {row.feedback}; unresolved_execution_error={unresolved_error(row)}\n'
                      f'Training diagnostic: {row.diagnostic[:350]}\n'
                      f'Training cell credit: {json.dumps(row.training_progress, ensure_ascii=False)}\n')
            minimum_trace = min(300, max(50, limit_chars // 12))
            if used + len(header) + minimum_trace + reserve_note > limit_chars:
                self.last_omissions.append({'task_id': row.task_id,
                    'reason': 'whole_instruction_and_minimum_trace_do_not_fit',
                    'instruction_chars': len(task.instruction)})
                continue
            selected.append(row)
            headers.append(header)
            used += len(header) + minimum_trace + 3
        if not selected:
            return Context(build, '(No complete training example fits the evidence limit; '
                           'instructions were omitted whole, never truncated.)')
        available = max(0, limit_chars - sum(map(len, headers)) - reserve_note - 3 * len(selected))
        each = available // len(selected)
        blocks = []
        for row, header in zip(selected, headers):
            blocks.append(header + 'Executed procedure:\n' + execution_evidence(row, max(0, each - 21)))
            before = self.exposures.get(row.task_id, 0)
            self.exposures[row.task_id] = before + 1
            self.last_task_ids.append(row.task_id)
            self.last_selection.append({'task_id': row.task_id, 'success': row.success,
                'exposure_before': before, 'exposure_after': before + 1,
                'unresolved_execution_error': unresolved_error(row),
                'instruction_chars': len(self.tasks[row.task_id].instruction),
                'instruction_complete': True})
        note = ('\nSome examples were omitted whole to retain complete instructions.'
                if self.last_omissions else '')
        evidence = '\n\n'.join(blocks) + note
        if len(evidence) > limit_chars:
            raise AssertionError('Evidence formatter exceeded its allocated character budget')
        return Context(build, evidence)


def campaign_context(engine, build, evaluation, round_id):

    train = engine.evaluator.dataset.split('train', int(engine.search['train_limit']))
    if not hasattr(engine, 'evidence_exposures'):
        engine.evidence_exposures = {}
    selector = CoverageEvidence(engine.tasks, [t.id for t in train], engine.seed,
                                exposures=engine.evidence_exposures)
    cached = []
    for task in train:
        if hasattr(engine.evaluator, '_key'):
            row = engine.evaluator.cache.get(engine.evaluator._key(build, task, None))
        else:


            row = engine.evaluator.cache.get((build.execution_id, task.id, None))
            if row is None:
                row = engine.evaluator.cache.get((build.execution_id, task.id))
        if row is not None:
            cached.append(row)
    context = selector.context(build, evaluation, cached, round_id,
        limit_chars=int(engine.config['proposal']['evidence_chars']),
        max_examples=int(engine.options.get('campaign_evidence_examples', 3)))
    engine.events.emit('campaign_evidence', round=round_id, build=build.id,
        execution_id=build.execution_id, task_ids=selector.last_task_ids,
        selected=selector.last_selection, omitted=selector.last_omissions,
        pool_size=len({r.task_id for r in cached + list(evaluation.results)}),
        source='same_execution_ordinary_training_only', full_instructions=True)
    return context


class CampaignProposer(Proposer):

    def draft_revision(self, slot, context, seed, max_chars=2200):
        if slot not in SLOTS:
            raise ValueError('Revision target must be inspect, plan, edit or verify')
        target = context.build.get((slot, ''))
        if target is None:
            raise ValueError('Revision requires an existing instruction card')
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError('Revision max_chars must be a positive integer')
        limit = min(max_chars, CARD_CHARS)
        ordinary_needs = tuple(s for s in SLOTS if context.build.get((s, '')) is not None)
        focus = {
            'inspect': 'Determine which original-input observations are missing; inspect only what is needed to resolve the operation and mapping.',
            'plan': 'Resolve the requested operation, source-to-output mapping and conditional cases from the complete instruction and original inputs.',
            'edit': 'Turn the resolved rule into a precise transformation; repair the failing API or algorithm while preserving unrelated workbook content.',
            'verify': 'Check instruction-derived invariants and output coverage; distinguish an execution error from a logically wrong result and repair its cause.',
        }[slot]
        system = (
            f'Improve one harness card for a frozen spreadsheet agent. Write the COMPLETE replacement '
            f'for slot {slot}, at most {limit} characters. {focus}\n'
            'Use actual training evidence to select one substantive, reusable improvement. '
            'Keep useful procedures; remove harmful assumptions and duplication. State operational '
            'decisions, not a narrated failure report or vague requests to be careful. '
            'Rules must be conditional on the instruction: a blank output can be correct, so never '
            'require every target cell to be nonempty. Do not unconditionally convert types, erase '
            'formulas, skip erroring rows or apply one task-specific transformation to all tasks. '
            'Read original inputs, derive mappings separately for each case, and save every requested '
            'output. Never include task IDs, fixed coordinates, concrete expected values or copied '
            'example answers. Input facts are observations, not answers; when absent or incomplete, '
            'inspect raw inputs and continue solving.\n'
            'Reply with a ```json header {"slot":"' + slot + '","name":"short_name","needs":[]} '
            'then ONE ```text block with the full replacement. Do not append patch markers.')
        fact_kind = context.fact_kind if slot != 'inspect' else ''
        if fact_kind:
            from .demands import consumer_contract
            system += '\nRetain this consumer interface: ' + consumer_contract(fact_kind)
        else:
            system += f'\nDependencies may reference only existing ordinary card slots {list(ordinary_needs)}; never invent facts:* dependencies.'
        parts = [f'TARGET SLOT: {slot}\nCURRENT HARNESS:\n{render_build(context.build)}',
                 f'TRAINING EVIDENCE:\n{context.evidence}']
        if context.diagnosis:
            parts.append('TRAINING FAILURE ANALYSIS:\n' + context.diagnosis)
        if context.chemistry:
            parts.append('TRAINING INTERFACE EVIDENCE:\n' + context.chemistry)
        initial = [{'role': 'system', 'content': system}, {'role': 'user', 'content': '\n\n'.join(parts)}]
        messages = list(initial)
        for attempt in range(1 + int(self.config['repair_attempts'])):
            try:
                raw = self._chat(messages, seed + attempt, int(self.config['max_tokens']), 'propose_revision')
            except ContextTooLong as exc:
                self.events.emit('invalid_revision', mode='campaign_revision', slot=slot, error=str(exc))
                return None
            try:
                item = self._validate(parse_item(raw), 'forge', slot, target, None)
                if len(item.content) > limit:
                    raise ValueError(f'The complete revision must be at most {limit} characters')
                if fact_kind:
                    from .demands import validate_consumer
                    validate_consumer(item, fact_kind)
                elif any(need not in ordinary_needs for need in item.needs):
                    raise ValueError('Ordinary revision dependencies must be existing ordinary card slots, never facts:*')
                if item.content.strip() == target.content.strip() and set(item.needs) == set(target.needs):
                    self.events.emit('unchanged_revision', mode='campaign_revision', slot=slot,
                                     attempt=attempt, reason='same_content_and_dependencies')
                    return None
                self.events.emit('proposal', mode='campaign_revision', slot=slot,
                                 item=item.to_dict(), attempt=attempt)
                return item
            except (ValueError, KeyError, TypeError) as exc:
                self.events.emit('invalid_revision', mode='campaign_revision', slot=slot, attempt=attempt,
                                 error=str(exc)[:500], response=raw[-2000:])
                messages = initial + [{'role': 'assistant', 'content': strip_thinking(raw)[-2200:]},
                    {'role': 'user', 'content': f'Rejected: {str(exc)[:300]}. Return a corrected JSON '
                     f'header and ONE complete text replacement, at most {limit} characters.'}]
        return None

    def draft_producer(self, context, partner, seed, support_evidence='', runtime_check=None, max_attempts=None):
        clarification = (
            'Observer schema clarification: cell.kind uses Excel data_type codes: "s" for text, '
            '"n" for numbers, "f" for formulas, "b" for booleans, "d" for dates, "e" for errors. '
            'Never compare kind with "string" or "number". Header candidates must be nonempty '
            'and not restricted to row 1; retain their observed row/column. Select a compact sample '
            'of useful records, bound per-sheet/per-column lists, and omit empty header records. '
            'An empty result is legitimate when the input has no applicable records. Avoid '
            'repeating entire input tables; output must fit the observation character budget.\n')
        def checked(item):
            probes = runtime_check(item)
            if probes is not None:
                for probe in probes:
                    validate_producer_observation(probe)
            return probes
        return super().draft_producer(context, partner, seed,
            support_evidence=clarification + support_evidence,
            runtime_check=checked if runtime_check is not None else None,
            max_attempts=max_attempts)


def validate_producer_observation(probe, *, fact_kind='', input_has_text=False):

    if not probe.get('success', False):
        raise ValueError('Observer failed on original training inputs: ' + str(probe.get('error', ''))[-500:])
    if probe.get('truncated'):
        raise ValueError('Observer output was truncated; return fewer useful, nonempty records and shorter fields')
    if not (input_has_text and fact_kind in ('layout_profile', 'column_profile')):
        return
    try:
        facts = json.loads(probe.get('facts', ''))
    except (TypeError, ValueError):
        raise ValueError('Observer facts must be valid JSON')
    key = 'header_signature' if fact_kind == 'layout_profile' else 'headers'
    headers = []
    def collect(node):
        if isinstance(node, dict):
            for k, value in node.items():
                if k == key and isinstance(value, list):
                    headers.extend(value)
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
    collect(facts)
    if not any(isinstance(h, dict) and h.get('text') not in (None, '') for h in headers):
        raise ValueError('Original inputs contain text candidates, but every returned header is empty; check Excel kind codes and row filters')
