from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time

from rouge.core import Budget, InfrastructureError


def durable_json(path: Path, data: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

    if os.name != 'nt':
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class ReportLock:

    def __init__(self, directory: Path):
        self.path = directory / '.frozen_report.lock'
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        try:

            if os.fstat(self.stream.fileno()).st_size == 0:
                self.stream.write(b'0')
                self.stream.flush()
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise InfrastructureError('Another process owns this frozen report directory') from exc
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class ReportBudget(Budget):

    SCHEMA = 1

    def __init__(self, path: Path, identity: str, limit: float, tool_price: float,
                 *, resume: bool = False):
        super().__init__(limit, tool_price)
        self.path = Path(path)
        self.identity = identity
        self._transaction = threading.RLock()
        self.sequence = 0
        self.uncertain_recovery_cost = 0.0
        self.recoveries: list[dict] = []
        self.poisoned = False
        if self.path.exists():
            if not resume:
                raise ValueError('Existing report budget requires --resume')
            self._restore()
        elif resume:
            raise ValueError('Cannot resume without the durable report budget checkpoint')
        else:
            self._persist()

    def _restore(self):
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('schema_version') != self.SCHEMA or data.get('identity') != self.identity:
            raise ValueError('Report budget identity/schema mismatch')
        saved = data['budget']
        if saved['limit'] != self.limit or saved['tool_price'] != self.tool_price:
            raise ValueError('Report budget limit/tool price changed on resume')
        for key in super().snapshot():
            value = saved[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid report budget field: {key}')
            if key in ('prompt_tokens', 'generated_tokens', 'tool_calls', 'failed_calls') and not isinstance(value, int):
                raise ValueError(f'Report budget counter is not integral: {key}')
            setattr(self, key, value)
        self.sequence = data['sequence']
        self.uncertain_recovery_cost = data['uncertain_recovery_cost']
        self.recoveries = data['recoveries']
        if (not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1 or
                isinstance(self.uncertain_recovery_cost, bool) or
                not isinstance(self.uncertain_recovery_cost, (int, float)) or
                not math.isfinite(self.uncertain_recovery_cost) or self.uncertain_recovery_cost < 0 or
                not isinstance(self.recoveries, list)):
            raise ValueError('Invalid report budget recovery metadata')
        if any(not isinstance(row, dict) or row.get('policy') != 'charge_unresolved_reservation' or
               isinstance(row.get('amount'), bool) or not isinstance(row.get('amount'), (int, float)) or
               not math.isfinite(row['amount']) or row['amount'] <= 0 for row in self.recoveries):
            raise ValueError('Invalid unresolved-reservation recovery record')
        if abs(sum(row['amount'] for row in self.recoveries) - self.uncertain_recovery_cost) > 1e-6:
            raise ValueError('Report budget recovery records do not reconcile')
        measured = self.prompt_tokens + self.generated_tokens + self.tool_calls * self.tool_price
        if abs(self.spent - measured - self.uncertain_recovery_cost) > 1e-6:
            raise ValueError('Report budget measured and recovery costs do not reconcile')
        if self.reserved:
            amount = self.reserved
            self.spent += amount
            self.uncertain_recovery_cost += amount
            self.reserved = 0.0
            self.recoveries.append({'amount': amount, 'from_sequence': self.sequence,
                                    'time_unix': time.time(), 'policy': 'charge_unresolved_reservation'})
            self._persist()

    def _persist(self):
        if self.poisoned:
            raise InfrastructureError('Durable report budget failed; resume from its last checkpoint')
        data = {'schema_version': self.SCHEMA, 'identity': self.identity,
                'sequence': self.sequence + 1, 'budget': super().snapshot(),
                'uncertain_recovery_cost': self.uncertain_recovery_cost, 'recoveries': self.recoveries}
        try:
            durable_json(self.path, data)
        except OSError as exc:
            self.poisoned = True
            raise InfrastructureError(f'Cannot persist report budget: {exc}') from exc
        self.sequence += 1

    def reserve(self, cost: float) -> float:
        with self._transaction:
            if self.poisoned:
                raise InfrastructureError('Durable report budget is unavailable')
            hold = super().reserve(cost)
            self._persist()
            return hold

    def release(self, hold: float) -> None:
        with self._transaction:
            super().release(hold)
            self._persist()

    def settle_model(self, hold: float, prompt: int, generated: int, seconds: float,
                     failed: bool = False) -> None:
        with self._transaction:
            super().settle_model(hold, prompt, generated, seconds, failed)
            self._persist()

    def settle_tool(self, hold: float, seconds: float, failed: bool = False) -> None:
        with self._transaction:
            super().settle_tool(hold, seconds, failed)
            self._persist()

    def snapshot(self) -> dict:
        with self._transaction:
            return {**super().snapshot(), 'uncertain_recovery_cost': self.uncertain_recovery_cost,
                    'recovery_count': len(self.recoveries), 'checkpoint_sequence': self.sequence,
                    'accounting': 'measured_usage_plus_charged_unresolved_reservations',
                    'exact_usage': self.uncertain_recovery_cost == 0}
