import contextlib
from tests.workspace import TemporaryDirectory
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from rouge.batching import BatchedLocalModel, Request, compatible, groups_by_padding
from rouge.core import Budget, Events, InfrastructureError


def request(budget, seed=0, temperature=0):
    return Request([], {"budget": budget, "max_tokens": 4, "temperature": temperature,
                        "purpose": "unit-test", "seed": seed}, Future())


class BatchTests(unittest.TestCase):
    def test_sampling_and_separate_budgets_cannot_share_a_batch(self):
        b = Budget(100)
        a = request(b)
        self.assertTrue(compatible(a, request(b, seed=10)))
        self.assertFalse(compatible(a, request(b, temperature=.8)))
        self.assertFalse(compatible(a, request(Budget(100))))

    def test_padded_prefill_bound(self):
        rows = [request(Budget(100)) for _ in range(4)]
        groups = list(groups_by_padding(rows, [100, 4000, 100, 5000], 4, 8192))
        self.assertEqual([len(g) for g in groups], [2, 1, 1])
        for group in groups:
            self.assertLessEqual(max(n for _, n in group) * len(group), 8192)

    def model(self, root, fail=False):
        class Input:
            shape = (2, 4)
            def to(self, device): return self

        class Output:
            def __getitem__(self, index): return self
            def tolist(self): return [[5, 9, 9, 9], [6, 7, 8, 9]]

        class Tokenizer:
            pad_token_id = 9
            def __call__(self, *args, **kwargs): return {"input_ids": Input()}
            def decode(self, tokens, **kwargs): return str([t for t in tokens if t != 9])

        def generate(**kwargs):
            if fail: raise RuntimeError("simulated GPU capacity failure")
            return Output()

        model = BatchedLocalModel.__new__(BatchedLocalModel)
        model._render = lambda messages: "text"
        model.tokenizer = Tokenizer()
        model.model = SimpleNamespace(device="cuda", generate=generate,
                                      generation_config=SimpleNamespace(eos_token_id=[9]))
        model.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        model.events = Events(root / "events.jsonl")
        return model

    def test_bill_true_lengths_including_eos_but_not_batch_padding(self):
        with TemporaryDirectory() as d:
            budget = Budget(100)
            a, b = request(budget), request(budget)
            self.model(Path(d))._batch([(a, 2), (b, 4)])
            self.assertEqual(a.future.result(), "[5]")
            self.assertEqual(b.future.result(), "[6, 7, 8]")
            self.assertEqual(budget.prompt_tokens, 6)
            self.assertEqual(budget.generated_tokens, 6)
            self.assertEqual(budget.spent, 12)
            self.assertEqual(budget.reserved, 0)

    def test_failed_batch_fails_every_request_and_settles_reservations(self):
        with TemporaryDirectory() as d:
            budget = Budget(100)
            requests = [request(budget), request(budget)]
            self.model(Path(d), fail=True)._batch(list(zip(requests, [2, 4])))
            for r in requests:
                with self.assertRaises(InfrastructureError): r.future.result()
            self.assertEqual(budget.reserved, 0)
            self.assertEqual(budget.failed_calls, 2)


if __name__ == "__main__":
    unittest.main()
