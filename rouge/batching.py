from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

from .core import InfrastructureError
from .model import LocalModel, fit_messages


@dataclass
class Request:
    messages: list
    kwargs: dict
    future: Future


def compatible(a: Request, b: Request) -> bool:
    return (a.kwargs["temperature"] == b.kwargs["temperature"] == 0 and
            a.kwargs["max_tokens"] == b.kwargs["max_tokens"] and
            a.kwargs["budget"] is b.kwargs["budget"])


def groups_by_padding(requests, counts, size, token_limit):

    group, width = [], 0
    for request, count in zip(requests, counts):
        if group and (len(group) >= size or max(width, count) * (len(group) + 1) > token_limit):
            yield group
            group, width = [], 0
        group.append((request, count))
        width = max(width, count)
    if group:
        yield group


class BatchedLocalModel(LocalModel):
    def __init__(self, config, events):
        super().__init__(config, events)
        self.tokenizer.padding_side = "left"
        self.batch_size = int(config.get("batch_size", 4))
        self.prefill_tokens = int(config.get("batch_prefill_tokens", 8192))
        self.requests = queue.Queue()

        self.dispatcher = threading.Thread(target=self._dispatch, name="rouge-batch-gpu", daemon=True)
        self.dispatcher.start()

    def chat(self, messages, **kwargs):
        future = Future()
        self.requests.put(Request(messages, kwargs, future))
        return future.result()

    def _dispatch(self):
        pending = None
        while True:
            first = pending or self.requests.get()
            pending = None
            group = [first]
            if first.kwargs["temperature"] == 0:
                until = time.monotonic() + 0.02
                while len(group) < self.batch_size and time.monotonic() < until:
                    try:
                        following = self.requests.get(timeout=max(0.0001, until - time.monotonic()))
                    except queue.Empty:
                        break
                    if not compatible(first, following):
                        pending = following
                        break
                    group.append(following)
            if len(group) == 1:
                try:
                    first.future.set_result(self._chat(first.messages, **first.kwargs))
                except BaseException as exc:
                    first.future.set_exception(exc)
                continue
            valid, counts = [], []
            for request in group:
                try:
                    request.messages = fit_messages(request.messages, self.count, request.kwargs["max_tokens"],
                                                     self.context_limit, self.events, request.kwargs["purpose"])
                    counts.append(self.count(request.messages))
                    valid.append(request)
                except BaseException as exc:
                    request.future.set_exception(exc)
            for sub in groups_by_padding(valid, counts, self.batch_size, self.prefill_tokens):
                self._batch(sub)

    def _batch(self, group):
        held = []
        for request, count in group:
            try:
                hold = request.kwargs["budget"].reserve(count + request.kwargs["max_tokens"])
                held.append((request, count, hold))
            except BaseException as exc:
                request.future.set_exception(exc)
        if not held:
            return
        start = time.perf_counter()
        try:
            texts = [self._render(r.messages) for r, _, _ in held]
            inputs = self.tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True)
            width = int(inputs["input_ids"].shape[-1])
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=held[0][0].kwargs["max_tokens"],
                                             do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            rows = output[:, width:].tolist()
            eos = self.model.generation_config.eos_token_id
            stops = set(eos if isinstance(eos, list) else [eos])
            completions = []
            for tokens in rows:

                end = next((i + 1 for i, tok in enumerate(tokens) if tok in stops), len(tokens))
                tokens = tokens[:end]
                completions.append((self.tokenizer.decode(tokens, skip_special_tokens=True), len(tokens)))
        except BaseException as exc:
            elapsed = time.perf_counter() - start
            failure = InfrastructureError(f"HF batch generation failed: {type(exc).__name__}: {str(exc)[-700:]}")
            for request, count, hold in held:
                request.kwargs["budget"].settle_model(hold, count, request.kwargs["max_tokens"],
                                                      elapsed / len(held), failed=True)
                request.future.set_exception(failure)
            return
        elapsed = time.perf_counter() - start
        self.events.emit("model_batch", batch_size=len(held), padded_prompt_tokens=width * len(held),
                         seconds=elapsed)
        for (request, count, hold), (text, generated) in zip(held, completions):
            request.kwargs["budget"].settle_model(hold, count, generated, elapsed / len(held))
            self.events.emit("model_call", purpose=request.kwargs["purpose"], seed=request.kwargs["seed"],
                             prompt_tokens=count, output_tokens=generated, batch_size=len(held),
                             seconds=elapsed / len(held))
            request.future.set_result(text)
