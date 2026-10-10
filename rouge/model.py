from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core import Budget, ContextTooLong, Events, InfrastructureError


def checkpoint_spec(path: Path) -> dict:

    path = path.expanduser().resolve()
    if not path.is_dir():
        raise InfrastructureError(f"Local model directory does not exist: {path}")
    try:
        raw = (path / "config.json").read_bytes()
        spec = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise InfrastructureError(f"Cannot read checkpoint config.json at {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise InfrastructureError(f"Checkpoint config.json at {path} must be a JSON object")
    model_type, architectures = spec.get("model_type"), spec.get("architectures")
    if not isinstance(model_type, str) or not model_type.strip():
        raise InfrastructureError("Checkpoint config.json must declare a model_type; refusing name-based inference")
    if (not isinstance(architectures, list) or not architectures or
            any(not isinstance(name, str) or not name for name in architectures)):
        raise InfrastructureError("Checkpoint config.json must declare its exact causal-LM architectures")
    indexes = [path / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")]
    index = next((file for file in indexes if file.is_file()), None)
    if index is not None:
        try:
            names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            if not names or any(not isinstance(name, str) for name in names):
                raise ValueError("weight_map is empty or invalid")


            if any(Path(name).is_absolute() or "\\" in name or ".." in Path(name).parts for name in names):
                raise ValueError("weight shard path escapes checkpoint directory")
            missing = sorted(name for name in names if not (path / name).is_file())
            if missing:
                raise ValueError(f"missing weight shards: {missing[:5]}")
            weight_files = len(names)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InfrastructureError(f"Invalid checkpoint weight index {index.name}: {exc}") from exc
    else:
        weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
        weight_files = len([file for file in weights if file.is_file()])
        if not weight_files:
            raise InfrastructureError(f"No local checkpoint weights found at {path}")
    return {"path": str(path), "model_type": model_type, "architectures": architectures,
            "config_sha256": hashlib.sha256(raw).hexdigest(), "weight_files": weight_files}


def _local_preflight(config: dict, transformers_module=None):

    if sys.version_info < (3, 10):
        raise InfrastructureError(f"Python {sys.version.split()[0]} is unsupported; use the server's existing "
                                  "Python >=3.10 model environment, not /usr/bin/python3")
    spec = checkpoint_spec(Path(config["path"]))
    try:
        if transformers_module is None:
            import transformers as transformers_module
        version = str(transformers_module.__version__)
        numbers = re.match(r"^(\d+)\.(\d+)", version)
        if numbers is None:
            raise ValueError(f"unrecognized Transformers version {version!r}")
        major, minor = (int(part) for part in numbers.groups())
        if not (4 <= major < 6) or (major == 4 and minor < 51):
            raise ValueError(f"Transformers {version} is outside the supported range >=4.51,<6")
        hf_config = transformers_module.AutoConfig.from_pretrained(
            spec["path"], local_files_only=True, trust_remote_code=False)
        if hf_config.model_type != spec["model_type"]:
            raise ValueError(f"checkpoint model_type={spec['model_type']!r}, parsed as {hf_config.model_type!r}")
        model_class = transformers_module.AutoModelForCausalLM._model_mapping[type(hf_config)]
        classes = model_class if isinstance(model_class, (list, tuple)) else (model_class,)
        resolved = [cls.__name__ for cls in classes]
        if not any(name in resolved for name in spec["architectures"]):
            raise ValueError(f"checkpoint architectures={spec['architectures']}, runtime resolves {resolved}")
    except Exception as exc:
        raise InfrastructureError(f"Checkpoint/runtime mismatch for {spec['model_type']} at {spec['path']}: "
                                  f"{exc}. Activate the server's compatible model environment; "
                                  "do not relabel Qwen3 as Qwen2 or enable random-weight fallback") from exc
    return {**spec, "python": sys.version.split()[0], "transformers": version,
            "resolved_architectures": resolved, "weights_loaded": False}, hf_config


def preflight_local_model(config: dict) -> dict:

    report, _ = _local_preflight(config)
    return report


def verify_loaded_checkpoint(model, report: dict, loading_info: dict):

    actual = type(model).__name__
    if actual not in report["architectures"] or model.config.model_type != report["model_type"]:
        raise InfrastructureError(f"Loaded {actual}/{model.config.model_type}, expected "
                                  f"{report['architectures']}/{report['model_type']}")
    if not isinstance(loading_info, dict):
        raise InfrastructureError("Transformers did not return checkpoint loading diagnostics")
    faults = {key: loading_info.get(key) for key in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading_info.get(key)}
    if faults:
        detail = {key: list(value)[:8] for key, value in faults.items()}
        raise InfrastructureError(f"Incomplete or incompatible frozen checkpoint: {detail}. "
                                  "No experiment was started; do not accept newly initialized weights")


def fit_messages(messages, count, max_tokens: int, limit: int, events: Events, purpose: str):

    messages, removed = list(messages), 0
    prompt = count(messages)
    while prompt + max_tokens > limit and len(messages) > 2:
        del messages[2:4]
        removed += 2
        prompt = count(messages)
    if removed:
        events.emit("context_trim", purpose=purpose, removed_messages=removed)
    if prompt + max_tokens > limit:
        raise ContextTooLong(f"Context too long for {purpose}: {prompt}+{max_tokens}>{limit}")
    return messages


class LocalModel:
    def __init__(self, config: dict, events: Events):
        report, hf_config = _local_preflight(config)
        events.emit("model_preflight", **report)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(report["path"])
        self.torch, self.events, self.name = torch, events, path.name
        self.context_limit = int(config.get("context_tokens", 8192))
        self.thinking = bool(config.get("enable_thinking", False))
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                                       trust_remote_code=False)
        import transformers
        major = int(transformers.__version__.split(".")[0])
        dtype_key = "dtype" if major >= 5 else "torch_dtype"
        try:
            self.model, loading_info = AutoModelForCausalLM.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False,
                config=hf_config, output_loading_info=True, ignore_mismatched_sizes=False,
                device_map=config.get("device_map", "auto"),
                attn_implementation=config.get("attn_implementation", "sdpa"),
                **{dtype_key: getattr(torch, config.get("dtype", "float16"))})
        except Exception as exc:
            raise InfrastructureError(f"Frozen checkpoint loading failed in {sys.executable}: {exc}. "
                                      "Fix the runtime/checkpoint before starting the search") from exc
        verify_loaded_checkpoint(self.model, report, loading_info)
        self.preflight = {**report, "weights_loaded": True, "loaded_architecture": type(self.model).__name__}
        events.emit("model_checkpoint_verified", **self.preflight)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._lock = threading.Lock()


        self._generation_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rouge-gpu")

    def _render(self, messages) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.thinking)

    def count(self, messages) -> int:
        return len(self.tokenizer(self._render(messages), add_special_tokens=False)["input_ids"])

    def chat(self, messages, *, budget: Budget, seed: int, max_tokens: int, temperature: float,
             purpose: str, top_p: float = 0.95) -> str:
        return self._generation_pool.submit(self._chat, messages, budget=budget, seed=seed,
                                            max_tokens=max_tokens, temperature=temperature,
                                            purpose=purpose, top_p=top_p).result()

    def _chat(self, messages, *, budget: Budget, seed: int, max_tokens: int, temperature: float,
              purpose: str, top_p: float = 0.95) -> str:
        torch = self.torch
        messages = fit_messages(messages, self.count, max_tokens, self.context_limit, self.events, purpose)
        with self._lock:
            inputs = self.tokenizer(self._render(messages), return_tensors="pt", add_special_tokens=False)
            prompt = int(inputs["input_ids"].shape[-1])
            hold = budget.reserve(prompt + max_tokens)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0,
                      "pad_token_id": self.tokenizer.pad_token_id}
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=top_p)
            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = self.model.generate(**inputs, **kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                budget.settle_model(hold, prompt, max_tokens, time.perf_counter() - start, failed=True)

                raise InfrastructureError(f"CUDA out of memory for {purpose} at {prompt} tokens; "
                                          "use a larger GPU lane group or a smaller model/context. "
                                          + str(exc)[-700:]) from exc
            except Exception as exc:
                budget.settle_model(hold, prompt, max_tokens, time.perf_counter() - start, failed=True)
                raise InfrastructureError(f"Generation failed for {purpose}: {exc}") from exc
            tokens = output[0, prompt:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        elapsed = time.perf_counter() - start
        budget.settle_model(hold, prompt, len(tokens), elapsed)
        self.events.emit("model_call", purpose=purpose, seed=seed, prompt_tokens=prompt,
                         output_tokens=len(tokens), seconds=round(elapsed, 3))
        return text


class OpenAIModel:


    def __init__(self, config: dict, events: Events):
        self.base = config["base_url"].rstrip("/")
        self.url = self.base + "/chat/completions"
        self.key = os.environ.get(config.get("api_key_env", ""), "") if config.get("api_key_env") else ""
        self.context_limit = int(config.get("context_tokens", 8192))
        self.timeout = float(config.get("request_timeout", 600))
        self.thinking = bool(config.get("enable_thinking", False))


        self.template_kwargs = bool(config.get("send_template_kwargs", True))
        self.name = config.get("served_name") or self._served_name() or Path(config["path"]).name
        self.events, self.tokenizer = events, None
        tokenizer = config.get("tokenizer_path") or config.get("path")
        if tokenizer and Path(tokenizer).is_dir():
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
            except Exception:
                self.tokenizer = None

    def _served_name(self) -> str | None:

        request = urllib.request.Request(self.base + "/models", headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                ids = [x["id"] for x in json.loads(response.read().decode("utf-8")).get("data", [])]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
            return None
        return ids[0] if len(ids) == 1 else None

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        return headers

    def count(self, messages) -> int:
        if self.tokenizer is not None:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=self.thinking)
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

        return sum(len(m["content"]) // 3 + 8 for m in messages) + 16

    def chat(self, messages, *, budget: Budget, seed: int, max_tokens: int, temperature: float,
             purpose: str, top_p: float = 0.95) -> str:
        messages = fit_messages(messages, self.count, max_tokens, self.context_limit, self.events, purpose)
        estimate = self.count(messages)
        hold = budget.reserve(estimate + max_tokens)
        body = {"model": self.name, "messages": messages, "max_tokens": max_tokens, "seed": seed,
                "temperature": temperature, "top_p": top_p if temperature > 0 else 1.0}
        if self.template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        request = urllib.request.Request(self.url, json.dumps(body).encode(), self._headers(), method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            prompt = int(usage.get("prompt_tokens", estimate))
            generated = int(usage.get("completion_tokens", len(text) // 3))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[-500:]
            budget.settle_model(hold, estimate, 0, time.perf_counter() - start, failed=True)
            if exc.code == 400 and ("context" in detail.lower() or "too long" in detail.lower()):
                raise ContextTooLong(f"Server rejected {purpose}: {detail}") from exc
            hint = f" (served model name {self.name!r}; set model.served_name)" if exc.code == 404 else ""
            raise InfrastructureError(f"Model server HTTP {exc.code}{hint}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, IndexError) as exc:

            budget.settle_model(hold, estimate, max_tokens, time.perf_counter() - start, failed=True)
            raise InfrastructureError(f"Model server unavailable for {purpose}: {exc}") from exc
        elapsed = time.perf_counter() - start
        budget.settle_model(hold, prompt, generated, elapsed)
        self.events.emit("model_call", purpose=purpose, seed=seed, prompt_tokens=prompt,
                         output_tokens=generated, seconds=round(elapsed, 3))
        return text


def make_model(config: dict, events: Events):
    backend = config.get("backend", "hf")
    if backend == "hf":
        if int(config.get("batch_size", 1)) > 1:
            from .batching import BatchedLocalModel
            return BatchedLocalModel(config, events)
        return LocalModel(config, events)
    if backend == "openai":
        return OpenAIModel(config, events)
    raise ValueError(f"Unknown model backend {backend!r}")
