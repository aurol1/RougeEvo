import hashlib
import json
import sys
import tarfile
from tests.workspace import TemporaryDirectory
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rouge.core import Events, InfrastructureError
from rouge.model import LocalModel, _local_preflight, checkpoint_spec, verify_loaded_checkpoint


class Config:
    model_type = "qwen3"


class Qwen3ForCausalLM:
    config = Config()
    def eval(self):
        return self


class Qwen2ForCausalLM:
    config = SimpleNamespace(model_type="qwen2")


class RuntimePreflightTests(unittest.TestCase):
    def checkpoint(self, root, **changes):
        spec = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], **changes}
        (root / "config.json").write_text(json.dumps(spec), encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"fixture, not model weights")
        return {"path": str(root)}

    def transformers(self, cls=Qwen3ForCausalLM, version="4.51.3", info=None):
        hf = Config()
        loader = Mock(return_value=(cls(), {} if info is None else info))
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
        module = SimpleNamespace(__version__=version,
            AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=hf)),
            AutoModelForCausalLM=SimpleNamespace(_model_mapping={Config: cls}, from_pretrained=loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)))
        return module

    def test_preflight_checks_architecture_without_loading_weights(self):
        with TemporaryDirectory() as d:
            cfg, module = self.checkpoint(Path(d)), self.transformers()
            report, parsed = _local_preflight(cfg, module)
            self.assertEqual(report["model_type"], "qwen3")
            self.assertEqual(report["resolved_architectures"], ["Qwen3ForCausalLM"])
            self.assertFalse(report["weights_loaded"])
            self.assertIsInstance(parsed, Config)
            module.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_old_runtime_fails_before_weight_loader(self):
        with TemporaryDirectory() as d:
            cfg, module = self.checkpoint(Path(d)), self.transformers(version="4.40.0")
            with self.assertRaisesRegex(InfrastructureError, "supported range"):
                _local_preflight(cfg, module)
            module.AutoModelForCausalLM.from_pretrained.assert_not_called()
            with patch("rouge.model.sys.version_info", (3, 8, 10)):
                with self.assertRaisesRegex(InfrastructureError, "Python >=3.10"):
                    _local_preflight(cfg, self.transformers())

    def test_qwen3_to_qwen2_fallback_is_rejected_before_weights(self):
        with TemporaryDirectory() as d:
            cfg, module = self.checkpoint(Path(d)), self.transformers(cls=Qwen2ForCausalLM)
            with self.assertRaisesRegex(InfrastructureError, "runtime resolves"):
                _local_preflight(cfg, module)
            module.AutoModelForCausalLM.from_pretrained.assert_not_called()
            module = self.transformers()
            module.AutoConfig.from_pretrained.return_value = Qwen2ForCausalLM.config
            with self.assertRaisesRegex(InfrastructureError, "parsed as 'qwen2'"):
                _local_preflight(cfg, module)

    def test_missing_metadata_and_shards_are_rejected(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            self.checkpoint(root, architectures=[])
            with self.assertRaisesRegex(InfrastructureError, "architectures"):
                checkpoint_spec(root)
            self.checkpoint(root)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "absent.safetensors"}}), encoding="utf-8")
            with self.assertRaisesRegex(InfrastructureError, "missing weight shards"):
                checkpoint_spec(root)
            index.write_text(json.dumps({"weight_map": {"weight": "../escape.safetensors"}}), encoding="utf-8")
            with self.assertRaisesRegex(InfrastructureError, "escapes checkpoint"):
                checkpoint_spec(root)

    def test_loading_info_rejects_new_random_or_ignored_weights(self):
        report = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
        verify_loaded_checkpoint(Qwen3ForCausalLM(), report, {})
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(InfrastructureError, "incompatible frozen checkpoint"):
                    verify_loaded_checkpoint(Qwen3ForCausalLM(), report, {key: ["q_proj.bias"]})
        with self.assertRaisesRegex(InfrastructureError, "Loaded Qwen2"):
            verify_loaded_checkpoint(Qwen2ForCausalLM(), report, {})

    def test_local_loader_explicitly_requests_config_and_loading_diagnostics(self):
        with TemporaryDirectory() as d:
            cfg, module = self.checkpoint(Path(d)), self.transformers()
            with patch.dict(sys.modules, {"transformers": module, "torch": SimpleNamespace(float16="float16")}):
                model = LocalModel(cfg, Events(Path(d) / "events.jsonl"))
            try:
                kwargs = module.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                self.assertTrue(kwargs["output_loading_info"])
                self.assertFalse(kwargs["ignore_mismatched_sizes"])
                self.assertIs(kwargs["config"], module.AutoConfig.from_pretrained.return_value)
                self.assertTrue(model.preflight["weights_loaded"])
            finally:
                model._generation_pool.shutdown()

    def test_actual_loader_missing_bias_cannot_start_generation_worker(self):
        with TemporaryDirectory() as d:
            cfg = self.checkpoint(Path(d))
            module = self.transformers(info={"missing_keys": ["model.layers.0.self_attn.q_proj.bias"]})
            with patch.dict(sys.modules, {"transformers": module, "torch": SimpleNamespace(float16="float16")}):
                with patch("rouge.model.ThreadPoolExecutor") as worker:
                    with self.assertRaisesRegex(InfrastructureError, "newly initialized weights"):
                        LocalModel(cfg, Events(Path(d) / "events.jsonl"))
                    worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
