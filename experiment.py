from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from rouge.config import load_config, resolve_path, validate
from rouge.core import Budget, Events, InfrastructureError, write_json
from rouge.data import Dataset, OfficialScorer
from rouge.deck import resolve_build
from rouge.sandbox import DockerRunner, TaskFailure

HERE = Path(__file__).resolve().parent
FREE_SECTIONS = ("model", "docker")
FAKE_MODEL = "FAKE-TEST-SERVER-NOT-A-MODEL"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_identity(config: dict, model=None) -> dict:

    cfg = config["model"]
    path = Path(cfg["path"])
    identity = {"backend": cfg.get("backend", "hf"), "path": str(path),
                "served_name": getattr(model, "name", None), "dtype": cfg.get("dtype"),
                "enable_thinking": bool(cfg.get("enable_thinking", False)),
                "context_tokens": cfg.get("context_tokens")}
    identity["device_map_strategy"] = cfg.get("device_map", "auto")
    identity["batch_size"] = int(cfg.get("batch_size", 1))
    identity["batch_prefill_tokens"] = int(cfg.get("batch_prefill_tokens", 8192))
    identity["placement"] = getattr(getattr(model, "model", None), "hf_device_map", None)
    if (path / "config.json").is_file():
        identity["config_sha256"] = sha256(path / "config.json")
    return identity


def execution_protocol(config: dict, runner: DockerRunner, sandbox: dict, dataset: Dataset) -> dict:

    from rouge.execution import SYSTEM
    from rouge.preview import PREVIEW_VERSION
    from rouge.support import SUPPORT_VERSION, SUPPORT_CATALOG_VERSION
    ex = config["execution"]
    protocol = {"max_steps": int(ex["max_steps"]), "max_tokens": int(ex["max_tokens"]),
                "temperature": float(ex["temperature"]), "top_p": float(ex.get("top_p", 0.95)),
                "observation_chars": int(ex["observation_chars"]),
                "parallel_tasks": int(ex.get("parallel_tasks", 1)),
                "preview_chars": int(ex.get("preview_chars", 0)),
                "preview_version": PREVIEW_VERSION,
                "support_version": SUPPORT_VERSION,
                "support_catalog_version": SUPPORT_CATALOG_VERSION,
                "support_chars": int(ex.get("support_chars", 3000)),
                "reveal_expected": bool(ex.get("reveal_expected", True)),
                "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "sandbox": {"image": runner.image, "image_id": sandbox.get("image_id"),
                            "libreoffice": sandbox.get("libreoffice"), "timeout_seconds": runner.timeout,
                            "memory": runner.memory, "cpus": runner.cpus, "pids_limit": runner.pids},
                "scorer_sha256": dataset.metadata.get("scorer_sha256")}
    return {**protocol, "fingerprint": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}


def provenance(args) -> dict:
    files = sorted(HERE.glob("*.py")) + sorted((HERE / "rouge").glob("*.py"))
    packages = {}
    for name in ("numpy", "openpyxl", "pandas", "torch", "transformers", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpus = None
    if shutil.which("nvidia-smi"):
        try:
            gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                   "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=20).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            gpus = "unavailable"
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname(),
            "python": sys.version, "platform": platform.platform(), "packages": packages, "gpus": gpus,
            "argv": sys.argv, "source_hashes": {p.relative_to(HERE).as_posix(): sha256(p) for p in files}}


def _stale_workbook(path: Path) -> None:

    import openpyxl
    book = openpyxl.Workbook()
    sheet = book.active
    sheet["A1"], sheet["B1"], sheet["C1"] = 2, "=A1*21", "=_xlfn.SEQUENCE(1,1,5)"
    book.save(path)
    with zipfile.ZipFile(path) as src:
        parts = {n: src.read(n) for n in src.namelist()}
    xml, count = re.subn(r'<c r="B1"[^>]*>.*?</c>', '<c r="B1"><f>A1*21</f><v>0</v></c>',
                         parts["xl/worksheets/sheet1.xml"].decode(), count=1, flags=re.S)
    if count != 1:
        raise RuntimeError("Could not build the stale-cache probe workbook")
    parts["xl/worksheets/sheet1.xml"] = xml.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in parts.items():
            dst.writestr(name, data)


def probe_sandbox(runner: DockerRunner, root: Path) -> dict:
    work = root / "probe" / "work"
    work.mkdir(parents=True)
    work.chmod(0o777)
    budget = Budget(1e9)
    code, output = runner.run(work, ["python", "-c", "import openpyxl, pandas, numpy, socket;"
                                     "print('imports-ok');"
                                     "s=socket.socket();s.settimeout(3);"
                                     "print('network-open' if s.connect_ex(('1.1.1.1',53))==0 else 'network-blocked')"],
                              budget, "doctor")
    report = {"python_imports": "imports-ok" in output, "network_blocked": "network-open" not in output,
              "exit_code": code}
    _stale_workbook(work / "case_0_output.xlsx")
    try:
        start = time.perf_counter()
        runner.recalculate(work, budget)
        import openpyxl
        with (work / "recalculated" / "case_0_output.xlsx").open("rb") as stream:
            values = openpyxl.load_workbook(stream, data_only=True).active
            report["recalc_stale_formula"] = values["B1"].value
            report["recalc_dynamic_array"] = values["C1"].value
        report["recalc_seconds"] = round(time.perf_counter() - start, 1)
        report["recalc_ok"] = report["recalc_stale_formula"] == 42
    except TaskFailure as exc:
        report["recalc_ok"], report["recalc_error"] = False, str(exc)[-500:]


    from rouge.support import SUPPORT_KINDS, SUPPORT_VERSION, SupportProvider
    shutil.copyfile(work / "case_0_output.xlsx", work / "case_0_input.xlsx")
    provider = SupportProvider(runner, budget, Events(root / "support_events.jsonl"),
                               {"support_chars": 3000})
    report["input_observation_version"] = SUPPORT_VERSION
    report["input_observations"] = {}
    for kind in SUPPORT_KINDS:
        observation = provider.observe(kind, work)
        report["input_observations"][kind] = {
            "success": observation.success, "error": observation.error,
            "facts_chars": len(observation.facts), "returncode": observation.returncode}
    return report


def recalc_audit(runner: DockerRunner, dataset: Dataset, count: int, root: Path) -> dict:

    scorer, budget, failures = OfficialScorer(dataset.scorer_path), Budget(1e12), []
    tasks = dataset.split("train", count)
    for task in tasks:
        work = root / "audit" / task.id / "work"
        work.mkdir(parents=True)
        work.chmod(0o777)
        for i, case in enumerate(task.cases):
            shutil.copyfile(case.answer, work / f"case_{i}_output.xlsx")
        try:
            runner.recalculate(work, budget)
            ok = all(scorer.compare(task, case, work / "recalculated" / f"case_{i}_output.xlsx")
                     for i, case in enumerate(task.cases))
            if not ok:
                failures.append({"task": task.id, "reason": "golden fails after recalculation"})
        except Exception as exc:
            failures.append({"task": task.id, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
    return {"tasks": len(tasks), "pass": len(tasks) - len(failures), "failures": failures,
            "note": "Reference answers, not agent output: upper bound of the recalculation pipeline."}


def probe_model(config: dict, root: Path) -> dict:
    from rouge.model import make_model
    from rouge.parsing import strip_thinking
    model = make_model(config["model"], Events(root / "model_probe.jsonl"))
    start = time.perf_counter()
    text = model.chat([{"role": "user", "content": "Reply with the single word OK."}], budget=Budget(1e9),
                      seed=0, max_tokens=16, temperature=0.0, purpose="doctor")
    clean = strip_thinking(text).strip()
    return {"reply": clean[:80], "ok": clean == "OK", "seconds": round(time.perf_counter() - start, 2),
            "served_name": getattr(model, "name", None)}


def doctor(config: dict, deep: bool, audit: int) -> dict:
    model_cfg, report = config["model"], {"mode": "doctor"}
    backend = model_cfg.get("backend", "hf")
    needed = ("openpyxl", "numpy", "tqdm") + (("torch", "transformers", "accelerate") if backend == "hf" else ())
    report["packages"] = {name: importlib.util.find_spec(name) is not None for name in needed}
    path = Path(model_cfg["path"])
    if backend == "hf":
        report["model_weights"] = {"path": str(path), "config": (path / "config.json").is_file(),
                                   "safetensors": any(path.glob("*.safetensors"))}
        try:
            from rouge.model import preflight_local_model
            report["model_runtime"] = {"ok": True, **preflight_local_model(model_cfg)}
        except Exception as exc:
            report["model_runtime"] = {"ok": False, "error": str(exc)[:1000]}
    else:
        try:
            key = os.environ.get(model_cfg.get("api_key_env") or "", "")
            request = urllib.request.Request(model_cfg["base_url"].rstrip("/") + "/models",
                                             headers={"Authorization": f"Bearer {key}"} if key else {})
            with urllib.request.urlopen(request, timeout=10) as r:
                report["model_server"] = {"reachable": True,
                                          "models": [x["id"] for x in json.load(r).get("data", [])]}
        except (OSError, ValueError, KeyError) as exc:
            report["model_server"] = {"reachable": False, "error": str(exc)[:300]}
        tokenizer = Path(model_cfg.get("tokenizer_path") or path)
        report["tokenizer"] = {"path": str(tokenizer), "found": (tokenizer / "tokenizer_config.json").is_file(),
                               "note": "without it prompt tokens are over-estimated (budget stays safe)"}
    dataset = None
    try:
        dataset = Dataset(Path(config["dataset_manifest"]))
        report["dataset"] = {"valid": True, "audited": bool(dataset.metadata.get("scorer_compatibility_verified")),
                             "tasks": len(dataset.tasks), "splits": {k: len(v) for k, v in dataset.splits.items()},
                             "normalized_targets": any(t.targets for t in dataset.tasks.values()),
                             "fingerprint": dataset.fingerprint}
    except (OSError, ValueError, KeyError) as exc:
        report["dataset"] = {"valid": False, "error": str(exc)[:500]}
    root = Path(tempfile.mkdtemp(prefix="rouge-doctor-"))
    try:
        runner = DockerRunner(config["docker"], Events(root / "events.jsonl"))
        try:
            runner.check()
            report["docker"] = {"image_ready": True, **runner.describe()}
        except InfrastructureError as exc:
            report["docker"] = {"image_ready": False, "error": str(exc)[-800:]}
        if deep and report["docker"]["image_ready"]:
            report["sandbox"] = probe_sandbox(runner, root)
            if report["sandbox"].get("recalc_dynamic_array") != 5:
                report["sandbox"]["error"] = "dynamic-array functions not evaluated (LibreOffice < 24.8?): not ready"
            if audit and dataset is not None:
                report["recalc_audit"] = recalc_audit(runner, dataset, audit, root)
        if deep:
            try:
                report["model_probe"] = probe_model(config, root)
            except Exception as exc:
                report["model_probe"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    report["ready_checks"] = readiness(report, backend, deep, audit)
    report["failed_checks"] = [k for k, ok in report["ready_checks"].items() if not ok]
    report["ready"] = not report["failed_checks"]
    return report


def readiness(report: dict, backend: str, deep: bool, audit: int) -> dict[str, bool]:

    dataset, sandbox = report["dataset"], report.get("sandbox", {})
    checks = {"packages": all(report["packages"].values()),
              "dataset_audited": bool(dataset.get("valid") and dataset.get("audited")),
              "docker_image": bool(report["docker"]["image_ready"]),
              "model": bool(report["model_weights"]["config"] if backend == "hf"
                            else report["model_server"]["reachable"])}
    if backend == "hf":
        checks["model_runtime"] = bool(report.get("model_runtime", {}).get("ok"))
    if deep:
        from rouge.support import SUPPORT_KINDS, SUPPORT_VERSION
        observed = sandbox.get("input_observations", {})
        checks.update(sandbox_imports=bool(sandbox.get("python_imports")),
                      network_blocked=bool(sandbox.get("network_blocked")),
                      recalc_stale_formula=bool(sandbox.get("recalc_ok")),

                      recalc_dynamic_array=sandbox.get("recalc_dynamic_array") == 5,
                      input_observations=(sandbox.get("input_observation_version") == SUPPORT_VERSION
                                          and all(observed.get(k, {}).get("success") for k in SUPPORT_KINDS)),
                      model_reply=report.get("model_probe", {}).get("reply", "").strip() == "OK")
        if audit:
            result = report.get("recalc_audit", {})
            checks["recalc_audit"] = bool(result.get("tasks")) and not result.get("failures", [True])
    return checks


def set_value(config: dict, assignment: str) -> None:

    key, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    *parents, leaf = key.split(".")
    node = config
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise ValueError(f"Unknown config section in --set: {key}")
        node = node[part]
    if leaf not in node and (not parents or parents[0] not in FREE_SECTIONS):
        raise ValueError(f"Unknown config key in --set: {key} (typo?)")
    node[leaf] = value


def require_current_method(config):
    validate(config)
    if config['method'] != 'complement' or not all(config['complement'].get(key) for key in (
            'demand_aware', 'response_guided', 'frontier_guided', 'expedition_guided', 'campaign_guided')):
        raise ValueError('Use the current RogueEvo configuration in configs/rogueevo.json')


def run(config, args, output):
    from rouge.campaign_evolution import CampaignProposer
    from rouge.campaign_search import CampaignComplementSearch
    from rouge.execution import SpreadsheetEvaluator
    from rouge.model import make_model

    require_current_method(config)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    events = Events(output / 'events.jsonl')
    try:
        dataset = Dataset(Path(config['dataset_manifest']))
        if not dataset.metadata.get('scorer_compatibility_verified'):
            raise ValueError('Run validate_data.py first: scorer audit missing')
        runner = DockerRunner(config['docker'], events)
        runner.check()
        write_json(output / 'config.json', config)
        sandbox = runner.describe()
        write_json(output / 'provenance.json', {**provenance(args), 'docker': sandbox,
                                               'dataset_fingerprint': dataset.fingerprint})
        model = make_model(config['model'], events)
        budget = Budget(float(config['budget']['development']), float(config['budget']['tool_price']))
        if config['search'].get('max_seconds', 0):
            budget.deadline = time.monotonic() + config['search']['max_seconds']
        evaluator = SpreadsheetEvaluator(dataset, model, runner, config['execution'], output, events,
                                         int(config['seed']), budget, splits=('train', 'validation'))
        identity = model_identity(config, model)
        proposer = CampaignProposer(model, budget, config['proposal'], events)
        result = CampaignComplementSearch(config, evaluator, proposer, events, output,
                                           resolve_build(config['start_build'])).run()
        result = {**result, 'label': args.label or 'RogueEvo', 'mode': 'search',
                  'seed': int(config['seed']), 'model': identity,
                  'evidence_type': 'fake_test_server' if identity.get('served_name') == FAKE_MODEL else 'local_model_run',
                  'dataset_name': dataset.metadata['dataset'], 'protocol': dataset.metadata['protocol'],
                  'dataset_fingerprint': dataset.fingerprint,
                  'execution_protocol': execution_protocol(config, runner, sandbox, dataset),
                  'wall_seconds': time.monotonic() - started}
        write_json(output / 'final_info.json', result)
    except BaseException as exc:
        write_json(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)[:4000],
                                            'mode': 'search', 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        raise
    print(f"search {result['label']}: status={result['status']} "
          f"score={result.get('final_validation_score')} spent={result['budget']['spent']:.0f} -> {output}", flush=True)
    return 2 if result.get('partial') else 0


def main():
    parser = argparse.ArgumentParser(description='RogueEvo: environment checks and budgeted harness search.')
    parser.add_argument('--config', type=Path, default=HERE / 'configs/rogueevo.json')
    parser.add_argument('--mode', choices=('doctor', 'search'), default='doctor')
    parser.add_argument('--out-dir', '--out_dir', dest='out_dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--label')
    parser.add_argument('--deep', action='store_true')
    parser.add_argument('--recalc-audit', type=int, default=0, metavar='N')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.seed is not None:
            config['seed'] = args.seed
        if args.model_path:
            config['model']['path'] = str(args.model_path.expanduser().resolve())
        for assignment in args.set:
            set_value(config, assignment)
        require_current_method(config)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    config['dataset_manifest'] = resolve_path(config['dataset_manifest'])
    config['model']['path'] = resolve_path(config['model']['path'])
    if config['model'].get('tokenizer_path'):
        config['model']['tokenizer_path'] = resolve_path(config['model']['tokenizer_path'])
    if args.recalc_audit and not (args.mode == 'doctor' and args.deep):
        parser.error('--recalc-audit requires --mode doctor --deep')
    if args.mode == 'doctor':
        report = doctor(config, args.deep, args.recalc_audit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['ready'] else 1
    if args.out_dir is None:
        parser.error('search requires --out-dir')
    output = args.out_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new empty output directory for search')
    return run(config, args, output)


if __name__ == '__main__':
    sys.exit(main())
