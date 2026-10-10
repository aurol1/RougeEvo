from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .core import Budget, Events, InfrastructureError

DOCKER_FAILURES = (125, 126, 127)


class TaskFailure(ValueError):
    pass


class DockerRunner:
    def __init__(self, config: dict, events: Events):
        self.docker = config.get("binary", "docker")
        self.image = config["image"]
        self.timeout = int(config.get("timeout_seconds", 60))
        self.memory = config.get("memory", "2g")
        self.cpus = str(config.get("cpus", 1))
        self.pids = str(config.get("pids_limit", 128))
        self.events = events

    def check(self) -> None:
        if shutil.which(self.docker) is None:
            raise InfrastructureError(f"{self.docker} CLI is missing; install Docker and build the runner image")
        result = subprocess.run([self.docker, "image", "inspect", self.image], capture_output=True,
                                text=True, timeout=30)
        if result.returncode:
            hint = (" (permission denied: add your user to the docker group or use rootless docker)"
                    if "permission denied" in result.stderr.lower() else "")
            raise InfrastructureError(f"Docker image {self.image!r} unavailable{hint}: {result.stderr[-1000:]}")

    def command(self, workspace: Path, name: str, args: list[str], nonce: str) -> list[str]:

        payload = (f"{shlex.join(args)} > /tmp/rouge_out 2>&1; c=$?; tail -c 16000 /tmp/rouge_out; "
                   f"echo; echo __ROUGE_EXIT_{nonce}=$c")
        return [self.docker, "run", "--rm", "--name", name, "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", self.pids, "--memory", self.memory, "--cpus", self.cpus,
                "--user", "65534:65534", "--tmpfs", "/tmp:rw,nosuid,size=256m",
                "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HOME=/tmp",
                "--mount", f"type=bind,source={workspace.resolve()},target=/work",
                "--workdir", "/work", self.image, "sh", "-c", payload]

    def run(self, workspace: Path, args: list[str], budget: Budget, purpose: str) -> tuple[int, str]:
        hold = budget.reserve(budget.tool_price)
        name, nonce = "rouge-" + uuid.uuid4().hex, uuid.uuid4().hex[:12]
        log = workspace.parent / (name + ".log")
        start, timed_out, code = time.perf_counter(), False, -1
        try:
            with log.open("wb") as stream:
                result = subprocess.run(self.command(workspace, name, args, nonce), stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=self.timeout)
                code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True, timeout=30)
        except OSError as exc:
            budget.release(hold)
            raise InfrastructureError(f"Cannot start docker: {exc}") from exc
        elapsed = time.perf_counter() - start
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 16000))
            output = stream.read().decode("utf-8", errors="replace")
        marks = re.findall(rf"__ROUGE_EXIT_{nonce}=(\d+)", output)
        output = re.sub(rf"\n?__ROUGE_EXIT_{nonce}=\d+\n?", "", output)
        if timed_out:
            code = 124
        elif marks:
            code = int(marks[-1])
        elif code in DOCKER_FAILURES:
            budget.release(hold)
            raise InfrastructureError(f"Docker failed ({code}) for {purpose}: {output[-1000:]}")

        budget.settle_tool(hold, elapsed, failed=code != 0)
        self.events.emit("tool_call", purpose=purpose, seconds=round(elapsed, 3), returncode=code,
                         timed_out=timed_out)
        return code, output

    def recalculate(self, workspace: Path, budget: Budget) -> list[Path]:
        originals = sorted(workspace.glob("case_*_output.xlsx"))
        if any(x.is_symlink() for x in originals):
            raise TaskFailure("Output workbooks must be regular files")
        if not originals:
            return []
        dest = workspace / "recalculated"
        if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
            raise TaskFailure("recalculated must be a plain directory")
        dest.mkdir(exist_ok=True)
        dest.chmod(0o777)
        for old in dest.iterdir():
            if old.is_dir() and not old.is_symlink():
                raise TaskFailure("Unexpected directory in recalculated outputs")
            old.unlink()

        command = ["sh", "-c", 'cp -r /opt/rouge-lo /tmp/lo && exec "$@"', "lo",
                   "libreoffice", "-env:UserInstallation=file:///tmp/lo", "--headless",
                   "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", "/work/recalculated",
                   *[f"/work/{x.name}" for x in originals]]
        code, output = self.run(workspace, command, budget, "formula_recalculation")
        paths = [dest / x.name for x in originals]
        if code != 0 or any(not x.is_file() or x.is_symlink() for x in paths):
            raise TaskFailure(f"Recalculation failed ({code}): {output[-500:]}")
        return paths

    def describe(self) -> dict:
        info = {"image": self.image}
        for key, cmd in (("image_id", [self.docker, "image", "inspect", "-f", "{{.Id}}", self.image]),
                         ("libreoffice", [self.docker, "run", "--rm", "--network", "none", self.image,
                                          "libreoffice", "--version"])):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                info[key] = result.stdout.strip()[-300:] or result.stderr.strip()[-300:]
            except (OSError, subprocess.SubprocessError) as exc:
                info[key] = f"unavailable: {type(exc).__name__}"
        return info
