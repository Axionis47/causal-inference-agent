"""Private spool client for the enforcing Compose Docker worker. Never executes code."""

import json
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

from causal_agent.common import trace


def require(queue: Path) -> None:
    try:
        health = json.loads((queue / "health.json").read_text())
        if health["enforced"] is not True or time.time() - health["at"] > 5:
            raise ValueError("stale worker")
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError("The enforcing Docker worker is unavailable. Check docker compose logs sandbox.") from exc


def run(queue: Path, csv: Path, out: Path, timeout: float, inputs: Path | None, cancel: threading.Event | None):
    require(queue)
    job = queue / uuid4().hex
    job.mkdir(mode=0o700)
    finished = False
    try:
        bundle = job / "input"
        bundle.mkdir()
        shutil.copyfile(csv, bundle / "table.csv")
        if inputs:
            shutil.copytree(inputs, bundle / "bundle")
        shutil.copyfile(out / "code.py", job / "code.py")
        trace.write(job / "request.json", json.dumps({"timeout": timeout, "expires_at": time.time() + 300}))
        (job / "ready").touch()
        deadline = time.monotonic() + 300  # bounded queue wait, script itself is capped at 60 s
        while not (job / "result.json").exists():
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            require(queue)
            if time.monotonic() >= deadline:
                raise RuntimeError("Docker worker queue timed out")
            time.sleep(0.05)
        result = json.loads((job / "result.json").read_text())
        if (job / "output").is_dir():
            for p in (job / "output").iterdir():
                if p.is_file() and not p.is_symlink():
                    trace.write(out / p.name, p.read_bytes())
        finished = True
        return result["ok"], result["stdout"], result["stderr"]
    finally:
        if not finished:
            (job / "cancel").touch()
            # Supervisor observes deletion as cancellation too; it owns no application paths.
        shutil.rmtree(job, ignore_errors=True)
