"""Private spool adapter. Only the separate enforcing Docker service executes Python."""

import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langsmith import traceable

from backend.config import settings
from backend.storage import files


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ready() -> dict[str, Any]:
    try:
        health: dict[str, Any] = json.loads((settings().sandbox_queue / "health.json").read_text())
        checksum = digest(json.dumps(health["runtime"], sort_keys=True, separators=(",", ":")).encode())
        if (
            health["enforced"] is not True
            or time.time() - health["at"] > 5
            or checksum != health["runtime_hash"]
        ):
            raise ValueError("Stale or mismatched runtime")
        return health
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(
            "The isolated visualization runner is unavailable. Check the sandbox service and retry."
        ) from error


def ordinary(path: Path) -> bytes:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > 5 * 1024 * 1024:
        raise ValueError("Runner output must be a bounded regular file without links.")
    return path.read_bytes()


@traceable(name="exploration-python", run_type="tool")
def run(identity: UUID, code: str, csv: bytes, inputs: dict[str, bytes], runtime_hash: str) -> dict[str, Any]:
    """Idempotent script delivery. Results are retained independently of graph checkpoints."""
    store = files()
    receipt_key = f"exploration/{identity}/receipt.json"
    receipt_path = store.path(receipt_key)
    request_hash = digest(
        code.encode()
        + csv
        + b"".join(name.encode() + digest(raw).encode() for name, raw in sorted(inputs.items()))
    )
    if receipt_path.exists():
        receipt: dict[str, Any] = json.loads(receipt_path.read_bytes())
        if receipt["request_hash"] != request_hash or receipt["runtime_hash"] != runtime_hash:
            raise ValueError("A resumed script differs from its original inputs or runtime.")
        return receipt
    if ready()["runtime_hash"] != runtime_hash:
        raise RuntimeError("The sandbox runtime changed during this turn. Start a new message.")
    job = settings().sandbox_queue / str(identity)
    if not job.exists():
        temporary = settings().sandbox_queue / (str(identity) + ".preparing")
        temporary.mkdir(mode=0o700, exist_ok=True)
        bundle = temporary / "input" / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        (temporary / "input" / "table.csv").write_bytes(csv)
        for name, raw in inputs.items():
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        (temporary / "code.py").write_text(code)
        (temporary / "request.json").write_text(
            json.dumps(
                {
                    "timeout": 60,
                    "expires_at": time.time() + 300,
                    "runtime_hash": runtime_hash,
                    "request_hash": request_hash,
                }
            )
        )
        os.rename(temporary, job)
        (job / "ready").touch()
    else:
        spec = json.loads((job / "request.json").read_text())
        if spec.get("request_hash") != request_hash:
            raise ValueError("A queued script has different inputs.")
        (job / "ready").touch(exist_ok=True)
    deadline = time.monotonic() + 300
    while not (job / "result.json").exists():
        ready()
        if time.monotonic() > deadline:
            (job / "cancel").touch()
            raise RuntimeError("The visualization runner timed out. Retry to inspect the recorded attempt.")
        time.sleep(0.1)
    result: dict[str, Any] = json.loads((job / "result.json").read_text())
    if result["runtime_hash"] != runtime_hash:
        raise ValueError("Runner output has a different runtime identity.")
    saved: dict[str, dict[str, str]] = {}
    source = {"code.py": code.encode()}
    if (job / "output").is_dir():
        for path in (job / "output").iterdir():
            if path.suffix in {".png", ".csv", ".json", ".txt"}:
                source[path.name] = ordinary(path)
    if sum(map(len, source.values())) > 25 * 1024 * 1024:
        raise ValueError("Runner outputs exceeded the publication limit.")
    facts: dict[str, float] = {}
    if result["ok"]:
        try:
            parsed = json.loads(source.get("facts.json", b"{}"))
            if not isinstance(parsed, dict) or len(parsed) > 500:
                raise ValueError("facts.json must contain at most 500 named finite numbers.")
            for key, value in parsed.items():
                if (
                    not isinstance(key, str)
                    or len(key) > 200
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("facts.json must be a flat object of finite numeric values.")
                facts[key] = float(value)
        except (ValueError, TypeError) as error:
            result["ok"] = False
            result["stderr"] += f"\nInvalid facts.json: {error}"
    for name, raw in source.items():
        key = f"exploration/{identity}/{name}"
        store.put(key, raw)
        saved[name] = {"key": key, "sha256": digest(raw)}
    receipt = {**result, "request_hash": request_hash, "files": saved, "facts": facts}
    store.put(receipt_key, json.dumps(receipt, allow_nan=False).encode())
    return receipt
