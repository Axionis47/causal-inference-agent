"""Runs one drawing script where the config says: in a subprocess (the same interpreter, isolated mode, a scrubbed
environment) or in a container (the image from `make viz-image`, no network, the CSV read-only, the artifact folder as the
working directory). Either way: a timeout, and the folder swept afterwards so only code.py, figure.png and facts.json stay.
With the container chosen and no runtime at hand the run is refused, never quietly downgraded."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from causal_agent.common import config

KEEP = ("code.py", "figure.png", "facts.json")
MPL_DIR = ".mpl"
CSV_IN_CONTAINER = "/data/table.csv"


class Outcome(BaseModel):
    ok: bool
    stderr: str
    files: list[str]


def run(code: str, csv: Path, out_dir: Path, timeout: float = 60) -> Outcome:
    """Write the script as code.py in out_dir and run it there. The script sees the CSV path as VIZ_CSV and nothing else
    of the caller's environment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code.py").write_text(code)
    box = config.get().sandbox
    if box.kind == "docker":
        ok, stderr = _in_container(csv, out_dir, timeout, box.image)
    elif box.kind == "subprocess":
        ok, stderr = _in_subprocess(csv, out_dir, timeout)
    else:
        ok, stderr = False, f"VIZ_SANDBOX={box.kind!r} is not a sandbox this tool knows (subprocess or docker)"
    _sweep(out_dir)
    return Outcome(ok=ok, stderr=stderr, files=sorted(p.name for p in out_dir.iterdir()))


def _in_subprocess(csv: Path, out_dir: Path, timeout: float) -> tuple[bool, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(out_dir),
        "VIZ_CSV": str(Path(csv).resolve()),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(out_dir / MPL_DIR),
    }
    return _call([sys.executable, "-I", "code.py"], out_dir, env, timeout)


def _in_container(csv: Path, out_dir: Path, timeout: float, image: str) -> tuple[bool, str]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "VIZ_SANDBOX=docker but no docker binary is on the PATH"
    probe = subprocess.run([docker, "info"], capture_output=True, text=True)
    if probe.returncode != 0:
        return False, "VIZ_SANDBOX=docker but the container runtime is not running: " + (probe.stderr or probe.stdout).strip()[-300:]
    cmd = [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        "1g",
        "--cpus",
        "1",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{Path(csv).resolve()}:{CSV_IN_CONTAINER}:ro",
        "-v",
        f"{out_dir.resolve()}:/work",
        "-w",
        "/work",
        "-e",
        f"VIZ_CSV={CSV_IN_CONTAINER}",
        "-e",
        "MPLBACKEND=Agg",
        "-e",
        f"MPLCONFIGDIR=/work/{MPL_DIR}",
        "-e",
        "HOME=/work",
        image,
        "python",
        "-I",
        "code.py",
    ]
    return _call(cmd, out_dir, {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}, timeout)


def _call(cmd: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[bool, str]:
    try:
        done = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return done.returncode == 0, done.stderr
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:g}s"


def _sweep(out_dir: Path) -> None:
    for p in out_dir.iterdir():
        if p.name in KEEP:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
