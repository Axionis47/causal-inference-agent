"""Runs one drawing script in a subprocess. First cut: the same interpreter, isolated mode, a scrubbed environment, a
timeout, and the folder swept afterwards. A VM will take its place later with the same interface: run(code, csv, out_dir)
-> Outcome."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

KEEP = ("code.py", "figure.png", "facts.json")
MPL_DIR = ".mpl"


class Outcome(BaseModel):
    ok: bool
    stderr: str
    files: list[str]


def run(code: str, csv: Path, out_dir: Path, timeout: float = 60) -> Outcome:
    """Write the script as code.py in out_dir and run it there. The script sees the CSV path as VIZ_CSV and nothing else
    of the caller's environment. Afterwards only code.py, figure.png and facts.json stay in the folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code.py").write_text(code)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(out_dir),
        "VIZ_CSV": str(Path(csv).resolve()),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(out_dir / MPL_DIR),
    }
    try:
        done = subprocess.run([sys.executable, "-I", "code.py"], cwd=out_dir, env=env, capture_output=True, text=True, timeout=timeout)
        ok, stderr = done.returncode == 0, done.stderr
    except subprocess.TimeoutExpired:
        ok, stderr = False, f"timed out after {timeout:g}s"
    _sweep(out_dir)
    return Outcome(ok=ok, stderr=stderr, files=sorted(p.name for p in out_dir.iterdir()))


def _sweep(out_dir: Path) -> None:
    for p in out_dir.iterdir():
        if p.name in KEEP:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
