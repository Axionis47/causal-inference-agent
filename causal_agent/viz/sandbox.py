"""Runs one drawing script inside the strongest fence the machine has. `VIZ_SANDBOX=auto` (the default) picks by code: the
macOS seatbelt (`sandbox-exec`: no network, reads limited to the interpreter, the system, the CSV and the artifact folder,
writes to the artifact folder and the temp folders), `bwrap` or `unshare -rn` on Linux (no network; bwrap also mounts the file
system read-only but the artifact folder), else a plain subprocess (the same interpreter, isolated mode, a scrubbed environment,
isolation by environment only). `docker` runs the image from `make viz-image`. Either way: a timeout, and the folder swept
afterwards so only code.py, figure.png and facts.json stay. The kind that ran is on the outcome, and the artifact records it.
With a kind chosen by name and not at hand the run is refused, never quietly downgraded."""

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
PROFILE = ".sandbox.sb"
CSV_IN_CONTAINER = "/data/table.csv"
KINDS = ("auto", "seatbelt", "bwrap", "unshare", "subprocess", "docker")
SEATBELT = "/usr/bin/sandbox-exec"


class Outcome(BaseModel):
    ok: bool
    stderr: str
    files: list[str]
    kind: str


def resolve(kind: str) -> str:
    """The sandbox that will run: `auto` becomes the strongest one at hand; a name stays a name."""
    if kind != "auto":
        return kind
    if sys.platform == "darwin" and Path(SEATBELT).exists():
        return "seatbelt"
    if sys.platform.startswith("linux"):
        if shutil.which("bwrap"):
            return "bwrap"
        if shutil.which("unshare"):
            return "unshare"
    return "subprocess"


def run(code: str, csv: Path, out_dir: Path, timeout: float = 60) -> Outcome:
    """Write the script as code.py in out_dir and run it there. The script sees the CSV path as VIZ_CSV and nothing else
    of the caller's environment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code.py").write_text(code)
    box = config.get().sandbox
    kind = resolve(box.kind)
    if kind == "docker":
        ok, stderr = _in_container(csv, out_dir, timeout, box.image)
    elif kind == "seatbelt":
        ok, stderr = _in_seatbelt(csv, out_dir, timeout)
    elif kind == "bwrap":
        ok, stderr = _in_bwrap(csv, out_dir, timeout)
    elif kind == "unshare":
        ok, stderr = _in_unshare(csv, out_dir, timeout)
    elif kind == "subprocess":
        ok, stderr = _in_subprocess(csv, out_dir, timeout)
    else:
        ok, stderr = False, f"VIZ_SANDBOX={box.kind!r} is not a sandbox this tool knows ({', '.join(KINDS)})"
    _sweep(out_dir)
    return Outcome(ok=ok, stderr=stderr, files=sorted(p.name for p in out_dir.iterdir()), kind=kind)


def _env(csv: Path, out_dir: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(out_dir),
        "VIZ_CSV": str(Path(csv).resolve()),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(out_dir / MPL_DIR),
    }


def _in_subprocess(csv: Path, out_dir: Path, timeout: float) -> tuple[bool, str]:
    return _call([sys.executable, "-I", "code.py"], out_dir, _env(csv, out_dir), timeout)


def _seatbelt_profile(csv: Path, out_dir: Path) -> str:
    """The seatbelt rules: everything allowed, then the network denied, then writes denied except the artifact folder and the
    temp folders, then reads under the home folder denied except the artifact folder, the CSV, the interpreter and its base.
    The last matching rule wins."""
    home = Path.home().resolve()
    out = out_dir.resolve()
    table = Path(csv).resolve()
    reads = [out, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), home / ".local" / "share" / "uv"]
    # the folders between the home folder and an allowed path: their names may be read, never their contents
    ancestors = {a for p in [*reads, table] for a in p.parents if a == home or home in a.parents}
    q = lambda p: '"' + str(p).replace("\\", "\\\\").replace('"', '\\"') + '"'  # noqa: E731
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
        f'(allow file-write* (subpath {q(out)}) (subpath "/private/var/folders") (subpath "/private/tmp") (subpath "/dev"))',
        f"(deny file-read* (subpath {q(home)}))",
        "(allow file-read-metadata " + " ".join(f"(literal {q(a)})" for a in sorted(ancestors)) + ")",
        f"(allow file-read* (literal {q(table)}) " + " ".join(f"(subpath {q(p)})" for p in reads) + ")",
    ]
    return "\n".join(lines) + "\n"


def _in_seatbelt(csv: Path, out_dir: Path, timeout: float) -> tuple[bool, str]:
    if not Path(SEATBELT).exists():
        return False, "VIZ_SANDBOX=seatbelt but sandbox-exec is not on this machine"
    profile = out_dir / PROFILE
    profile.write_text(_seatbelt_profile(csv, out_dir))
    return _call([SEATBELT, "-f", str(profile), sys.executable, "-I", "code.py"], out_dir, _env(csv, out_dir), timeout)


def _in_bwrap(csv: Path, out_dir: Path, timeout: float) -> tuple[bool, str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return False, "VIZ_SANDBOX=bwrap but bwrap is not on the PATH"
    out = str(out_dir.resolve())
    cmd = [
        bwrap,
        "--unshare-net",
        "--die-with-parent",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", out, out,
        "--ro-bind", str(Path(csv).resolve()), str(Path(csv).resolve()),
        "--chdir", out,
        sys.executable, "-I", "code.py",
    ]  # fmt: skip
    return _call(cmd, out_dir, _env(csv, out_dir), timeout)


def _in_unshare(csv: Path, out_dir: Path, timeout: float) -> tuple[bool, str]:
    unshare = shutil.which("unshare")
    if unshare is None:
        return False, "VIZ_SANDBOX=unshare but unshare is not on the PATH"
    return _call([unshare, "-rn", sys.executable, "-I", "code.py"], out_dir, _env(csv, out_dir), timeout)


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
