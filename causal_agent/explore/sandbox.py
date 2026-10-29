"""Execute generated Python only in disposable Docker containers. No native runner or fallback."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from causal_agent.common import config, trace
from causal_agent.explore import queue

KEEP = ("code.py", "figure.png", "facts.json")
KEEP_SUFFIXES = (".csv", ".json", ".png", ".txt")  # what else a script may leave: a table, a number file, a picture, a note
MAX_FILE = 5 * 1024 * 1024  # a kept file larger than this is swept: a table of facts is small
CAP = 20_000  # the tail of stdout and of stderr that comes back
MPL_DIR = ".mpl"
CSV_IN_CONTAINER = "/data/table.csv"


class Outcome(BaseModel):
    ok: bool
    stdout: str  # the last CAP characters the script printed
    stderr: str  # the last CAP characters it wrote to stderr, or why the fence refused to run it
    files: list[str]
    kind: str


def run(code: str, csv: Path, out_dir: Path, timeout: float = 60, *, inputs: Path | None = None, cancel: threading.Event | None = None) -> Outcome:
    """The host launches Docker only. Inputs are read-only; application credentials never enter the container."""
    box = config.get().sandbox
    with trace.span("sandbox", run_type="tool", inputs={"kind": "docker", "image": box.image, "out_dir": str(out_dir), "timeout": timeout}) as span:
        trace.write(out_dir / "code.py", code)
        try:
            ok, stdout, stderr = _in_container(csv, out_dir, timeout, box.image, inputs=inputs, cancel=cancel)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            ok, stdout, stderr = False, "", f"Docker sandbox unavailable: {exc}"
        _sweep(out_dir)
        outcome = Outcome(ok=ok, stdout=stdout, stderr=stderr, files=sorted(p.name for p in out_dir.iterdir()), kind="docker")
        span.end(outputs=outcome.model_dump())
        return outcome


Ran = tuple[bool, str, str]


def require_runtime(image: str | None = None) -> tuple[str, str]:
    """Validate the installed Docker runtime before spending a model budget."""
    box = config.get().sandbox
    if box.queue:
        queue.require(Path(box.queue))
        return "docker-worker", box.image
    image = image or box.image
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required: install Docker Desktop or Docker Engine and build the image with make explore-image")
    probe = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=10)
    if probe.returncode != 0:
        raise RuntimeError("Docker is not running: " + (probe.stderr or probe.stdout).strip()[-300:])
    inspected = subprocess.run([docker, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, timeout=10)
    if inspected.returncode != 0:
        raise RuntimeError(f"Docker sandbox image {image!r} is unavailable. Run make explore-image before exploring data.")
    # Resolve the tag once so an image replacement cannot change a task midway through launch.
    return docker, inspected.stdout.strip()


def _in_container(csv: Path, out_dir: Path, timeout: float, image: str, *, inputs: Path | None = None, cancel=None) -> Ran:
    if spool := config.get().sandbox.queue:
        return queue.run(Path(spool), csv, out_dir, timeout, inputs, cancel)
    docker, image = require_runtime(image)
    uid = os.getuid() if os.name == "posix" else 10001
    gid = os.getgid() if os.name == "posix" else 10001
    if uid == 0:
        uid = gid = 10001
        os.chown(out_dir, uid, gid)
    container_name = "explore-" + uuid4().hex
    cmd = [
        docker,
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        container_name,
        "--label",
        f"causal.explore.attempt={os.environ.get('EXPLORE_ATTEMPT_ID', container_name)}",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        *(["-v", f"{inputs.resolve()}:/inputs:ro", "-e", "INPUTS_DIR=/inputs"] if inputs else []),
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "OMP_NUM_THREADS=1",
        "--network",
        "none",
        "--memory",
        "1g",
        "--cpus",
        "1",
        "--user",
        f"{uid}:{gid}",
        "-v",
        f"{Path(csv).resolve()}:{CSV_IN_CONTAINER}:ro",
        "-v",
        f"{out_dir.resolve()}:/work",
        "-w",
        "/work",
        "-e",
        f"DATA_CSV={CSV_IN_CONTAINER}",
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
    try:
        return _call(cmd, out_dir, {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}, timeout, cancel=cancel)
    finally:
        subprocess.run([docker, "rm", "-f", container_name], capture_output=True, timeout=10)


def _call(cmd: list[str], cwd: Path, env: dict[str, str], timeout: float, *, cancel=None) -> Ran:
    """Bound pipe memory on every platform; terminate the sandbox launcher on cancellation."""
    if cancel is not None and cancel.is_set():
        return False, "", "cancelled"
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        env = {**env, "SYSTEMROOT": os.environ["SYSTEMROOT"]}
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name == "posix")
    tails = {"stdout": b"", "stderr": b""}
    lock = threading.Lock()

    def drain(name, stream):
        while chunk := stream.read(65536):
            with lock:
                tails[name] = (tails[name] + chunk)[-CAP:]

    readers = [threading.Thread(target=drain, args=(name, stream), daemon=True) for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr))]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    failure = ""
    try:
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                failure = "cancelled"
                break
            if time.monotonic() >= deadline:
                failure = f"timed out after {timeout:g}s"
                break
            try:
                size = sum(p.lstat().st_size for p in cwd.rglob("*") if p.is_file() and not p.is_symlink())
            except OSError:
                size = 0  # generated files may be removed while the task runs
            if size > 25 * 1024 * 1024:
                failure = "task output exceeded 25 MiB"
                break
            time.sleep(0.05)
    finally:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            elif proc.poll() is None:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()
        for reader in readers:
            reader.join(timeout=1)
        for stream in (proc.stdout, proc.stderr):
            if stream and all(not r.is_alive() for r in readers):
                stream.close()
    with lock:
        stdout, stderr = (tails[k].decode(errors="replace") for k in ("stdout", "stderr"))
    return not failure and proc.returncode == 0, stdout, (stderr + "\n" + failure).strip()


def _sweep(out_dir: Path) -> None:
    """Keep the script, the picture, the numbers, and any small table, number file, picture or note the script wrote; drop
    the rest: the matplotlib cache, pickles, anything large."""
    for p in out_dir.iterdir():
        mode = p.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)) or p.is_file() and p.stat().st_nlink > 1:
            p.unlink()

            continue
        if p.is_file() and p.name in KEEP and p.stat().st_size <= MAX_FILE:
            continue
        if p.is_file() and not p.is_symlink() and p.suffix in KEEP_SUFFIXES and p.stat().st_size <= MAX_FILE:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()


def cancel_attempt(attempt_id: str) -> None:
    """The trusted A2A process owner reaps only this attempt's containers after worker exit."""
    docker = shutil.which("docker")
    if docker is None:
        return
    found = subprocess.run([docker, "ps", "-aq", "--filter", f"label=causal.explore.attempt={attempt_id}"], capture_output=True, text=True, timeout=10)
    ids = [value for value in found.stdout.split() if len(value) == 12 and all(c in "0123456789abcdef" for c in value)]
    if ids:
        subprocess.run([docker, "rm", "-f", *ids], capture_output=True, timeout=10)
