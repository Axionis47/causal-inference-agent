"""One bounded script at a time in a persistent, network-disabled Docker container.

The private queue is accessible to the trusted supervisor only. The child has
kernel-enforced access to its own inputs, scratch and installed libraries.
"""

import json
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

QUEUE = Path("/queue")
TASKS = Path("/runtime/task")
CAP = 20000


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    os.replace(temp, path)


def copy_inputs(source, dest):
    total = 0
    dest.mkdir()
    for p in source.rglob("*"):
        mode = p.lstat().st_mode
        target = dest / p.relative_to(source)
        if stat.S_ISDIR(mode):
            target.mkdir(exist_ok=True)
        elif stat.S_ISREG(mode) and p.stat().st_nlink == 1:
            total += p.stat().st_size
            if total > 400 * 1024 * 1024:
                raise ValueError("Input bundle exceeds 400 MiB")
            shutil.copyfile(p, target)
            target.chmod(0o444)
        else:
            raise ValueError("Only ordinary input files are allowed")


def execute(job):
    task = TASKS
    for old in task.iterdir():
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old)
        else:
            old.unlink()
    work, inputs = task / "work", task / "input"
    work.mkdir()
    (work / "tmp").mkdir()
    copy_inputs(job / "input", inputs)
    code = (job / "code.py").read_bytes()
    if len(code) > 100000:
        raise ValueError("Script exceeds 100 KB")
    (work / "code.py").write_bytes(code)
    for path in (work, work / "tmp", work / "code.py"):
        os.chown(path, 20000, 20000)
    spec = json.loads((job / "request.json").read_text())
    if spec.get("expires_at", time.time() + 60) < time.time():
        raise ValueError("Job expired before execution")
    timeout = min(60, max(0.1, float(spec["timeout"])))
    proc = subprocess.Popen(
        ["python", "-I", "/opt/worker/isolation.py", str(work), str(inputs)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={"PATH": "/usr/local/bin:/usr/bin"},
    )
    tails = {"stdout": b"", "stderr": b""}

    def drain(key, stream):
        with stream:
            while chunk := stream.read(65536):
                tails[key] = (tails[key] + chunk)[-CAP:]

    readers = [threading.Thread(target=drain, args=(k, s), daemon=True) for k, s in (("stdout", proc.stdout), ("stderr", proc.stderr))]
    for reader in readers:
        reader.start()
    deadline, failure = time.monotonic() + timeout, ""
    try:
        while proc.poll() is None:
            if (job / "cancel").exists() or not job.exists():
                failure = "cancelled"
                break
            if time.monotonic() > deadline:
                failure = f"timed out after {timeout:g}s"
                break
            try:
                size = sum(p.lstat().st_size for p in work.rglob("*") if not p.is_symlink())
                if size > 25 * 1024 * 1024:
                    failure = "task output exceeded 25 MiB"
                    break
            except OSError:
                pass
            time.sleep(0.05)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        for reader in readers:
            reader.join(timeout=2)
    output = job / "output"
    output.mkdir()
    os.chown(output, 10001, 10001)
    for p in work.iterdir():
        st = p.lstat()
        if stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size <= 5 * 1024 * 1024 and p.suffix in {".json", ".csv", ".png", ".txt"}:
            shutil.copyfile(p, output / p.name)
    return {
        "ok": proc.returncode == 0 and not failure,
        "stdout": tails["stdout"].decode(errors="replace"),
        "stderr": (tails["stderr"].decode(errors="replace") + "\n" + failure).strip(),
    }


def main():
    if not Path("/.dockerenv").exists():
        raise RuntimeError("This launcher only runs inside Docker")
    QUEUE.mkdir(exist_ok=True)
    # Reclaim ownership before chmod on an existing volume, without CAP_FOWNER.
    os.chown(QUEUE, 0, 0)
    QUEUE.chmod(0o700)
    os.chown(QUEUE, 10001, 10001)
    TASKS.mkdir(exist_ok=True)
    # Probe the actual restrictions before advertising readiness.
    probe = QUEUE / "probe"
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir()
    (probe / "input").mkdir()
    (probe / "input" / "table.csv").write_text("x\n1\n")
    (probe / "code.py").write_text(
        "import os, socket\nassert os.getuid()==20000\ntry: socket.socket()\nexcept PermissionError: pass\nelse: raise RuntimeError('network not fenced')\ntry: open('/queue/health.json')\nexcept FileNotFoundError: pass\nelse: raise RuntimeError('queue not fenced')\n"
    )
    (probe / "request.json").write_text('{"timeout":10}')
    result = execute(probe)
    shutil.rmtree(probe)
    for old in TASKS.iterdir():
        shutil.rmtree(old)
    if not result["ok"]:
        raise RuntimeError("Sandbox enforcement unavailable: " + result["stderr"])

    def heartbeat():
        while True:
            write(QUEUE / "health.json", {"at": time.time(), "enforced": True})
            time.sleep(1)

    threading.Thread(target=heartbeat, daemon=True).start()
    while True:
        for job in sorted(QUEUE.iterdir()):
            if job.is_dir() and time.time() - job.stat().st_mtime > 900:
                shutil.rmtree(job)
                continue
            if not job.is_dir() or not (job / "ready").exists() or (job / "result.json").exists():
                continue
            try:
                if (job / "started").exists():
                    raise RuntimeError("Worker restarted during execution; task was not replayed")
                (job / "started").touch()
                result = execute(job)
            except Exception as exc:
                result = {"ok": False, "stdout": "", "stderr": f"Docker worker failed: {exc}"}
            finally:
                for old in TASKS.iterdir():
                    if old.is_dir() and not old.is_symlink():
                        shutil.rmtree(old, ignore_errors=True)
                    else:
                        old.unlink(missing_ok=True)
            if job.exists():
                write(job / "result.json", result)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
