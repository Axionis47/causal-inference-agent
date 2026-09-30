"""The sandbox picks its runner from the config: `auto` becomes the strongest fence at hand, a name stays a name and is refused,
never downgraded, when it is not there. The fence holds: no network, no read of the home folder, no write outside the artifact
folder; the CSV is readable and the figure is written."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from causal_agent.viz import sandbox

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
HELLO = "import os\nopen('facts.json', 'w').write('{}')\nopen('figure.png', 'wb').write(b'x')\nprint(os.environ['VIZ_CSV'])\n"
READS_CSV = HELLO + "assert len(open(os.environ['VIZ_CSV']).read()) > 100\n"
NETWORK = "import socket\ns = socket.socket()\ns.settimeout(2)\ns.connect(('127.0.0.1', 9))\n"
HOME_LIST = "import os, pwd\nprint(os.listdir(pwd.getpwuid(os.getuid()).pw_dir))\n"
HOME_WRITE = "import os, pwd\nopen(os.path.join(pwd.getpwuid(os.getuid()).pw_dir, '.causal_desk_sandbox_probe'), 'w').write('x')\n"


def _daemon_up() -> bool:
    d = shutil.which("docker")
    return bool(d) and subprocess.run([d, "info"], capture_output=True).returncode == 0


def test_auto_is_the_default_and_picks_the_strongest_fence_at_hand(tmp_path, monkeypatch):
    monkeypatch.delenv("VIZ_SANDBOX", raising=False)
    out = sandbox.run(READS_CSV, CSV, tmp_path / "a")
    assert out.ok and out.files == ["code.py", "facts.json", "figure.png"]
    expected = "seatbelt" if sys.platform == "darwin" else "bwrap" if shutil.which("bwrap") else "unshare" if shutil.which("unshare") else "subprocess"
    assert out.kind == expected == sandbox.resolve("auto")


def test_the_subprocess_runner_runs_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "subprocess")
    out = sandbox.run(HELLO, CSV, tmp_path / "s")
    assert out.ok and out.kind == "subprocess" and out.files == ["code.py", "facts.json", "figure.png"]


def test_an_unknown_sandbox_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "vm")
    out = sandbox.run(HELLO, CSV, tmp_path / "b")
    assert not out.ok and "not a sandbox this tool knows" in out.stderr and out.files == ["code.py"] and out.kind == "vm"


def test_a_fence_named_and_not_at_hand_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "bwrap")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = sandbox.run(HELLO, CSV, tmp_path / "c")
    assert not out.ok and "bwrap is not on the PATH" in out.stderr and out.files == ["code.py"]


def test_the_container_is_refused_without_a_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "docker")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = sandbox.run(HELLO, CSV, tmp_path / "d")
    assert not out.ok and "no docker binary" in out.stderr and out.files == ["code.py"]


@pytest.mark.skipif(sys.platform != "darwin" or not Path(sandbox.SEATBELT).exists(), reason="the seatbelt is macOS only")
def test_the_seatbelt_blocks_the_network_the_home_folder_and_writes_outside_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "seatbelt")
    ok = sandbox.run(READS_CSV, CSV, tmp_path / "e")
    assert ok.ok and ok.kind == "seatbelt" and ok.files == ["code.py", "facts.json", "figure.png"]  # the CSV reads, the figure writes
    net = sandbox.run(HELLO + NETWORK, CSV, tmp_path / "f")
    assert not net.ok and ("Operation not permitted" in net.stderr or "PermissionError" in net.stderr)
    home = sandbox.run(HELLO + HOME_LIST, CSV, tmp_path / "g")
    assert not home.ok and ("Operation not permitted" in home.stderr or "PermissionError" in home.stderr)
    write = sandbox.run(HELLO + HOME_WRITE, CSV, tmp_path / "h")
    assert not write.ok and ("Operation not permitted" in write.stderr or "PermissionError" in write.stderr)
    assert not (Path.home() / ".causal_desk_sandbox_probe").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux") or not shutil.which("bwrap"), reason="bwrap is Linux only")
def test_bwrap_blocks_the_network_and_writes_outside_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "bwrap")
    ok = sandbox.run(READS_CSV, CSV, tmp_path / "i")
    assert ok.ok and ok.kind == "bwrap"
    net = sandbox.run(HELLO + NETWORK, CSV, tmp_path / "j")
    assert not net.ok
    write = sandbox.run(HELLO + HOME_WRITE, CSV, tmp_path / "k")
    assert not write.ok and not (Path.home() / ".causal_desk_sandbox_probe").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux") or not shutil.which("unshare"), reason="unshare is Linux only")
def test_unshare_blocks_the_network(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "unshare")
    ok = sandbox.run(READS_CSV, CSV, tmp_path / "l")
    assert ok.ok and ok.kind == "unshare"
    net = sandbox.run(HELLO + NETWORK, CSV, tmp_path / "m")
    assert not net.ok


@pytest.mark.skipif(not _daemon_up(), reason="no container runtime running")
def test_the_container_runs_the_script_with_the_csv_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "docker")
    out = sandbox.run(HELLO + "open(os.environ['VIZ_CSV'], 'a')\n", CSV, tmp_path / "n")
    assert not out.ok and "Read-only" in out.stderr  # the table cannot be written
    out = sandbox.run(HELLO, CSV, tmp_path / "o")
    assert out.ok and out.kind == "docker" and out.files == ["code.py", "facts.json", "figure.png"]
