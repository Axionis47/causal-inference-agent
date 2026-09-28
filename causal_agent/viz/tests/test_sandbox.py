"""The sandbox picks its runner from the config: a subprocess by default, a container when asked; the container is refused,
never downgraded, when no runtime is there."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from causal_agent.viz import sandbox

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
HELLO = "import os\nopen('facts.json', 'w').write('{}')\nopen('figure.png', 'wb').write(b'x')\nprint(os.environ['VIZ_CSV'])\n"


def _daemon_up() -> bool:
    d = shutil.which("docker")
    return bool(d) and subprocess.run([d, "info"], capture_output=True).returncode == 0


def test_the_subprocess_runner_is_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VIZ_SANDBOX", raising=False)
    out = sandbox.run(HELLO, CSV, tmp_path / "a")
    assert out.ok and out.files == ["code.py", "facts.json", "figure.png"]


def test_an_unknown_sandbox_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "vm")
    out = sandbox.run(HELLO, CSV, tmp_path / "b")
    assert not out.ok and "not a sandbox this tool knows" in out.stderr and out.files == ["code.py"]


def test_the_container_is_refused_without_a_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "docker")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = sandbox.run(HELLO, CSV, tmp_path / "c")
    assert not out.ok and "no docker binary" in out.stderr and out.files == ["code.py"]


@pytest.mark.skipif(not _daemon_up(), reason="no container runtime running")
def test_the_container_runs_the_script_with_the_csv_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZ_SANDBOX", "docker")
    out = sandbox.run(HELLO + "open(os.environ['VIZ_CSV'], 'a')\n", CSV, tmp_path / "d")
    assert not out.ok and "Read-only" in out.stderr  # the table cannot be written
    out = sandbox.run(HELLO, CSV, tmp_path / "e")
    assert out.ok and out.files == ["code.py", "facts.json", "figure.png"]
