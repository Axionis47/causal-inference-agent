"""Real Docker invariants shared by user exploration and family diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

from causal_agent.explore import sandbox

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
HELLO = "import os\nopen('facts.json', 'w').write('{}')\nopen('figure.png', 'wb').write(b'x')\nprint(os.environ['DATA_CSV'])\n"


def test_missing_docker_fails_visibly_without_host_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = sandbox.run(HELLO, CSV, tmp_path / "missing")
    assert not out.ok and "Docker is required" in out.stderr and out.files == ["code.py"]


def test_docker_probe_timeout_is_a_visible_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "docker")

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker info", 10)

    monkeypatch.setattr(subprocess, "run", fail)
    out = sandbox.run(HELLO, CSV, tmp_path / "unavailable")
    assert not out.ok and "Docker sandbox unavailable" in out.stderr


def test_container_reads_supplied_table_but_no_adjacent_host_data(tmp_path):
    projected = tmp_path / "projection.csv"
    original = tmp_path / "original.csv"
    projected.write_text("t,x\n0,1\n1,2\n")
    original.write_text("t,x,y\n0,1,100\n1,2,200\n")
    script = f"""import os
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
table = pd.read_csv(os.environ['DATA_CSV'])
assert list(table.columns) == ['t', 'x'] and len(table) == 2
model = LogisticRegression(solver='liblinear').fit(table[['x']], table['t'])
assert model.predict_proba(table[['x']]).shape == (2, 2)
try:
    Path({str(original)!r}).read_text()
except OSError:
    print('original read blocked')
else:
    raise AssertionError('the original outcome data were readable')
os.symlink({str(original)!r}, 'facts.json')
try:
    Path('facts.json').read_text()
except OSError:
    print('symlink read blocked')
else:
    raise AssertionError('a symlink exposed the original outcome data')
try:
    os.link({str(original)!r}, 'linked.csv')
except OSError:
    print('hardlink blocked')
else:
    raise AssertionError('a hardlink exposed the original outcome data')
try:
    Path(os.environ['DATA_CSV']).write_text('changed')
except OSError:
    print('input write blocked')
else:
    raise AssertionError('the input was writable')
table.describe().to_csv('diagnostics.csv')
print('pandas ran')
"""
    out = sandbox.run(script, projected, tmp_path / "step")
    assert out.ok, out.stderr
    assert "original read blocked" in out.stdout and "input write blocked" in out.stdout and "pandas ran" in out.stdout
    assert "symlink read blocked" in out.stdout and "hardlink blocked" in out.stdout
    assert "diagnostics.csv" in out.files and "facts.json" not in out.files and "linked.csv" not in out.files
    assert projected.read_text() == "t,x\n0,1\n1,2\n"
    assert original.read_text() == "t,x,y\n0,1,100\n1,2,200\n"


def test_the_container_runs_the_script_with_the_csv_read_only(tmp_path, monkeypatch):
    out = sandbox.run(HELLO + "open(os.environ['DATA_CSV'], 'a')\n", CSV, tmp_path / "n")
    assert not out.ok and "Read-only" in out.stderr  # the table cannot be written
    out = sandbox.run(HELLO, CSV, tmp_path / "o")
    assert out.ok and out.kind == "docker" and out.files == ["code.py", "facts.json", "figure.png"]


def test_what_the_script_printed_comes_back_cut_to_its_tail_and_its_tables_are_kept(tmp_path, monkeypatch):
    script = HELLO + (
        "open('table.csv', 'w').write('a,b\\n1,2\\n')\n"
        "open('notes.txt', 'w').write('kept')\n"
        "open('big.csv', 'w').write('x' * (sandbox_max + 1))\n"
        "open('model.pkl', 'wb').write(b'swept')\n"
        "import os; os.mkdir('scratch'); open('scratch/y', 'w').write('swept')\n"
        "print('tail ' + 'z' * 30000)\n"
    ).replace("sandbox_max", str(sandbox.MAX_FILE))
    out = sandbox.run(script, CSV, tmp_path / "p")
    assert out.ok and out.files == ["code.py", "facts.json", "figure.png", "notes.txt", "table.csv"]
    assert len(out.stdout) == sandbox.CAP and out.stdout.rstrip().endswith("z") and "tail" not in out.stdout  # the tail, not the head
    assert (tmp_path / "p" / "table.csv").read_text() == "a,b\n1,2\n"


def test_inputs_and_root_are_readonly_and_credentials_network_are_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_APPLICATION_KEY", "must-not-enter-container")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "saved.json").write_text('{"estimate": 2.5}')
    script = """import os, socket
from pathlib import Path
assert 'SECRET_APPLICATION_KEY' not in os.environ
assert os.getuid() != 0
assert Path('/inputs/saved.json').read_text() == '{"estimate": 2.5}'
for path in ['/inputs/saved.json', '/etc/probe']:
    try:
        Path(path).write_text('changed')
    except OSError:
        pass
    else:
        raise AssertionError('unexpected writable input/root')
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('network escaped')
print('boundary verified')
"""
    out = sandbox.run(script, CSV, tmp_path / "step", inputs=inputs)
    assert out.ok, out.stderr
    assert "boundary verified" in out.stdout
    assert (inputs / "saved.json").read_text() == '{"estimate": 2.5}'


def test_timeout_and_cancellation_remove_containers(tmp_path, monkeypatch):
    attempt = "test-cleanup-" + tmp_path.name
    monkeypatch.setenv("EXPLORE_ATTEMPT_ID", attempt)
    timeout = sandbox.run("import time\ntime.sleep(30)", CSV, tmp_path / "timeout", timeout=2)
    assert not timeout.ok and "timed out" in timeout.stderr
    cancel = threading.Event()
    timer = threading.Timer(2, cancel.set)
    timer.start()
    try:
        stopped = sandbox.run("import time\ntime.sleep(30)", CSV, tmp_path / "cancel", cancel=cancel)
    finally:
        timer.cancel()
    assert not stopped.ok and "cancelled" in stopped.stderr
    remaining = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=causal.explore.attempt={attempt}"], capture_output=True, text=True, check=True)
    assert not remaining.stdout.strip()
