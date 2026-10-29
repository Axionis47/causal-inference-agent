"""Model-free acceptance of the actual Compose sandbox from the application container."""

import hashlib
import json
import tempfile
import threading
from pathlib import Path

from causal_agent.explore import sandbox

CODE = """import os, json, socket
from pathlib import Path
import numpy, pandas, scipy, statsmodels.api, sklearn, matplotlib.pyplot as plt
from PIL import Image
assert os.getuid() == 20000
assert not any(k in os.environ for k in ('DATABASE_URL','S3_SECRET_KEY','GOOGLE_APPLICATION_CREDENTIALS','LANGSMITH_API_KEY'))
for forbidden in ('/queue/health.json','/proc/1/environ','/app/.env','/opt/worker/isolation.py'):
    assert not Path(forbidden).exists()
source = Path(os.environ['DATA_CSV'])
for action in (lambda: source.write_text('changed'), lambda: source.chmod(0o666), lambda: source.unlink(), lambda: socket.socket(), lambda: os.setuid(0), lambda: os.chroot('/')):
    try: action()
    except (PermissionError, OSError): pass
    else: raise AssertionError('sandbox boundary did not enforce')
data = pandas.read_csv(source)
assert list(data.columns) == ['x','y','protected'] and len(data) == 3
data.plot(x='x', y='y'); plt.savefig('figure.png')
Path('facts.json').write_text(json.dumps({'rows':len(data)}))
"""


def main():
    sandbox.require_runtime()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        csv = root / "input.csv"
        csv.write_text("x,y,protected\n1,2,9\n2,4,8\n3,6,7\n")
        before = hashlib.sha256(csv.read_bytes()).hexdigest()
        output = root / "output"
        output.mkdir()
        result = sandbox.run(CODE, csv, output)
        assert result.ok, result
        assert json.loads((output / "facts.json").read_text()) == {"rows": 3}
        assert (output / "figure.png").stat().st_size > 100
        assert hashlib.sha256(csv.read_bytes()).hexdigest() == before
        timed = root / "timed"
        timed.mkdir()
        result = sandbox.run("while True: pass", csv, timed, timeout=0.3)
        assert not result.ok and "timed out" in result.stderr, result
        cancelled = root / "cancelled"
        cancelled.mkdir()
        event = threading.Event()
        timer = threading.Timer(0.3, event.set)
        timer.start()
        result = sandbox.run("while True: pass", csv, cancelled, cancel=event)
        timer.join()
        assert not result.ok and "cancelled" in result.stderr, result
        # The following job proves that cancellation cleanup releases the worker.
        clean = root / "clean"
        clean.mkdir()
        result = sandbox.run("from pathlib import Path; assert not Path('/task/work/figure.png').exists(); print('clean')", csv, clean)
        assert result.ok and "clean" in result.stdout, result
        print(json.dumps({"sandbox": "docker", "boundaries": "passed", "timeout": "passed", "cancellation": "passed", "fresh_task": "passed"}))


if __name__ == "__main__":
    main()
