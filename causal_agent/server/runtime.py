"""Compose supervisor: persistent A2A agent and HTTP app, both trusted services."""

import os
import signal
import subprocess
import sys
import time

from causal_agent.common import trace


def main():
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for module, args in [("causal_agent.explore.server", ["--port", "8001"]), ("causal_agent.server", ["--host", "0.0.0.0", "--port", "8000"])]:
            children.append(subprocess.Popen([sys.executable, "-m", module, *args], env=trace.child_env(dict(os.environ))))
        while not stopping and all(child.poll() is None for child in children):
            time.sleep(0.2)
        if not stopping:
            print(f"Runtime child exited: {[child.poll() for child in children]}", file=sys.stderr, flush=True)
    finally:
        for child in children:
            child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    if not stopping:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
