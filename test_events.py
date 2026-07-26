"""Pin the SSE event names the browser depends on.

Renaming one of these breaks the live tape silently: the API keeps returning
200 and the UI simply stops updating. This is the one thing in the project that
is a unit test rather than a benchmark, because the failure is invisible.

    python test_events.py
"""
import re
import sys
from pathlib import Path

from causal.api import EVENTS

PINNED = {"stage_started", "stage_done", "waiting_for_you", "completed", "failed"}


def main() -> int:
    ok = True
    if set(EVENTS) != PINNED:
        print(f"FAIL vocabulary changed: {sorted(set(EVENTS) ^ PINNED)}")
        ok = False
    else:
        print(f"PASS vocabulary is {sorted(PINNED)}")

    # every event name emitted anywhere must be one of the pinned five
    emitted = set()
    for path in (Path("causal/graph.py"), Path("causal/api.py")):
        text = path.read_text()
        emitted |= set(re.findall(r'"event":\s*"(\w+)"', text))
        emitted |= set(re.findall(r'_event\(state,\s*"(\w+)"', text))
    stray = emitted - PINNED
    if stray:
        print(f"FAIL emitted but not pinned: {sorted(stray)}")
        ok = False
    else:
        print(f"PASS all {len(emitted)} emitted names are pinned")

    print("EVENTS GREEN" if ok else "EVENTS DRIFTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
