#!/usr/bin/env python
"""Hold web/FLOW.md to the code it describes.

A drawn flow that no longer matches the app is worse than no drawing: the next
person reads a map of a building that has been rearranged. This checks the
three claims worth checking automatically.

    python test_flow.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FLOW = Path("web/FLOW.md")
APP = Path("web/src/App.tsx")
API = Path("causal/api.py")


def transitions(text: str) -> set[tuple[str, str]]:
    """Every `a --> b` inside the mermaid block."""
    return {
        (m.group(1), m.group(2))
        for m in re.finditer(r"^\s*(\w+)\s*-->\s*(\w+)", text, re.M)
        if m.group(1) != "[*]"
    }


def main() -> int:
    ok = True
    flow = FLOW.read_text()
    app = APP.read_text()
    api = API.read_text()
    edges = transitions(flow)

    # 1. every view the app can render is on the map
    views = set(re.findall(r'view === "(\w+)"', app))
    drawn = {a for a, _ in edges} | {b for _, b in edges}
    missing = views - drawn
    if missing:
        print(f"FAIL views in App.tsx but not in FLOW.md: {sorted(missing)}")
        ok = False
    else:
        print(f"PASS all {len(views)} rendered views appear on the map")

    # 2. no dead end: every state can be left by something other than a reset
    ends = {a for a, _ in edges}
    stuck = [s for s in drawn if s not in ends]
    if stuck:
        print(f"FAIL states with no way out: {stuck}")
        ok = False
    else:
        print(f"PASS every state has an exit")

    escape_only = [
        s for s in drawn
        if s in ends and {b for a, b in edges if a == s} == {"ask"}
    ]
    if escape_only:
        print(f"FAIL states whose only exit discards the job: {escape_only}")
        ok = False
    else:
        print("PASS no state can only be left by throwing the job away")

    # 3. endpoints the map names actually exist
    named = set(re.findall(r"POST /jobs/\{id\}/(\w+)|POST /(jobs)\b", flow))
    paths = {p for pair in named for p in pair if p}
    for path in sorted(paths):
        route = "/jobs" if path == "jobs" else f"/jobs/{{job_id}}/{path}"
        if f'"{route}"' not in api:
            print(f"FAIL FLOW.md names {route}, which api.py does not serve")
            ok = False
    else:
        print(f"PASS the {len(paths)} endpoints named on the map exist")

    print("FLOW GREEN" if ok else "FLOW DRIFTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
