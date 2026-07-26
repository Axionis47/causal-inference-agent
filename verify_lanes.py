#!/usr/bin/env python
"""Run every lane against its real dataset and check it against the literature.

    python verify_lanes.py

Exits 0 if every lane runs and every published benchmark is met.

A lane with no benchmark is checked only for a finite estimate on a real
sample, and says so in the output. Warnings are counted, never suppressed:
"runs cleanly" includes not muttering on the way through.
"""
from __future__ import annotations

import sys
import time
import warnings

from causal.estimate import Estimate, LaneError
from fixtures import cases, refusals

ROW = "{:<14} {:<14} {:>12} {:>12} {:>8} {:>6} {:>7} {:>6}  {}"


def check_case(case) -> tuple[list[str], bool]:
    lines, ok = [], True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.monotonic()
        try:
            est: Estimate = case.run(case.frame, **case.kwargs)
        except LaneError as exc:
            return [ROW.format(case.name, "-", "-", "-", "-", "-", "-", "-",
                               f"FAIL refused: {exc}")], False
        secs = time.monotonic() - start
    warns = len(caught)

    if not case.checks and not case.ranges:  # no published number: sanity only
        sane = est.value == est.value and abs(est.value) != float("inf") and est.n > 0
        ok = sane
        lines.append(ROW.format(
            case.name, est.estimand, "(none)", f"{est.value:.4g}", "-", "-",
            str(est.n), str(warns), "PASS sane" if sane else "FAIL not finite"))
        return lines, ok

    for label, low, high in case.ranges:
        hit = low <= est.value <= high
        ok = ok and hit
        lines.append(ROW.format(
            case.name, label, f"{low:g}..{high:g}", f"{est.value:.6g}", "-",
            "range", str(est.n), str(warns), "PASS" if hit else "FAIL"))

    for label, expected, band in case.checks:
        rel = abs(est.value - expected) / abs(expected)
        hit = rel <= band
        ok = ok and hit
        covers = (est.ci_low is not None and est.ci_low <= expected <= est.ci_high)
        lines.append(ROW.format(
            case.name, label, f"{expected:.6g}", f"{est.value:.6g}",
            f"{rel * 100:.2f}%", f"{band * 100:.0f}%", str(est.n), str(warns),
            ("PASS" if hit else "FAIL") + ("  ci covers truth" if covers else "")))
    return lines, ok


def main() -> int:
    print(ROW.format("lane", "estimand", "expected", "got", "off by",
                     "band", "n", "warns", "result"))
    print("-" * 104)
    all_ok = True
    for case in cases():
        lines, ok = check_case(case)
        all_ok = all_ok and ok
        for line in lines:
            print(line)

    print()
    print("refusals (a lane must decline when the data cannot support it)")
    print("-" * 104)
    for name, thunk, expected_text in refusals():
        try:
            thunk()
            print(f"  FAIL  {name}: it ran, but should have refused")
            all_ok = False
        except LaneError as exc:
            hit = expected_text in str(exc)
            all_ok = all_ok and hit
            print(f"  {'PASS' if hit else 'FAIL'}  {name}: {exc}")

    print()
    for case in cases():
        print(f"  {case.name:<14} {case.why}")

    print()
    print("ALL GREEN" if all_ok else "FAILURES ABOVE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
