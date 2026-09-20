"""The core packages reach a family only through the registry: none of them names one, or the engine behind one."""

from __future__ import annotations

import re
from pathlib import Path

CORE = ("common", "profile", "memory", "viz", "lane")
WORDS = ("adjustment", "diff_in_diff", "discontinuity", "dowhy", "pyfixest", "rdrobust", "synthetic_control", "interrupted_series")
ROOT = Path(__file__).resolve().parents[2]


def test_no_core_module_names_a_family_or_an_engine():
    hits = []
    for pkg in CORE:
        for path in (ROOT / pkg).rglob("*.py"):
            if "tests" in path.parts:
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                for w in WORDS:
                    if re.search(rf"\b{w}\b", line, re.I):
                        hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not hits, "\n".join(hits)
