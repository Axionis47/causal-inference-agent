"""Numbers in prose, and whether each is grounded. The chat after a run and the exploration agent share one rule: a number
stated in words is within one percent of a number the material holds, or a rounding of one, or it is refused."""

from __future__ import annotations

import re
from collections.abc import Iterable

TOL = 0.01
NUM_RE = re.compile(r"(?<![\w:.\-])-?\d+(?:\.\d+)?(?![\w.:\-]*[a-zA-Z_])")


def numbers_in(text: str) -> list[float]:
    """The numbers worth checking in a text: anything with a decimal point, and any whole number of ten or more."""
    out = []
    for tok in NUM_RE.findall(text):
        try:
            v = float(tok)
        except ValueError:
            continue
        if abs(v) >= 10 or "." in tok:
            out.append(v)
    return out


def close(v: float, x: float) -> bool:
    """Within one percent of x, or a rounding of x to at most four places."""
    return abs(v - x) <= max(abs(x), 1e-9) * TOL or any(round(x, d) == v for d in range(5))


def in_line(v: float, line: str) -> bool:
    """The number appears in the line's own text, within the tolerance."""
    return any(close(v, x) for x in _all_numbers(line))


def grounded(v: float, numbers: Iterable[float]) -> bool:
    """The number is within the tolerance of one of the numbers given."""
    return any(close(v, x) for x in numbers)


def _all_numbers(text: str) -> list[float]:
    out = []
    for tok in NUM_RE.findall(text):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out
