"""The one column key and the address grammar. Every package that names a column or cites an address goes through here.

A key is the column name with every run of non-alphanumerics folded to one underscore, lowered:
"test preparation course" -> test_preparation_course, "race/ethnicity" -> race_ethnicity.

Addresses: dataset[.note|.profile.<facet>], change:<n>[.note], col:<key>[.note|.when|.set_by|.affected|.profile.<facet>],
claim:<key>[.<field>|.check], probe:<family>.<name>, check:<contrast>.<name>, figure:<id>[.<series>.<i>].
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def key(name: str) -> str:
    s = _NON_ALNUM.sub("_", str(name).strip()).strip("_").lower()
    return s or "col"


def norm_address(address: str) -> str:
    """The column segment of a col: or claim:col: address compares as a key, the rest lowercased, so a cite written
    with the column's original spelling (col:Total_Emp.note) resolves to the same card as col:total_emp.note."""
    a = str(address).strip()
    if a.startswith("col:"):
        name, dot, rest = a[4:].partition(".")
        return "col:" + key(name) + (dot + rest.lower() if dot else "")
    if a.startswith("claim:col:"):
        name, dot, rest = a[10:].partition(".")
        return "claim:col:" + key(name) + (dot + rest.lower() if dot else "")
    return a
