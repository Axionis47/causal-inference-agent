"""A message that is nothing but edits: lines of `address = value`, one field each, the way the page's handoff view sends a
correction. Such a message is the person's word on those fields and never reaches the Reader; anything else in the message
means the Reader reads the whole of it."""

from __future__ import annotations

import re

LINE = re.compile(r"^\s*((?:claim|col):\w+\.\w+)\s*=\s*(.+?)\s*$")


def edits_in(message: str) -> list[tuple[str, str]] | None:
    """Every non-empty line as (address, value), or None when any line is not an edit."""
    out: list[tuple[str, str]] = []
    for line in message.splitlines():
        if not line.strip():
            continue
        m = LINE.match(line)
        if m is None:
            return None
        out.append((m.group(1), m.group(2)))
    return out or None
