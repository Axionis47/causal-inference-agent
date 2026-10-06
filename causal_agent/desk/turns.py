"""What kind of turn a message is, decided first and cheaply. The decision model answers the questions `decider.yaml` spells
about the message and where the conversation stands; code turns the answers into a kind, or into nothing when the model is
not sure enough, and then the Reader decides for itself. Only an update reaches the Reader's writes; a confident explore, ask
or run skips the Reader altogether."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from causal_agent.common import llm

Kind = Literal["update", "explore", "ask", "run", "transform"]
KINDS: tuple[Kind, ...] = ("update", "explore", "ask", "run", "transform")
FILE = Path(__file__).with_name("decider.yaml")


@lru_cache(maxsize=1)
def spec() -> dict[str, Any]:
    return dict(yaml.safe_load(FILE.read_text()) or {})


def state_of(message: str, asked: str | None, at_ready: bool, last_reply: str) -> dict[str, Any]:
    """What the decision model is shown: the message, what the desk asked if anything, whether the desk has just shown the
    design and asked for the word, and the first line of its last reply."""
    return {
        "message": message,
        "the desk asked": asked or "nothing; the person spoke freely",
        "the desk has just shown the design it would run and asked for the word to run it": "yes" if at_ready else "no",
        "the desk's last reply began": (last_reply or "").strip().splitlines()[0][:200] if last_reply else "",
    }


def classify(
    message: str, *, asked: str | None, at_ready: bool, last_reply: str, node: str = "decide", threshold: float | None = None
) -> tuple[Kind | None, float]:
    """The kind of the turn and the confidence it rests on, or None when the decision model is not sure enough and the Reader
    should decide. A message that is about the data is an update whatever else it does; otherwise the kind the model chose.
    The threshold is decider.yaml's unless one is given (the eval sweeps it)."""
    cfg = spec()
    out = llm.decide(state_of(message, asked, at_ready, last_reply), cfg["questions"], node=node)
    threshold = float(cfg.get("threshold", 0.6)) if threshold is None else threshold
    kind, about = out.answers.get("kind"), out.answers.get("about_data")
    if about is not None and about.noul is not None and about.noul >= 0.5 and about.confidence >= threshold:
        return "update", about.confidence
    if kind is not None and kind.choice in KINDS and kind.confidence >= threshold:
        return kind.choice, kind.confidence  # type: ignore[return-value]
    return None, (kind.confidence if kind is not None else 0.0)
