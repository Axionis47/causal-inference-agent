"""The figures a run leaves behind, drawn from its artifacts alone. This is the desk's fallback for a run whose lane wrote
no figures.json: the estimate against its falsifications (every lane) and the effect by period when the run left dynamic
effects. Each family's own postviz draws more, from inside the lane, with addresses the run checked."""

from __future__ import annotations

from causal_agent.common.contracts import RunRecord
from causal_agent.viz.postviz import common
from causal_agent.viz.spec import FigureSpec


def _estimates(rec: RunRecord) -> list[dict]:
    return (rec.specialist_result or {}).get("estimates") or rec.artifacts.get("estimates") or []


def _refutations(rec: RunRecord) -> list[dict]:
    return (rec.specialist_result or {}).get("refutations") or rec.artifacts.get("refutations") or []


def effect_and_refutations(rec: RunRecord, prefix: str = "placebo") -> FigureSpec | None:
    return common.effect_and_refutations(_estimates(rec), _refutations(rec), prefix)


def dynamic_effects(rec: RunRecord) -> FigureSpec | None:
    dyn = (rec.specialist_result or {}).get("dynamic") or rec.artifacts.get("dynamic") or {}
    return common.event_study(dyn)


def figures(rec: RunRecord, prefix: str = "placebo") -> list[FigureSpec]:
    """Every post-run figure the artifacts allow, in the order the chat shows them. `prefix` is how the lane addresses its
    falsifications; the family says which."""
    return [f for f in (effect_and_refutations(rec, prefix), dynamic_effects(rec)) if f is not None]
