"""The node plumbing every lane shares: the stream writer, the honest stop, the cards and cites, the case as the code weighed
it, the rejection text a judgement is re-prompted with, and the three factories for the nodes that differ only by what a
lane binds to them. A lane's own judgements stay in its package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from langgraph.config import get_stream_writer
from langgraph.types import Command

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import Cited, Feasibility, Handoff
from causal_agent.common.llm import structured
from causal_agent.lane import case as C
from causal_agent.lane.state import LaneState

MAX_RELATE_ATTEMPTS = 3
MAX_REVISIONS = 3
MAX_MODEL_RETRIES = 3
MAX_PICK_ATTEMPTS = 2


def writer() -> Callable[[Any], None]:
    try:
        return get_stream_writer()
    except Exception:  # outside a graph run
        return lambda _x: None


def question(state: LaneState) -> str:
    return state.get("question") or ""


def stop(stage: str, reason: str, facts: list[str], fix: str, extra: dict | None = None) -> Command:
    """The honest stop: a Feasibility record and the jump to the feasibility node."""
    f = Feasibility(stage=stage, reason=reason, facts=facts, what_would_fix=fix)
    return Command(goto="feasibility", update={"feasibility": f, **(extra or {})})


def card(h: Handoff, key: str) -> str:
    return h.brief_text(key)


def case_of(state: LaneState) -> C.Case:
    c = state.get("case")
    return c if isinstance(c, C.Case) else C.Case()


def cites(h: Handoff, *addresses: str) -> list[str]:
    """The claim addresses that resolve in the pack, else the change card."""
    ok = [a for a in addresses if h.resolve(a)]
    return ok or ["change:1.note"]


def rejected(errors: list[str] | None) -> str:
    """What a judgement is re-prompted with after the harness refused its answer."""
    return ("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else ""


def keys(state: LaneState) -> tuple[str | None, str, list[str]]:
    """The treatment (None when the change is the rule itself), the outcome, and every other loaded column."""
    h = state["handoff"]
    assert h is not None
    t = _key(h.treatment) if h.treatment else None
    y = _key(h.outcome)
    return t, y, [k for k in state.get("columns", {}) if k not in (t, y)]


def table(state: LaneState) -> pd.DataFrame:
    return pd.read_csv(state["table_path"])


def after_checks(state: LaneState) -> str:
    return "assess" if any(r.level != "pass" for r in state["checks"]) else "pick_estimator"


def feasibility(state: LaneState) -> dict:
    f = state["feasibility"]
    assert f is not None
    writer()({"feasibility": f.model_dump()})
    return {}


# ------------------------------------------------------------------ the nodes that differ only by what the lane binds


def make_case(load_beliefs: Callable[[], dict]) -> Callable[[LaneState], dict]:
    """The `case` node: the pack weighed by code against the lane's beliefs.yaml."""

    def case(state: LaneState) -> dict:
        h = state["handoff"]
        assert h is not None
        c = C.weigh(h, load_beliefs())
        writer()({"case": c.render()})
        return {"case": c}

    return case


SettledClaims = Callable[[Handoff, str, C.Case], tuple[dict[str, bool], dict[str, str]]]


def make_settled(settled_claims: SettledClaims) -> tuple[Callable[[Handoff, str, C.Case], str], Callable[[Any, Handoff, C.Case], Any]]:
    """`settled_text`, the block a relate prompt shows for what the pack settles; `apply_settled`, the model's relation with
    every settled claim overwritten and a cited reason for each."""

    def settled_text(h: Handoff, k: str, case: C.Case) -> str:
        claims, cites_ = settled_claims(h, k, case)
        if not claims:
            return ""
        return (
            "\nSETTLED BY THE PACK (copy these answers; cite the address)\n"
            + "\n".join(f"  {c} = {str(v).lower()} [{cites_[c]}]" for c, v in claims.items())
            + "\n"
        )

    def apply_settled(r: Any, h: Handoff, case: C.Case) -> Any:
        claims, cites_ = settled_claims(h, r.column, case)
        if not claims:
            return r
        out = r.model_copy(deep=True)
        for c, v in claims.items():
            setattr(out, c, v)
            if v and not any(cites_[c] in reason.cites for reason in out.reasons):
                out.reasons.append(Cited(reason=f"{r.column}: {c} settled by the pack", cites=[cites_[c]]))
        return out

    return settled_text, apply_settled


def make_relate(prompts: Any, relation_cls: type[Any]) -> Callable[[dict], dict]:
    """The `relate` node: one column's relation to the change and the outcome, as the lane's prompts ask it."""

    def relate(task: dict) -> dict:
        user = prompts.RELATE_USER.format(**task)
        parsed, th = structured(relation_cls, prompts.RELATE_SYSTEM, user, node=f"relate:{task['column']}")
        parsed.column = task["column"]
        return {"relations": [parsed], "debug": [th]}

    return relate
