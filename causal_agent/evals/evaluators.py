"""Evaluators every family shares. One metric each; a case that does not name the expectation is not graded. They work on a
local RunTree and on an uploaded dict alike."""

from __future__ import annotations


def outs(run) -> dict:
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def exp(example) -> dict:
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


def status_match(run, example):
    got, e = outs(run).get("status"), exp(example)
    want = e.get("status_in") or [e.get("status")]
    return {"score": int(got in want), "comment": f"expected one of {want}, got {got}"}


def estimator_allowed(run, example):
    allowed = exp(example).get("estimator_in")
    if not allowed:
        return {"score": 1, "comment": "not graded"}
    got = outs(run).get("estimator")
    return {"score": int(got in allowed), "comment": f"{got} in {allowed}"}


def _sign_ok(v, sign) -> bool:
    return v is not None and ((sign == "positive" and v > 0) or (sign == "negative" and v < 0))


def sign_match(run, example):
    """`effect_sign` as one word grades `effect`; as a mapping contrast -> word it grades `effects`; `effect_sign_if_done`
    grades only when the lane finished."""
    e, o = exp(example), outs(run)
    sign = e.get("effect_sign")
    if sign is None and e.get("effect_sign_if_done"):
        if o.get("status") != "done":
            return {"score": 1, "comment": "stopped; the sign is graded only when the lane finishes"}
        sign = e["effect_sign_if_done"]
    if sign is None:
        return {"score": 1, "comment": "not graded"}
    if isinstance(sign, dict):
        effects = o.get("effects") or {}
        bad = [f"{k}={effects.get(k)}" for k, s in sign.items() if not _sign_ok(effects.get(k), s)]
        return {"score": int(not bad), "comment": "; ".join(bad) or "signs as expected"}
    v = o.get("effect")
    return {"score": int(_sign_ok(v, sign)), "comment": f"effect {v}, expected {sign}"}


def flags_contain(run, example):
    e = exp(example)
    want, want_hard = set(e.get("flags_contain") or []), set(e.get("hard_flags_contain") or [])
    if not want and not want_hard:
        return {"score": 1, "comment": "not graded"}
    flags, hard = set(outs(run).get("flags") or []), set(outs(run).get("hard_flags") or [])
    missing, missing_hard = want - flags, want_hard - hard
    ok = not missing and not missing_hard
    return {"score": int(ok), "comment": f"missing flags {sorted(missing)}; missing hard {sorted(missing_hard)}" if not ok else "flags as expected"}


def stopped_at(run, example):
    e, o = exp(example), outs(run)
    want = e.get("stopped_at_in") or ([e["stopped_at"]] if e.get("stopped_at") else None)
    if not want:
        return {"score": 1, "comment": "not graded"}
    if o.get("status") == "done" and "done" in (e.get("status_in") or []):
        return {"score": 1, "comment": "finished; the stop is graded only when infeasible"}
    got = o.get("stage")
    hard, need = set(o.get("hard_flags") or []), set(e.get("hard_flags_contain") or [])
    ok = got in want and need <= hard
    return {"score": int(ok), "comment": f"stage {got} (want one of {want}); hard flags {sorted(hard)} (need {sorted(need)})"}


def placebo_ran(run, example):
    e = exp(example)
    want = set(e.get("placebo_ran") or [])
    if not want and e.get("placebo_ran_if_done") and outs(run).get("status") == "done":
        want = set(e["placebo_ran_if_done"])
    if not want:
        return {"score": 1, "comment": "not graded"}
    missing = want - set(outs(run).get("placebos") or [])
    return {"score": int(not missing), "comment": f"missing placebos {sorted(missing)}" if missing else "all ran"}


def thoughts_present(run, example):
    nodes = {t.get("node") for t in outs(run).get("thoughts") or [] if t.get("text")}
    return {"score": int(bool(nodes)), "comment": f"thoughts from {sorted(nodes)}" if nodes else "no thoughts logged"}


SHARED = [status_match, estimator_allowed, sign_match, flags_contain, stopped_at, placebo_ran]
