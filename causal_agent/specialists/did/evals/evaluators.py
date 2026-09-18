"""Code evaluators for the diff-in-diff lane. One metric each. Work locally (RunTree) and uploaded (dict)."""

from __future__ import annotations


def _outs(run):
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def _exp(example):
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


def status_match(run, example):
    got, want = _outs(run).get("status"), _exp(example).get("status")
    return {"score": int(got == want), "comment": f"expected {want}, got {got}"}


def shape_match(run, example):
    want = _exp(example).get("shape")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("shape")
    pre_want = _exp(example).get("periods_pre")
    pre_got = _outs(run).get("periods_pre")
    ok = got == want and (pre_want is None or pre_got == pre_want)
    return {"score": int(ok), "comment": f"shape {got} (want {want}); pre periods {pre_got} (want {pre_want})"}


def estimator_allowed(run, example):
    allowed = _exp(example).get("estimator_in")
    if not allowed:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("estimator")
    return {"score": int(got in allowed), "comment": f"{got} in {allowed}"}


def sign_match(run, example):
    sign = _exp(example).get("effect_sign")
    if sign is None:
        return {"score": 1, "comment": "not graded"}
    v = _outs(run).get("effect")
    ok = v is not None and ((sign == "positive" and v > 0) or (sign == "negative" and v < 0))
    return {"score": int(ok), "comment": f"effect {v}, expected {sign}"}


def controls_ok(run, example):
    want = set(_exp(example).get("controls_exclude") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    included = set(_outs(run).get("controls") or [])
    kept_out = set(_outs(run).get("controls_excluded") or [])
    bad = want & included
    missing = want - kept_out - included  # neither excluded nor absorbed: the relate step never saw it or said nothing
    ok = not bad and not missing
    return {"score": int(ok), "comment": (f"should not be controls: {sorted(bad)}; " if bad else "") + (f"not judged at all: {sorted(missing)}" if missing else "kept out as expected")}


def flags_contain(run, example):
    want = set(_exp(example).get("flags_contain") or []) | set(_exp(example).get("hard_flags_contain") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    flags = set(_outs(run).get("flags") or [])
    hard = set(_outs(run).get("hard_flags") or [])
    missing = set(_exp(example).get("flags_contain") or []) - flags
    missing_hard = set(_exp(example).get("hard_flags_contain") or []) - hard
    ok = not missing and not missing_hard
    return {"score": int(ok), "comment": f"missing flags {sorted(missing)}; missing hard {sorted(missing_hard)}" if not ok else "flags as expected"}


def stopped_at(run, example):
    want = _exp(example).get("stopped_at")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("stage")
    return {"score": int(got == want), "comment": f"stage {got} (want {want})"}


def placebo_ran(run, example):
    want = set(_exp(example).get("placebo_ran") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    ran = set(_outs(run).get("placebos") or [])
    missing = want - ran
    return {"score": int(not missing), "comment": f"missing placebos {sorted(missing)}" if missing else "all ran"}


ALL = [status_match, shape_match, estimator_allowed, sign_match, controls_ok, flags_contain, stopped_at, placebo_ran]
