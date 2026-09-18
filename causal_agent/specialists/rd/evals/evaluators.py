"""Code evaluators for the discontinuity lane. One metric each. Work locally (RunTree) and uploaded (dict)."""

from __future__ import annotations

import re


def _outs(run):
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def _exp(example):
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


def _key(name) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name or "").strip()).strip("_").lower()
    return s or "col"


def status_match(run, example):
    got = _outs(run).get("status")
    exp = _exp(example)
    want = exp.get("status_in") or [exp.get("status")]
    return {"score": int(got in want), "comment": f"expected one of {want}, got {got}"}


def score_match(run, example):
    exp = _exp(example)
    if exp.get("score_column") is None:
        return {"score": 1, "comment": "not graded"}
    o = _outs(run)
    ok_col = _key(o.get("score_column")) == _key(exp["score_column"])
    ok_cut = o.get("cutoff") is not None and abs(float(o["cutoff"]) - float(exp.get("cutoff", 0))) < 1e-6
    ok_side = exp.get("treated_side") is None or o.get("treated_side") == exp["treated_side"]
    ok = ok_col and ok_cut and ok_side
    return {"score": int(ok), "comment": f"score {o.get('score_column')} cutoff {o.get('cutoff')} side {o.get('treated_side')} (want {exp['score_column']} {exp.get('cutoff')} {exp.get('treated_side')})"}


def kind_match(run, example):
    want = _exp(example).get("kind")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("kind")
    return {"score": int(got == want), "comment": f"kind {got} (want {want})"}


def cluster_match(run, example):
    want = _exp(example).get("cluster")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("cluster")
    return {"score": int(_key(got) == _key(want)), "comment": f"cluster {got} (want {want})"}


def estimator_allowed(run, example):
    allowed = _exp(example).get("estimator_in")
    if not allowed:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("estimator")
    return {"score": int(got in allowed), "comment": f"{got} in {allowed}"}


def sign_match(run, example):
    exp = _exp(example)
    sign = exp.get("effect_sign")
    if sign is None and exp.get("effect_sign_if_done"):
        if _outs(run).get("status") != "done":
            return {"score": 1, "comment": "stopped; the sign is graded only when the lane finishes"}
        sign = exp["effect_sign_if_done"]
    if sign is None:
        return {"score": 1, "comment": "not graded"}
    v = _outs(run).get("effect")
    ok = v is not None and ((sign == "positive" and v > 0) or (sign == "negative" and v < 0))
    return {"score": int(ok), "comment": f"effect {v}, expected {sign}"}


def checks_ran(run, example):
    want = set(_exp(example).get("checks_ran") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    ran = set(_outs(run).get("checks_ran") or [])
    missing = want - ran
    return {"score": int(not missing), "comment": f"checks not run: {sorted(missing)}" if missing else "all ran"}


def flags_contain(run, example):
    exp = _exp(example)
    want = set(exp.get("flags_contain") or [])
    want_hard = set(exp.get("hard_flags_contain") or [])
    if not want and not want_hard:
        return {"score": 1, "comment": "not graded"}
    flags = set(_outs(run).get("flags") or [])
    hard = set(_outs(run).get("hard_flags") or [])
    missing, missing_hard = want - flags, want_hard - hard
    ok = not missing and not missing_hard
    return {"score": int(ok), "comment": f"missing flags {sorted(missing)}; missing hard {sorted(missing_hard)}" if not ok else "flags as expected"}


def assess_cites(run, example):
    want = _exp(example).get("assess_cites_contain") or []
    if not want:
        return {"score": 1, "comment": "not graded"}
    cites = _outs(run).get("assess_cites") or []
    if cites is None:
        return {"score": 0, "comment": "assess did not run"}
    missing = [w for w in want if not any(w in c for c in cites)]
    return {"score": int(not missing), "comment": f"assess did not cite {missing}" if missing else "cited"}


def stopped_at(run, example):
    exp = _exp(example)
    want = exp.get("stopped_at_in") or ([exp["stopped_at"]] if exp.get("stopped_at") else None)
    if not want:
        return {"score": 1, "comment": "not graded"}
    o = _outs(run)
    if o.get("status") == "done" and "done" in (exp.get("status_in") or []):
        return {"score": 1, "comment": "finished; the stop is graded only when infeasible"}
    got = o.get("stage")
    return {"score": int(got in want), "comment": f"stage {got} (want one of {want})"}


def placebo_ran(run, example):
    exp = _exp(example)
    want = set(exp.get("placebo_ran") or [])
    if not want and exp.get("placebo_ran_if_done") and _outs(run).get("status") == "done":
        want = set(exp["placebo_ran_if_done"])
    if not want:
        return {"score": 1, "comment": "not graded"}
    ran = set(_outs(run).get("placebos") or [])
    missing = want - ran
    return {"score": int(not missing), "comment": f"missing placebos {sorted(missing)}" if missing else "all ran"}


def secondaries_contain(run, example):
    want = set(_exp(example).get("secondaries_contain") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    got = set(_outs(run).get("secondaries") or [])
    missing = want - got
    return {"score": int(not missing), "comment": f"missing secondaries {sorted(missing)}" if missing else "all reported"}


def thoughts_present(run, example):
    thoughts = _outs(run).get("thoughts") or []
    nodes = {t.get("node") for t in thoughts if t.get("text")}
    return {"score": int(bool(nodes)), "comment": f"thoughts from {sorted(nodes)}" if nodes else "no thoughts logged"}


ALL = [status_match, score_match, kind_match, cluster_match, estimator_allowed, sign_match, checks_ran, flags_contain, assess_cites, stopped_at,
       placebo_ran, secondaries_contain, thoughts_present]
