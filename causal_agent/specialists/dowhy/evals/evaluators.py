"""Code evaluators for the adjustment lane. One metric each. Work locally (RunTree) and uploaded (dict)."""

from __future__ import annotations


def _outs(run):
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def _exp(example):
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


def status_match(run, example):
    got, want = _outs(run).get("status"), _exp(example).get("status")
    return {"score": int(got == want), "comment": f"expected {want}, got {got}"}


def adjustment_set_ok(run, example):
    e, o = _exp(example), _outs(run)
    got = set(o.get("adjustment_set") or [])
    if "adjustment_set_contains" in e:
        missing = set(e["adjustment_set_contains"]) - got
        return {"score": int(not missing), "comment": f"missing: {sorted(missing)}" if missing else "contains all"}
    if "adjustment_set_subset_of" in e:
        extra = got - set(e["adjustment_set_subset_of"])
        return {"score": int(not extra), "comment": f"unexpected: {sorted(extra)}" if extra else "within allowed"}
    return {"score": 1, "comment": "not graded for this case"}


def not_in_graph_ok(run, example):
    """Columns that must not be adjusted for: either excluded by the specialist or never handed over by the router."""
    want = set(_exp(example).get("not_in_graph") or [])
    nodes = set(_outs(run).get("nodes") or [])
    bad = want & nodes
    return {"score": int(not bad), "comment": f"in the graph: {sorted(bad)}" if bad else "none of them in the graph"}


def estimator_allowed(run, example):
    allowed = _exp(example).get("estimator_in")
    got = _outs(run).get("estimator")
    if not allowed:
        return {"score": 1, "comment": "not graded"}
    return {"score": int(got in allowed), "comment": f"{got} in {allowed}"}


def sign_match(run, example):
    want = _exp(example).get("effect_sign") or {}
    effects = _outs(run).get("effects") or {}
    bad = []
    for k, sign in want.items():
        v = effects.get(k)
        if v is None or (sign == "positive" and v <= 0) or (sign == "negative" and v >= 0):
            bad.append(f"{k}={v}")
    return {"score": int(not bad and (bool(want) or True)), "comment": "; ".join(bad) or "signs as expected"}


def placebo_passes(run, example):
    if not _exp(example).get("placebo_passes"):
        return {"score": 1, "comment": "not graded"}
    ref = _outs(run).get("refuters") or {}
    results = [v.get("placebo_treatment_refuter") for v in ref.values()]
    ok = bool(results) and all(results)
    return {"score": int(ok), "comment": f"placebo per contrast: {results}"}


def contrasts_count(run, example):
    want = _exp(example).get("contrasts")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("contrasts")
    return {"score": int(got == want), "comment": f"expected {want}, got {got}"}


def stopped_at(run, example):
    e, o = _exp(example), _outs(run)
    want = e.get("stopped_at")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = o.get("stage")
    hard = set(o.get("hard_flags") or [])
    need = set(e.get("hard_flags_contain") or [])
    ok = got == want and need <= hard
    return {"score": int(ok), "comment": f"stage {got} (want {want}); hard flags {sorted(hard)} (need {sorted(need)})"}


ALL = [status_match, adjustment_set_ok, not_in_graph_ok, estimator_allowed, sign_match, placebo_passes, contrasts_count, stopped_at]
