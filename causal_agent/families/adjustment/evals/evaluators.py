"""The adjustment lane's own evaluators. One metric each; the shared ones are in causal_agent.evals.evaluators."""

from __future__ import annotations

from causal_agent.evals.evaluators import SHARED, exp, outs


def adjustment_set_ok(run, example):
    e, o = exp(example), outs(run)
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
    want = set(exp(example).get("not_in_graph") or [])
    nodes = set(outs(run).get("nodes") or [])
    bad = want & nodes
    return {"score": int(not bad), "comment": f"in the graph: {sorted(bad)}" if bad else "none of them in the graph"}


def placebo_passes(run, example):
    if not exp(example).get("placebo_passes"):
        return {"score": 1, "comment": "not graded"}
    ref = outs(run).get("refuters") or {}
    results = [v.get("placebo_treatment_refuter") for v in ref.values()]
    ok = bool(results) and all(results)
    return {"score": int(ok), "comment": f"placebo per contrast: {results}"}


def contrasts_count(run, example):
    want = exp(example).get("contrasts")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = outs(run).get("contrasts")
    return {"score": int(got == want), "comment": f"expected {want}, got {got}"}


ALL = [*SHARED, adjustment_set_ok, not_in_graph_ok, placebo_passes, contrasts_count]
