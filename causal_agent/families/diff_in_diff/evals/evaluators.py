"""The diff-in-diff lane's own evaluators. One metric each; the shared ones are in causal_agent.evals.evaluators."""

from __future__ import annotations

from causal_agent.evals.evaluators import SHARED, exp, outs


def shape_match(run, example):
    want = exp(example).get("shape")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = outs(run).get("shape")
    pre_want = exp(example).get("periods_pre")
    pre_got = outs(run).get("periods_pre")
    ok = got == want and (pre_want is None or pre_got == pre_want)
    return {"score": int(ok), "comment": f"shape {got} (want {want}); pre periods {pre_got} (want {pre_want})"}


def controls_ok(run, example):
    want = set(exp(example).get("controls_exclude") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    included = set(outs(run).get("controls") or [])
    kept_out = set(outs(run).get("controls_excluded") or [])
    bad = want & included
    missing = want - kept_out - included  # neither excluded nor absorbed: the relate step never saw it or said nothing
    ok = not bad and not missing
    return {
        "score": int(ok),
        "comment": (f"should not be controls: {sorted(bad)}; " if bad else "")
        + (f"not judged at all: {sorted(missing)}" if missing else "kept out as expected"),
    }


ALL = [*SHARED, shape_match, controls_ok]
