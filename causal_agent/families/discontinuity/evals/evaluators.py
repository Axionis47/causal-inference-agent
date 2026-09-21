"""The discontinuity lane's own evaluators. One metric each; the shared ones are in causal_agent.evals.evaluators."""

from __future__ import annotations

import re

from causal_agent.evals.evaluators import SHARED, exp, outs


def _key(name) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name or "").strip()).strip("_").lower()
    return s or "col"


def score_match(run, example):
    e = exp(example)
    if e.get("score_column") is None:
        return {"score": 1, "comment": "not graded"}
    o = outs(run)
    ok_col = _key(o.get("score_column")) == _key(e["score_column"])
    ok_cut = o.get("cutoff") is not None and abs(float(o["cutoff"]) - float(e.get("cutoff", 0))) < 1e-6
    ok_side = e.get("treated_side") is None or o.get("treated_side") == e["treated_side"]
    ok = ok_col and ok_cut and ok_side
    return {
        "score": int(ok),
        "comment": f"score {o.get('score_column')} cutoff {o.get('cutoff')} side {o.get('treated_side')} (want {e['score_column']} {e.get('cutoff')} {e.get('treated_side')})",
    }


def kind_match(run, example):
    want = exp(example).get("kind")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = outs(run).get("kind")
    return {"score": int(got == want), "comment": f"kind {got} (want {want})"}


def cluster_match(run, example):
    want = exp(example).get("cluster")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = outs(run).get("cluster")
    return {"score": int(_key(got) == _key(want)), "comment": f"cluster {got} (want {want})"}


def checks_ran(run, example):
    want = set(exp(example).get("checks_ran") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    ran = set(outs(run).get("checks_ran") or [])
    missing = want - ran
    return {"score": int(not missing), "comment": f"checks not run: {sorted(missing)}" if missing else "all ran"}


def assess_cites(run, example):
    want = exp(example).get("assess_cites_contain") or []
    if not want:
        return {"score": 1, "comment": "not graded"}
    cites = outs(run).get("assess_cites") or []
    if cites is None:
        return {"score": 0, "comment": "assess did not run"}
    missing = [w for w in want if not any(w in c for c in cites)]
    return {"score": int(not missing), "comment": f"assess did not cite {missing}" if missing else "cited"}


def secondaries_contain(run, example):
    want = set(exp(example).get("secondaries_contain") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    got = set(outs(run).get("secondaries") or [])
    missing = want - got
    return {"score": int(not missing), "comment": f"missing secondaries {sorted(missing)}" if missing else "all reported"}


def thoughts_present(run, example):
    thoughts = outs(run).get("thoughts") or []
    nodes = {t.get("node") for t in thoughts if t.get("text")}
    return {"score": int(bool(nodes)), "comment": f"thoughts from {sorted(nodes)}" if nodes else "no thoughts logged"}


ALL = [*SHARED, score_match, kind_match, cluster_match, checks_ran, assess_cites, secondaries_contain, thoughts_present]
