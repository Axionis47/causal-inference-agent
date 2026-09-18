"""Code evaluators for the desk. One metric each. Work locally (RunTree) and uploaded (dict)."""

from __future__ import annotations

# method words and the family each names; a word is invented only when its family is not in the run's record
METHOD_WORDS = {"regression discontinuity": "discontinuity", "difference-in-differences": "diff_in_diff", "instrumental variable": "instrument",
                "synthetic control": "synthetic_control", "interrupted time series": "interrupted_series", "propensity": "adjustment", "backdoor": "adjustment"}


def _outs(run):
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def _exp(example):
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


def runs_count(run, example):
    want = _exp(example).get("runs")
    got = _outs(run).get("runs")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    return {"score": int(got == want), "comment": f"{got} runs (want {want})"}


def family_match(run, example):
    want = _exp(example).get("family")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    fams = _outs(run).get("families") or []
    return {"score": int(bool(fams) and fams[0] == want), "comment": f"first run routed to {fams[:1]} (want {want})"}


def answers_grounded(run, example):
    fails = _outs(run).get("gate_failures") or []
    return {"score": int(not fails), "comment": f"turns that fell back: {fails}" if fails else "every after-run reply passed the gate"}


def kinds_contain(run, example):
    want = set(_exp(example).get("kinds_contain") or [])
    if not want:
        return {"score": 1, "comment": "not graded"}
    got = set(_outs(run).get("kinds") or [])
    missing = want - got
    return {"score": int(not missing), "comment": f"missing turn kinds {sorted(missing)}" if missing else "kinds as expected"}


def revision_reran(run, example):
    if not _exp(example).get("revision_reran"):
        return {"score": 1, "comment": "not graded"}
    o = _outs(run)
    ok = "revise" in (o.get("kinds") or []) and (o.get("runs") or 0) >= 2
    return {"score": int(ok), "comment": "revision led to a second run" if ok else "no second run after the revision"}


def then_and_now(run, example):
    if not _exp(example).get("then_and_now"):
        return {"score": 1, "comment": "not graded"}
    briefs = _outs(run).get("briefs") or []
    ok = len(briefs) >= 2 and "Then and now" in briefs[1]
    return {"score": int(ok), "comment": "second brief states then and now" if ok else "second brief lacks then and now"}


def no_method_invented(run, example):
    o = _outs(run)
    text = " ".join(o.get("replies") or []).lower()
    considered = set(o.get("families_considered") or [])
    hits = [w for w, fam in METHOD_WORDS.items() if w in text and fam not in considered]
    return {"score": int(not hits), "comment": f"named families the record never considered: {hits}" if hits else "every method named is in the record"}


def thoughts_present(run, example):
    thoughts = _outs(run).get("thoughts") or []
    nodes = {t.get("node") for t in thoughts if t.get("text")}
    return {"score": int(any(n.startswith("turn") for n in nodes)), "comment": f"thoughts from {sorted(nodes)[:8]}"}


ALL = [runs_count, family_match, answers_grounded, kinds_contain, revision_reran, then_and_now, no_method_invented, thoughts_present]
