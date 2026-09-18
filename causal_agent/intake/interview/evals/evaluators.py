"""Code evaluators for the interview. One metric each. Work locally (RunTree) and uploaded (dict)."""

from __future__ import annotations

import re

METRIC_RE = re.compile(r"\bhow many\b|\bhow much of\b|\bwhat (share|percent|percentage|proportion|fraction|rate)\b", re.I)
FAMILY_WORDS = ["regression discontinuity", "difference-in-differences", "diff-in-diff", "instrumental variable", "propensity", "synthetic control",
                "interrupted time series", "diff_in_diff", "adjustment lane", "backdoor", "estimator"]


def _outs(run):
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def _exp(example):
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


def ready(run, example):
    want = _exp(example).get("ready")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("ready")
    return {"score": int(bool(got) == bool(want)), "comment": f"ready {got} (want {want})"}


def ready_within_turns(run, example):
    want = _exp(example).get("ready_within_turns")
    if want is None:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("ready_at_turn")
    ok = got is not None and got <= want
    return {"score": int(ok), "comment": f"ready at turn {got} (want <= {want})"}


def claims_match(run, example):
    want = _exp(example).get("claims") or {}
    if not want:
        return {"score": 1, "comment": "not graded"}
    got = _outs(run).get("claims") or {}
    bad = []
    for path, v in want.items():
        g = got.get(path)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            ok = g is not None and abs(float(g) - float(v)) < 1e-9
        else:
            ok = str(g).lower() == str(v).lower()
        if not ok:
            bad.append(f"{path}={g!r} (want {v!r})")
    return {"score": int(not bad), "comment": "; ".join(bad) if bad else "all match"}


def families(run, example):
    exp = _exp(example)
    want_s, want_x = set(exp.get("surviving_contains") or []), set(exp.get("struck_contains") or [])
    if not want_s and not want_x:
        return {"score": 1, "comment": "not graded"}
    o = _outs(run)
    s, x = set(o.get("surviving") or []), set(o.get("struck") or [])
    bad = sorted(want_s - s) + [f"{f} not struck" for f in sorted(want_x - x)]
    return {"score": int(not bad), "comment": "; ".join(bad) if bad else "families as expected"}


def no_metric_question(run, example):
    replies = _outs(run).get("replies") or []
    hits = [r[:80] for r in replies if METRIC_RE.search(r)]
    return {"score": int(not hits), "comment": f"asked for numbers: {hits}" if hits else "no metric questions"}


def no_family_word(run, example):
    replies = " ".join(_outs(run).get("replies") or []).lower()
    hits = [w for w in FAMILY_WORDS if w in replies]
    return {"score": int(not hits), "comment": f"named: {hits}" if hits else "no method words"}


def refutation_shown(run, example):
    exp = _exp(example)
    if not exp.get("refutation_shown"):
        return {"score": 1, "comment": "not graded"}
    o = _outs(run)
    refuted = set(o.get("refuted_at_turn0") or [])
    want = set(exp.get("refuted_at_turn0") or [])
    if want - refuted:
        return {"score": 0, "comment": f"not refuted at turn 0: {sorted(want - refuted)}"}
    first = (o.get("replies") or [""])[0]
    return {"score": int("%" in first or "rows" in first), "comment": "first reply shows the file's number" if "%" in first else "first reply shows no number"}


def question_per_open(run, example):
    misses = _outs(run).get("uncovered_open") or []
    return {"score": int(not misses), "comment": f"open claims without a question: {misses}" if misses else "every open claim asked"}


def no_fallback(run, example):
    fb = _outs(run).get("fallback_turns") or []
    return {"score": int(not fb), "comment": f"templated questions used at turns {fb}" if fb else "model replies passed the gate"}


def thoughts_present(run, example):
    thoughts = _outs(run).get("thoughts") or []
    nodes = {t.get("node") for t in thoughts if t.get("text")}
    return {"score": int(bool(nodes)), "comment": f"thoughts from {sorted(nodes)[:6]}" if nodes else "no thoughts logged"}


ALL = [ready, ready_within_turns, claims_match, families, no_metric_question, no_family_word, refutation_shown, question_per_open, no_fallback, thoughts_present]
