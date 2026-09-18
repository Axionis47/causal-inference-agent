"""Code evaluators for the router. One metric each. Work locally (RunTree) and uploaded (dict)."""

from __future__ import annotations


def _outs(run):
    return run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}


def _exp(example):
    return example.outputs if hasattr(example, "outputs") else example.get("outputs", {}) or {}


def family_match(run, example):
    got, want = _outs(run).get("family"), _exp(example).get("family")
    return {"score": int(got == want), "comment": f"expected {want}, got {got}"}


def outcome_match(run, example):
    got, want = _outs(run).get("outcome"), _exp(example).get("outcome")
    return {"score": int(got == want), "comment": f"expected {want}, got {got}"}


def treatment_match(run, example):
    got, want = _outs(run).get("treatment"), _exp(example).get("treatment")
    return {"score": int(got == want), "comment": f"expected {want}, got {got}"}


def gate_passed(run, example):
    o = _outs(run)
    ok = not o.get("gate_errors") and o.get("family") is not None
    return {"score": int(ok), "comment": "; ".join(o.get("gate_errors") or []) or "clean"}


def alternatives_acknowledged(run, example):
    want = set(_exp(example).get("admissible_also") or [])
    got = set(_outs(run).get("admissible") or [])
    missing = want - got
    return {"score": int(not missing), "comment": f"missing from admissible: {sorted(missing)}" if missing else "all acknowledged"}


def rejections_correct(run, example):
    want = set(_exp(example).get("must_reject") or [])
    got = set(_outs(run).get("rejected") or [])
    wrong = want - got
    return {"score": int(not wrong), "comment": f"should have been rejected but were not: {sorted(wrong)}" if wrong else "all rejected"}


ALL = [family_match, outcome_match, treatment_match, gate_passed, alternatives_acknowledged, rejections_correct]
