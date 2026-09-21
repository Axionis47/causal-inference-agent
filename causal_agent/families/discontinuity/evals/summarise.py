"""The discontinuity lane's result as the graded summary: what the evaluators read."""

from __future__ import annotations


def summarise(result: dict) -> dict:
    d = result.get("design") or {}
    sc = result.get("score") or {}
    shape = result.get("shape") or {}
    checks = result.get("checks") or []
    est = [e for e in result.get("estimates") or [] if e.get("error") is None and e.get("method") == d.get("estimator")]
    f = result.get("feasibility") or {}
    a = result.get("assessment")
    return {
        "status": result.get("status"),
        "stage": f.get("stage"),
        "score_column": sc.get("column"),
        "cutoff": sc.get("cutoff"),
        "treated_side": sc.get("treated_side") if sc.get("column") else None,
        "takeup_column": sc.get("takeup_column"),
        "kind": shape.get("kind"),
        "cluster": shape.get("cluster_column"),
        "estimator": d.get("estimator"),
        "estimand": d.get("estimand"),
        "effect": est[0]["value"] if est else None,
        "ci": [est[0]["ci_low"], est[0]["ci_high"]] if est else None,
        "checks_ran": sorted({c["name"] for c in checks}),
        "flags": sorted({c["name"] for c in checks if c["level"] != "pass"}),
        "hard_flags": sorted({c["name"] for c in checks if c["level"] == "hard"}),
        "assess_cites": a.get("cites") if a else None,
        "assess_action": a.get("action") if a else None,
        "placebos": sorted({r["refuter"] for r in result.get("refutations") or []}),
        "placebos_failed": sorted({r["refuter"] for r in result.get("refutations") or [] if r.get("passed") is False}),
        "secondaries": sorted({e["method"] for e in result.get("estimates") or [] if e.get("secondary") and e.get("error") is None}),
        "run_dir": result.get("run_dir"),
    }
