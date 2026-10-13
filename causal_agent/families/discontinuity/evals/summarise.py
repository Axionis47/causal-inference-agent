"""The discontinuity lane's result as the graded summary: what the evaluators read."""

from __future__ import annotations


def summarise(result: dict) -> dict:
    d = (result.get("designs") or {}).get(result.get("lead_design_id")) or {}
    sc = d.get("score") or {}
    shape = d.get("shape") or {}
    checks = result.get("checks") or []
    est = [e for e in result.get("estimates") or [] if e.get("error") is None and e.get("design_id") == result.get("lead_design_id") and not e.get("secondary")]
    f = result.get("feasibility") or {}
    a = next((entry.get("record") for entry in reversed(result.get("reasoning_history") or []) if entry.get("node") == "line"), None)
    return {
        "status": result.get("status"),
        "terminal_kind": (result.get("terminal") or {}).get("kind"),
        "interpretation_errors": result.get("interpretation_errors") or {},
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
        "line_cites": [*a.get("cites", []), *[c for risk in a.get("risks", []) for c in risk.get("cites", [])], *[u["about"] for u in a.get("unsure", [])]]
        if a
        else None,
        "line_clean": a.get("clean") if a else None,
        "falsifications": sorted({r["refuter"] for r in result.get("falsifications") or []}),
        "falsifications_failed": sorted({r["refuter"] for r in result.get("falsifications") or [] if r.get("passed") is False}),
        "endpoints": sorted({d["endpoint_id"] for d in (result.get("designs") or {}).values()}),
        "sensitivities": sorted({s["method"] for s in result.get("sensitivities") or []}),
        "run_dir": result.get("run_dir"),
    }
