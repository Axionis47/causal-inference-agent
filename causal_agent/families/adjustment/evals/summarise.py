"""The adjustment lane's result as the graded summary: what the evaluators read."""

from __future__ import annotations


def summarise(result: dict) -> dict:
    d = (result.get("designs") or {}).get(result.get("lead_design_id")) or {}
    est = [e for e in result.get("estimates") or [] if e.get("design_id") == result.get("lead_design_id") and not e.get("secondary") and e.get("error") is None]
    refs: dict[str, dict[str, bool | None]] = {}
    for r in result.get("falsifications") or []:
        refs.setdefault(r["design_id"] + "." + r["contrast"], {})[r["refuter"]] = r.get("passed")
    f = result.get("feasibility") or {}
    checks = (d.get("checks") or {}).get("results") or []
    return {
        "status": result.get("status"),
        "stage": f.get("stage"),
        "adjustment_set": (d.get("estimand") or {}).get("adjustment_set"),
        "excluded": [x["column"] for x in (d.get("graph") or {}).get("excluded") or []],
        "nodes": list((d.get("graph") or {}).get("nodes") or []),
        "estimator": (d.get("estimator") or {}).get("name"),
        "contrasts": len(d.get("contrasts") or []) or None,
        "effects": {e["contrast"]: e["value"] for e in est},
        "falsifications": refs,
        "endpoint_ids": sorted({x["endpoint_id"] for x in (result.get("designs") or {}).values()}),
        "terminal": result.get("terminal"),
        "designs": {k: {"endpoint_id": v["endpoint_id"], "target": v["target"]} for k, v in (result.get("designs") or {}).items()},
        "estimates": result.get("estimates") or [],
        "hard_flags": sorted({c["name"] for c in checks if c["level"] == "hard"}),
        "run_dir": result.get("run_dir"),
    }
