"""The diff-in-diff lane's result as the graded summary: what the evaluators read."""

from __future__ import annotations


def summarise(result: dict) -> dict:
    d = result.get("design") or {}
    shape = result.get("shape") or {}
    checks = result.get("checks") or []
    est = [e for e in result.get("estimates") or [] if e.get("error") is None and e.get("method") == d.get("estimator")]
    f = result.get("feasibility") or {}
    return {
        "status": result.get("status"),
        "stage": f.get("stage"),
        "shape": shape.get("kind"),
        "periods_pre": shape.get("periods_pre"),
        "estimator": d.get("estimator"),
        "controls": (result.get("controls") or {}).get("included"),
        "controls_excluded": sorted(
            {x["column"] for x in ((result.get("controls") or {}).get("excluded") or []) + ((result.get("controls") or {}).get("dropped_fixed") or [])}
        ),
        "effect": est[0]["value"] if est else None,
        "flags": sorted({c["name"] for c in checks if c["level"] != "pass"}),
        "hard_flags": sorted({c["name"] for c in checks if c["level"] == "hard"}),
        "placebos": sorted({r["refuter"] for r in result.get("refutations") or []}),
        "run_dir": result.get("run_dir"),
    }
