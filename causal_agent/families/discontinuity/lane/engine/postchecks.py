"""Frozen-design falsification and sensitivity. No decision returns to reasoning."""

from __future__ import annotations

from causal_agent.common.contracts import Falsification, Sensitivity
from causal_agent.families.discontinuity.lane.engine import fit as F
from causal_agent.families.discontinuity.lane.engine.shape import covcol
from causal_agent.families.discontinuity.lane.knowledge import load_checks


def refit(d, table, *, spec=None, h=None, mask=None, cutoff=0.0, sharp=False):
    d.assert_integrity()
    if d.inference == "randomisation":
        hl, hr = h or d.bandwidths.h
        return F.fit_local_random(
            table,
            -hl,
            hr,
            fuzzy=False if sharp else d.fuzzy,
            reps=int(d.spec.get("reps", 1000)),
            seed=int(d.spec.get("seed", 7)),
            alpha=float(d.spec.get("alpha", 0.05)),
            mask=mask,
        )
    return F.fit(
        spec or d.spec,
        table,
        fuzzy=False if sharp else d.fuzzy,
        cluster=bool(d.cluster),
        vce=d.vce,
        covs=[covcol(k) for k in d.covariates.adjusted] or None,
        h=h or d.bandwidths.h,
        b=d.bandwidths.b,
        mask=mask,
        c=cutoff,
    )


def point(label, fitted, **extra):
    return dict(
        label=label,
        value=fitted.value,
        lo=fitted.ci_low,
        hi=fitted.ci_high,
        n_l=fitted.n_h_left,
        n_r=fitted.n_h_right,
        h=fitted.h_left,
        h_right=fitted.h_right,
        error=fitted.error,
        confidence_set=fitted.confidence_set.model_dump() if fitted.confidence_set else None,
        **extra,
    )


def falsify(d, table, entry):
    d.assert_integrity()
    outcomes = []
    details = []
    for label, side in [("control", table.x < 0), ("treated", table.x >= 0)]:
        subset = table[side]
        cutoff = float(subset.x.median())
        if not subset.x.min() < cutoff < subset.x.max():
            details.append(f"{label}: no internal artificial cutoff")
            continue
        spec = {"p": 1, "kernel": "tri", "masspoints": "adjust", "level": 95}
        kw = {"h": d.bandwidths.h, "b": d.bandwidths.b} if d.bandwidths.rule == "support_points" else {"bwselect": d.bandwidths.selector}
        f = F.fit(spec, subset, c=cutoff, fuzzy=False, cluster=bool(d.cluster), vce=d.vce, **kw)
        floor = int(load_checks()["effective_rows"]["min"]["soft"])
        valid = not f.error and min(f.n_h_left, f.n_h_right) >= floor
        if valid:
            outcomes.append(bool(f.covers_zero()))
        details.append(
            f"{label} artificial cutoff {cutoff:g}: {f.value}; interval {f.ci_low}/{f.ci_high}; {f.n_h_left}/{f.n_h_right} rows; "
            + ("informative" if valid else f"uninformative ({f.error or 'effective rows below floor'})")
        )
    return Falsification(design_id=d.id, contrast=d.contrast.key, refuter=entry.name, passed=all(outcomes) if outcomes else None, detail="; ".join(details))


def sensitivity(d, table, entry):
    d.assert_integrity()
    points = []
    if entry.name == "polynomial_grid":
        for order in entry.params.get("orders", []):
            points.append(point(f"p={order}", refit(d, table, spec={**d.spec, "p": int(order), "q": int(order) + 1})))
    elif entry.name == "kernel_grid":
        for kernel in entry.params.get("kernels", []):
            points.append(point(kernel, refit(d, table, spec={**d.spec, "kernel": kernel})))
    elif entry.name == "bandwidth_grid":
        grid = {"h_mse": d.bandwidths.h, "2h_mse": [2 * h for h in d.bandwidths.h]}
        if d.bandwidths.h_cer_left is not None:
            grid["h_cer"] = [d.bandwidths.h_cer_left, d.bandwidths.h_cer_right]
            grid["2h_cer"] = [2 * h for h in grid["h_cer"]]
        for name in entry.grid:
            points.append(point(name, refit(d, table, h=grid[name]))) if name in grid else points.append(
                dict(label=name, error="selector unavailable under support-points rule")
            )
    elif entry.name == "donut":
        for share in entry.radii_share_of_h:
            radius = share * min(d.bandwidths.h)
            mask = table.x.abs() >= radius
            points.append(point(f"donut {share:g}", refit(d, table, mask=mask), dropped=int((~mask).sum()), changed_population=True)) if (
                ~mask
            ).any() else points.append(dict(label=f"donut {share:g}", error="no rows removed"))
    elif entry.name == "window_sensitivity":
        for share in entry.params.get("window_shares", []):
            widths = [share * h for h in d.bandwidths.h]
            points.append(point(f"window x{share:g}", refit(d, table, h=widths), changed_population=True, window=widths))
    elif entry.name == "inference_rows":
        for row in entry.params["rows"]:
            points.append(point(row, refit(d, table, spec={**d.spec, "inference_row": row})))
    elif entry.name == "variance_grid":
        for vce in entry.params["clustered" if d.cluster else "independent"]:
            fitted = F.fit(
                d.spec,
                table,
                fuzzy=d.fuzzy,
                covs=[covcol(k) for k in d.covariates.adjusted] or None,
                cluster=bool(d.cluster),
                vce=vce,
                h=d.bandwidths.h,
                b=d.bandwidths.b,
            )
            points.append(point(vce, fitted))
    elif entry.name == "rosenbaum_bounds":
        bound = F.rosenbaum_bounds(
            table, min(d.bandwidths.h), [float(g) for g in entry.params["gammas"]], reps=int(entry.params.get("reps", 500)), seed=int(d.spec.get("seed", 7))
        )
        if bound.get("error"):
            return Sensitivity(design_id=d.id, contrast=d.contrast.key, method=entry.name, error=bound["error"], detail="Rosenbaum bounds did not run")
        points = [
            dict(gamma=g, p_low=lo, p_high=hi, window=min(d.bandwidths.h), changed_population=d.bandwidths.two_sided())
            for g, lo, hi in zip(bound["gamma"], bound["lower"], bound["upper"], strict=True)
        ]
    else:
        return Sensitivity(design_id=d.id, contrast=d.contrast.key, method=entry.name, error="unknown declared sensitivity", detail="unsupported sensitivity")
    values = [p["value"] for p in points if p.get("value") is not None and not p.get("error")]
    return Sensitivity(
        design_id=d.id,
        contrast=d.contrast.key,
        method=entry.name,
        range_low=float(min(values)) if values else None,
        range_high=float(max(values)) if values else None,
        points=points,
        detail=entry.in_words,
        error="no sensitivity fit succeeded" if points and all(p.get("error") for p in points) else None,
    )
