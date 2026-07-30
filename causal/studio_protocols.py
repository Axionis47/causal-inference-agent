"""Eight explicit design protocols and their pre-estimation contract checks.

The protocol decides which questions and diagnostics apply.  The preparation
agent may recommend fields, but these deterministic checks decide whether the
design can be frozen.  No function in this module sees an estimated treatment
effect, so it cannot tune a design toward a favourable result.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .checks import run as run_check
from .prep import numeric_frame


PROTOCOL_VERSION = "causal-protocols/1.1.0"


@dataclass(frozen=True)
class Protocol:
    lane: str
    visual: str
    pre_checks: tuple[str, ...]
    post_checks: tuple[str, ...]
    review_questions: tuple[str, ...]


PROTOCOLS = {
    "observational": Protocol(
        "observational", "causal DAG",
        ("required_roles", "temporal_order", "adjustment_set", "balance", "overlap"),
        ("confounder_strength", "subsample_stability", "specification_spread", "balance", "overlap", "leave_one_out"),
        (
            "Why could each selected covariate influence both treatment assignment and the outcome?",
            "Were all selected covariates measured before treatment?",
        ),
    ),
    "matching": Protocol(
        "matching", "causal DAG plus overlap map",
        ("required_roles", "temporal_order", "adjustment_set", "balance", "overlap"),
        ("confounder_strength", "subsample_stability", "specification_spread", "balance", "overlap", "leave_one_out"),
        (
            "Why could each matching variable influence both programme participation and the outcome?",
            "Were matching variables measured before treatment, and is there a plausible counterpart for each treated unit?",
        ),
    ),
    "iv": Protocol(
        "iv", "instrument DAG",
        ("required_roles", "temporal_order", "first_stage", "exclusion_review"),
        ("subsample_stability", "specification_spread"),
        (
            "What mechanism makes the instrument change treatment?",
            "Why can the instrument affect the outcome only through treatment?",
            "Could the instrument share a cause with the outcome?",
        ),
    ),
    "survival": Protocol(
        "survival", "baseline DAG plus follow-up timeline",
        ("required_roles", "temporal_order", "survival_support", "censoring_review"),
        ("confounder_strength", "subsample_stability", "specification_spread", "proportional_hazards", "leave_one_out"),
        (
            "What starts follow-up, and were treatment and covariates defined at or before that time?",
            "What does censoring mean, and could censoring depend on treatment or prognosis?",
        ),
    ),
    "did": Protocol(
        "did", "group/intervention timeline",
        ("required_roles", "temporal_order", "panel_support", "parallel_trends_review"),
        ("subsample_stability", "pre_trend"),
        (
            "Why would treated and control groups have followed parallel trends without the intervention?",
            "Could another group-specific event or anticipation occur at the same time?",
        ),
    ),
    "rdd": Protocol(
        "rdd", "assignment/cutoff diagram",
        ("required_roles", "temporal_order", "cutoff_support", "manipulation_review"),
        ("subsample_stability", "placebo_cutoff"),
        (
            "Was treatment assigned by this cutoff, with no other rule changing there?",
            "Could units precisely manipulate which side of the cutoff they occupy?",
        ),
    ),
    "mediation": Protocol(
        "mediation", "treatment–mediator–outcome DAG",
        ("required_roles", "temporal_order", "mediation_order", "adjustment_set"),
        ("confounder_strength", "subsample_stability", "specification_spread", "leave_one_out"),
        (
            "Was the mediator measured after treatment but before the outcome?",
            "What could jointly cause treatment and mediator, or mediator and outcome?",
            "Could treatment create a mediator–outcome confounder?",
        ),
    ),
    "time_series": Protocol(
        "time_series", "intervention timeline",
        ("required_roles", "temporal_order", "time_support", "concurrent_events_review"),
        ("subsample_stability", "placebo_date"),
        (
            "What happened at the intervention date, and was that date chosen before inspecting the outcome series?",
            "What else changed at the same time, including seasonality, reporting, or population composition?",
        ),
    ),
}


ROLE_KEYS = {
    "observational": {"treatment": "treatment", "outcome": "outcome", "covariates": "confounder"},
    "matching": {"treatment": "treatment", "outcome": "outcome", "covariates": "confounder"},
    "iv": {"instrument": "instrument", "treatment": "treatment", "outcome": "outcome", "covariates": "baseline covariate"},
    "survival": {"treatment": "treatment", "duration": "duration", "event": "event", "covariates": "confounder"},
    "did": {"outcome": "outcome", "group": "group", "period": "period", "unit": "unit id"},
    "rdd": {"outcome": "outcome", "running": "running variable"},
    "mediation": {"treatment": "treatment", "mediator": "mediator", "outcome": "outcome", "covariates": "confounder"},
    "time_series": {"time": "time", "outcome": "outcome"},
}


def get_protocol(lane: str) -> Protocol:
    return PROTOCOLS[lane]


def role_ledger(lane: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate selected method fields into a reviewable column-role ledger."""
    rows: list[dict[str, Any]] = []
    for key, role in ROLE_KEYS[lane].items():
        value = kwargs.get(key)
        values = value if isinstance(value, (list, tuple)) else [value]
        for column in values:
            if column not in (None, ""):
                rows.append({
                    "column": str(column),
                    "role": role,
                    "source": "human-confirmed method configuration",
                    "status": "confirmed on contract freeze",
                })
    return rows


def build_contract(
    *,
    dataset_id: str,
    lane: str,
    kwargs: dict[str, Any],
    context: dict[str, Any],
    cohort: dict[str, Any] | None,
    data_version: dict[str, Any] | None = None,
    answers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the immutable content whose hash identifies one design revision."""
    protocol = get_protocol(lane)
    # Revision is a UI/history ordinal, not part of the content identity. The
    # version ID and fingerprints bind the contract to exact prepared bytes.
    data_binding = {
        key: value for key, value in (data_version or {}).items()
        if key != "revision"
    }
    return {
        "schema_version": "1.1.0",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "lane": lane,
        "question": context.get("question", ""),
        "estimand": {
            "observational": "ATE", "matching": "ATT", "iv": "LATE",
            "survival": "hazard ratio", "did": "DiD ATT", "rdd": "local jump",
            "mediation": "indirect effect", "time_series": "level and slope change",
        }[lane],
        "population": context.get("population", ""),
        "unit": context.get("unit", ""),
        "assignment": context.get("assignment", ""),
        "timing": context.get("timing", ""),
        "method_arguments": kwargs,
        "role_ledger": role_ledger(lane, kwargs),
        "cohort": cohort,
        "data_version": data_binding,
        "visual": protocol.visual,
        "assumption_answers": answers or {},
    }


def contract_hash(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _finding(check: str, verdict: str, detail: str, remediation: str = "") -> dict[str, str]:
    return {
        "phase": "pre",
        "check": check,
        "verdict": verdict,
        "detail": detail,
        "remediation": remediation,
    }


def _required_columns(df: pd.DataFrame, lane: str, kwargs: dict[str, Any]) -> dict[str, str]:
    ledger = role_ledger(lane, kwargs)
    missing = [item["column"] for item in ledger if item["column"] not in df.columns]
    if missing:
        return _finding("required_roles", "fail", f"Columns are missing: {', '.join(missing)}.", "Select columns present in the analysis-ready table.")
    return _finding("required_roles", "pass", f"{len(ledger)} selected role assignments name real columns.")


def _first_stage(df: pd.DataFrame, kwargs: dict[str, Any]) -> dict[str, str]:
    treatment, instrument = kwargs.get("treatment"), kwargs.get("instrument")
    covariates = list(kwargs.get("covariates") or [])
    try:
        data = numeric_frame(df, [treatment, instrument, *covariates], "iv preflight")
        controls = sm.add_constant(data[covariates], has_constant="add")
        fit = sm.OLS(data[treatment], controls.join(data[[instrument]])).fit()
        f_stat = float(fit.tvalues[instrument] ** 2)
        p_value = float(fit.pvalues[instrument])
    except Exception as exc:
        return _finding("first_stage", "fail", f"First-stage check failed: {type(exc).__name__}: {exc}", "Correct the IV roles or choose another design.")
    if p_value >= 0.05:
        return _finding("first_stage", "fail", f"Instrument does not measurably move treatment (F={f_stat:.2f}, p={p_value:.3g}).", "Do not run IV with this instrument.")
    verdict = "pass" if f_stat >= 10 else "review"
    return _finding("first_stage", verdict, f"First-stage F={f_stat:.2f}; values below 10 are weak.", "Use weak-IV-robust inference or a better justified instrument." if verdict == "review" else "")


def run_preflight(
    df: pd.DataFrame,
    lane: str,
    kwargs: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, str]]:
    """Run only checks that cannot depend on the estimated treatment effect."""
    findings = [_required_columns(df, lane, kwargs)]
    timing = str(context.get("timing", "")).strip()
    findings.append(_finding(
        "temporal_order",
        "pass" if timing and "unknown" not in timing.lower() else "review",
        f"Recorded timing: {timing or 'not supplied'}.",
        "Confirm treatment and every adjustment variable precede the outcome.",
    ))

    if lane in {"observational", "matching"}:
        covariates = list(kwargs.get("covariates") or [])
        findings.append(_finding(
            "adjustment_set", "pass" if covariates else "review",
            f"{len(covariates)} pre-treatment covariate(s) selected.",
            "Confirm a causal reason for the adjustment set; do not select controls by p-value.",
        ))
        for name in ("balance", "overlap"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = run_check(name, df, lane, kwargs, {}).__dict__
            if name == "balance" and result.get("verdict") == "fail":
                result["verdict"] = "review"
                result["detail"] += "; imbalance is expected before matching but must be corrected afterward"
            if caught:
                warning_types = sorted({item.category.__name__ for item in caught})
                result["verdict"] = "review" if result.get("verdict") != "fail" else "fail"
                result["detail"] += f"; numerical warnings captured: {', '.join(warning_types)}"
            result.update({"phase": "pre", "remediation": "Revise the population or design before estimating; do not trim toward a desired effect."})
            findings.append(result)
    elif lane == "iv":
        findings.append(_first_stage(df, kwargs))
        findings.append(_finding("exclusion_review", "review", "Exclusion and instrument independence cannot be established from this table.", "A causal reviewer must justify the instrument mechanism."))
    elif lane == "survival":
        try:
            event = pd.to_numeric(df[kwargs["event"]], errors="coerce")
            duration = pd.to_numeric(df[kwargs["duration"]], errors="coerce")
            valid_event = set(event.dropna().unique()) <= {0, 1}
            enough = int(event.eq(1).sum()) >= 10
            positive_time = bool(duration.dropna().ge(0).all())
            verdict = "pass" if valid_event and enough and positive_time else "fail"
            detail = f"{int(event.eq(1).sum())} events; binary event={valid_event}; non-negative follow-up={positive_time}."
        except Exception as exc:
            verdict, detail = "fail", f"Survival support failed: {exc}"
        findings.append(_finding("survival_support", verdict, detail, "Correct duration/event coding before estimation."))
        findings.append(_finding("censoring_review", "review", "Independent censoring is a domain assumption.", "Document why censoring is unrelated to unobserved prognosis after conditioning."))
    elif lane == "did":
        groups = int(df[kwargs["group"]].nunique(dropna=True))
        periods = int(df[kwargs["period"]].nunique(dropna=True))
        structural = groups == 2 and periods == 2
        findings.append(_finding("panel_support", "pass" if structural else "fail", f"Found {groups} groups and {periods} periods; this estimator requires exactly two of each.", "Select the correct group/period fields or use a multi-period estimator."))
        findings.append(_finding("parallel_trends_review", "review", "With two periods, parallel pre-trends cannot be tested.", "Provide external justification or use data with multiple pre-periods."))
    elif lane == "rdd":
        running = pd.to_numeric(df[kwargs["running"]], errors="coerce").dropna()
        cutoff = float(kwargs["cutoff"])
        left, right = int((running < cutoff).sum()), int((running >= cutoff).sum())
        in_range = bool(len(running) and running.min() < cutoff < running.max())
        verdict = "pass" if in_range and min(left, right) >= 20 else "fail"
        findings.append(_finding("cutoff_support", verdict, f"{left} observations below and {right} at/above cutoff {cutoff:g}.", "Use a genuine in-range policy cutoff with support on both sides."))
        findings.append(_finding("manipulation_review", "review", "Precise manipulation and other rules at the cutoff require domain evidence.", "Document the assignment mechanism before estimation."))
    elif lane == "mediation":
        distinct = len({kwargs.get("treatment"), kwargs.get("mediator"), kwargs.get("outcome")}) == 3
        findings.append(_finding("mediation_order", "review" if distinct else "fail", "Treatment, mediator, and outcome are distinct; their causal ordering still requires confirmation." if distinct else "Treatment, mediator, and outcome must be different columns.", "Confirm treatment → mediator → outcome timing and mediator-confounding assumptions."))
        covariates = list(kwargs.get("covariates") or [])
        findings.append(_finding("adjustment_set", "review", f"{len(covariates)} baseline covariate(s) selected; sequential ignorability is not testable.", "Review both treatment–mediator and mediator–outcome confounding."))
    elif lane == "time_series":
        stamps = pd.to_datetime(df[kwargs["time"]], errors="coerce", format="mixed")
        intervention = pd.to_datetime(kwargs["intervention"], errors="coerce")
        before, after = int((stamps < intervention).sum()), int((stamps >= intervention).sum())
        valid = not pd.isna(intervention) and before >= 15 and after >= 15
        findings.append(_finding("time_support", "pass" if valid else "fail", f"{before} observations before and {after} after the intervention.", "Provide a valid prespecified date with at least 15 observations on each side."))
        findings.append(_finding("concurrent_events_review", "review", "The series cannot show that nothing else changed at the intervention.", "Document seasonality, reporting changes, and concurrent events."))
    return findings


def design_dot(contract: dict[str, Any]) -> str:
    """Create a compact proposed causal/design map from the confirmed roles."""
    roles: dict[str, list[str]] = {}
    for item in contract.get("role_ledger", []):
        roles.setdefault(item["role"], []).append(item["column"])
    lane = contract["lane"]
    lines = ["digraph causal_design {", "rankdir=LR;", 'node [shape=box style="rounded,filled" fontname="Helvetica"];']
    colours = {
        "treatment": "#dbeafe", "outcome": "#dcfce7", "confounder": "#fef3c7",
        "instrument": "#ede9fe", "mediator": "#ffedd5", "running variable": "#f3e8ff",
        "time": "#e0f2fe", "group": "#fce7f3", "period": "#fce7f3",
        "duration": "#e0f2fe", "event": "#dcfce7", "unit id": "#f3f4f6",
        "baseline covariate": "#fef3c7",
    }
    for item in contract.get("role_ledger", []):
        node = hashlib.sha1(item["column"].encode()).hexdigest()[:10]
        lines.append(f'{node} [label="{item["column"]}\\n{item["role"]}" fillcolor="{colours.get(item["role"], "#f3f4f6")}"];')
    def node(role: str) -> str | None:
        values = roles.get(role, [])
        return hashlib.sha1(values[0].encode()).hexdigest()[:10] if values else None
    treatment, outcome = node("treatment"), node("outcome") or node("event")
    if treatment and outcome:
        lines.append(f"{treatment} -> {outcome};")
    for role in ("confounder", "baseline covariate"):
        for column in roles.get(role, []):
            source = hashlib.sha1(column.encode()).hexdigest()[:10]
            if treatment:
                lines.append(f"{source} -> {treatment};")
            if outcome:
                lines.append(f"{source} -> {outcome};")
    if lane == "iv" and node("instrument") and treatment:
        lines.append(f'{node("instrument")} -> {treatment};')
    if lane == "mediation" and node("mediator"):
        mediator = node("mediator")
        if treatment:
            lines.append(f"{treatment} -> {mediator};")
        if outcome:
            lines.append(f"{mediator} -> {outcome};")
    if lane == "did" and node("group") and node("period") and outcome:
        lines.append('policy_exposure [label="group × period\\npolicy exposure" fillcolor="#dbeafe"];')
        lines.append(f'{node("group")} -> policy_exposure;')
        lines.append(f'{node("period")} -> policy_exposure;')
        lines.append(f"policy_exposure -> {outcome};")
    if lane == "rdd" and node("running variable") and outcome:
        lines.append('cutoff_assignment [label="crosses cutoff\\nassignment" fillcolor="#dbeafe"];')
        lines.append(f'{node("running variable")} -> cutoff_assignment;')
        lines.append(f"cutoff_assignment -> {outcome};")
        lines.append(f'{node("running variable")} -> {outcome} [style=dashed label="smooth trend"];')
    if lane == "survival" and node("duration") and node("event"):
        lines.append(f'{node("duration")} -> {node("event")} [style=dashed label="follow-up clock"];')
    if lane == "time_series" and node("time") and outcome:
        lines.append('intervention [label="intervention date" fillcolor="#dbeafe"];')
        lines.append(f'{node("time")} -> intervention;')
        lines.append(f"intervention -> {outcome};")
        lines.append(f'{node("time")} -> {outcome} [style=dashed label="baseline trend"];')
    lines.append("}")
    return "\n".join(lines)


def protocol_as_dict(lane: str) -> dict[str, Any]:
    return asdict(get_protocol(lane)) | {"version": PROTOCOL_VERSION}
