"""Executable publication policy for causal-studio results.

Prompts may explain this policy, but this code is the authority.  It produces
one of three decisions: allow, review, or block.  The LangGraph workflow uses a
`review` decision to pause for a person; a `block` decision cannot publish.
"""
from __future__ import annotations

from typing import Any

from .studio_prep import context_readiness


POLICY_VERSION = "causal-studio/2026-07-29.3"


def _finding(rule: str, severity: str, message: str, owner: str) -> dict[str, str]:
    return {"rule": rule, "severity": severity, "message": message, "owner": owner}


def evaluate(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether a result can be presented without human review."""
    findings: list[dict[str, str]] = []
    context = state.get("context") or {}
    ready, missing = context_readiness(context)
    if not ready:
        findings.append(_finding(
            "context.required",
            "block",
            f"Required context is missing: {', '.join(missing)}.",
            "data owner",
        ))
    if not state.get("estimate"):
        findings.append(_finding(
            "result.exists", "block", "No estimate was produced.", "analysis owner"
        ))

    design_contract = state.get("design_contract") or {}
    design_approval = state.get("design_approval") or {}
    data_version = state.get("data_version") or {}
    if (
        not design_contract
        or not design_approval.get("approved")
        or not design_approval.get("role_ledger_and_map_confirmed")
    ):
        findings.append(_finding(
            "design.contract_frozen",
            "block",
            "The lane-specific design contract was not reviewed and frozen before estimation.",
            "causal reviewer",
        ))
    if (
        not data_version.get("version_id")
        or (design_contract.get("data_version") or {}).get("version_id")
        != data_version.get("version_id")
    ):
        findings.append(_finding(
            "design.data_version_mismatch",
            "block",
            "The frozen design contract is not bound to the active prepared-data version.",
            "data owner",
        ))
    if design_contract.get("revision_timing") == "post_estimation_exploratory":
        findings.append(_finding(
            "design.post_hoc_revision",
            "review",
            "This contract was revised after an earlier result was visible and must be labelled exploratory.",
            "causal reviewer",
        ))
    preflight_failures = [
        item for item in state.get("preflight", []) if item.get("verdict") == "fail"
    ]
    if preflight_failures:
        findings.append(_finding(
            "design.preflight_failed",
            "block",
            "Pre-estimation checks failed: " + ", ".join(item.get("check", "unknown") for item in preflight_failures) + ".",
            "analysis owner",
        ))
    preflight_reviews = [
        item for item in state.get("preflight", []) if item.get("verdict") in {"review", "warn", "untestable"}
    ]
    if preflight_reviews:
        findings.append(_finding(
            "design.preflight_assumptions",
            "review",
            "Pre-estimation assumptions remain review-bound: " + ", ".join(item.get("check", "unknown") for item in preflight_reviews) + ".",
            "causal reviewer",
        ))

    possible_pii = (state.get("data_quality") or {}).get("possible_pii_columns", [])
    if possible_pii:
        findings.append(_finding(
            "data.possible_pii",
            "review",
            f"Possible identifying columns are present: {', '.join(possible_pii[:8])}.",
            "data owner",
        ))

    if str(context.get("assignment", "")).lower() in {"unknown", "observational / unknown"}:
        findings.append(_finding(
            "design.assignment_unknown",
            "review",
            "Treatment assignment is unknown; causal language requires a specialist review.",
            "causal reviewer",
        ))

    if bool(context.get("high_impact")):
        findings.append(_finding(
            "use.high_impact",
            "review",
            "The result may affect people, money, eligibility, or access.",
            "accountable decision owner",
        ))

    lane = state.get("lane", "")
    if lane in {"iv", "mediation"}:
        findings.append(_finding(
            f"design.{lane}.untestable_core_assumption",
            "review",
            "This design depends on a core assumption the uploaded data cannot verify.",
            "causal reviewer",
        ))

    failed = [
        d for d in state.get("diagnostics", [])
        if d.get("verdict") == "fail"
    ]
    if failed:
        names = ", ".join(d.get("check", "unknown") for d in failed)
        findings.append(_finding(
            "diagnostics.failed",
            "review",
            f"Sensitivity/diagnostic failures require review: {names}.",
            "causal reviewer",
        ))

    severity = {f["severity"] for f in findings}
    decision = "block" if "block" in severity else "review" if "review" in severity else "allow"
    owners = list(dict.fromkeys(f["owner"] for f in findings))
    return {
        "version": POLICY_VERSION,
        "decision": decision,
        "findings": findings,
        "required_reviewers": owners,
        "rule": "The design sets the claim ceiling; diagnostics may only lower it.",
    }


def render_report(state: dict[str, Any]) -> str:
    """Build the publishable report from fixed facts, including policy."""
    estimate = state["estimate"]
    policy = state["policy"]
    context = state["context"]
    diagnostics = state.get("diagnostics", [])
    preflight = state.get("preflight", [])
    contract = state.get("design_contract", {})
    design_approval = state.get("design_approval", {})
    interval = "not available"
    if estimate.get("ci_low") is not None:
        interval = f"{estimate['ci_low']:.4g} to {estimate['ci_high']:.4g}"
    diag_lines = "\n".join(
        f"- **{d['check']} — {d['verdict']}**: {d['detail']}" for d in diagnostics
    ) or "- No applicable checks returned a result."
    finding_lines = "\n".join(
        f"- **{f['severity']} · {f['rule']}**: {f['message']}" for f in policy["findings"]
    ) or "- No policy exceptions were triggered."
    approval = state.get("approval") or {}
    approval_line = (
        f"Human review: **approved** — {approval.get('note', 'no note')}"
        if approval.get("approved")
        else "Human review: not required"
    )
    preflight_lines = "\n".join(
        f"- **{d['check']} — {d['verdict']}**: {d['detail']}" for d in preflight
    ) or "- No pre-estimation checks were recorded."
    return f"""# Causal analysis report

## Question

{context['question']}

**Dataset context.** {context['description']} One row represents **{context['unit']}**.
The target population is **{context['population']}**. Treatment assignment was recorded as
**{context['assignment']}**, and timing as **{context['timing']}**.

## Frozen design contract

- Contract revision: **{contract.get('revision', 'unknown')}**
- Contract hash: `{contract.get('contract_hash', 'missing')}`
- Protocol: `{contract.get('protocol_version', 'missing')}`
- Prepared-data version: `{(contract.get('data_version') or {}).get('version_id', 'missing')}`
- Prepared-data fingerprint: `{(contract.get('data_version') or {}).get('prepared_fingerprint', 'missing')}`
- Revision timing: **{contract.get('revision_timing', 'not recorded')}**
- Estimand declared before execution: **{contract.get('estimand', 'unknown')}**
- Design reviewer: **{design_approval.get('reviewer', 'not recorded')}**

### Pre-estimation checks

{preflight_lines}

## Result

**{state['headline']}**

- Design: `{state['lane']}`
- Estimand: `{estimate['estimand']}`
- Estimate: `{estimate['value']:.6g}`
- 95% interval: `{interval}`
- Rows used: `{estimate['n']}`
- Claim strength: **{state['strength']}**

This result is limited by the identifying assumption shown in the design review.
A sensitivity check can reveal fragility; it cannot prove that an assumption is true.

## Sensitivity and diagnostics

{diag_lines}

## Publication policy

- Policy version: `{policy['version']}`
- Decision: **{policy['decision']}**
- {approval_line}

{finding_lines}

## Intended use

{context['intended_use']}
"""
