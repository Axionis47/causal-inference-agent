"""DAG/spec-derived guards on EDA tool arguments.

A constraint inspects the model's requested args against the causal model and
either clamps them (drop columns the DAG does not sanction) or rejects the call
(an argument that contradicts the model). Verdicts are advisory until the checks
accept tuned arguments; today they shape the observation fed back to the model.
Nothing here raises; a bad call becomes feedback, never a stage failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import EDAInputs


@dataclass
class Verdict:
    rejected: bool = False
    reason: str = ""
    clamped_args: dict[str, Any] = field(default_factory=dict)


def _adjustment_columns(inputs: EDAInputs) -> set[str]:
    """Columns the causal model sanctions for adjustment: the DAG's backdoor
    set when a graph exists, else the spec's candidate confounders, intersected
    with what the frame actually carries."""
    cols = set(inputs.frame.columns)
    if inputs.dag is not None:
        return set(inputs.dag.adjustment_set()) & cols
    return set(inputs.spec.candidate_confounders) & cols


def clamp_covariates(args: dict[str, Any], inputs: EDAInputs) -> Verdict:
    """Drop any requested covariate the causal model does not sanction."""
    requested = args.get("covariates") or []
    if not requested:
        return Verdict(clamped_args=dict(args))
    allowed = _adjustment_columns(inputs)
    kept = [c for c in requested if c in allowed]
    dropped = [c for c in requested if c not in allowed]
    out = dict(args)
    out["covariates"] = kept
    reason = (
        f"dropped {dropped}: not in the causal model's adjustment set"
        if dropped else ""
    )
    return Verdict(rejected=False, reason=reason, clamped_args=out)


def reject_moderator_in_adjustment_set(args: dict[str, Any], inputs: EDAInputs) -> Verdict:
    """A moderator must stratify, not double as an adjustment covariate."""
    moderator = args.get("moderator")
    if moderator and moderator in _adjustment_columns(inputs):
        return Verdict(
            rejected=True,
            reason=f"{moderator} is an adjustment-set covariate, not a stratifier",
        )
    return Verdict(clamped_args=dict(args))
