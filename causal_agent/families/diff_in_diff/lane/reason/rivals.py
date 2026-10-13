"""Outcome-blind candidate configurations, targets, support and identities."""

from __future__ import annotations

from itertools import product

import pandas as pd

from causal_agent.common.contracts import Handoff
from causal_agent.families.diff_in_diff.design import DidDesign
from causal_agent.families.diff_in_diff.lane.contracts import Candidate
from causal_agent.families.diff_in_diff.lane.engine.diagnostics import regular_support, untreated_support
from causal_agent.families.diff_in_diff.lane.knowledge import ENDPOINTS, load_checks, load_estimators
from causal_agent.families.diff_in_diff.lane.reason.ledger import controls
from causal_agent.families.diff_in_diff.lane.shape import period_label
from causal_agent.lane.endpoints import MAX_DESIGNS, design_identity
from causal_agent.lane.ledger import Ledger


def eligible_control(h: Handoff, panel: pd.DataFrame, column: str) -> bool:
    card = h.column(column)
    if card is None or column not in panel or not pd.api.types.is_numeric_dtype(panel[column]):
        return False
    timing = card.provenance.get("when")
    moved = card.provenance.get("moved_by_change")
    safe = card.when == "before" and timing is not None and timing.status == "confirmed"
    safe |= card.when in {"before", "at"} and card.moved_by_change is False and moved is not None and moved.status == "confirmed"
    return bool(safe and panel.groupby("unit")[column].nunique().max() > 1 and panel.groupby("time_index")[column].nunique().max() > 1)


def control_sets(h: Handoff, panel: pd.DataFrame, ledger: Ledger) -> tuple[list[list[str]], list[str]]:
    fixed, doubtful = controls(ledger)
    legal = {c.key for c in h.columns if c.role == "candidate" and eligible_control(h, panel, c.key)}
    fixed = [c for c in fixed if c in legal]
    doubtful = [c for c in doubtful if c in legal]
    cap = load_checks()["rivals"]["max_control_toggles"]
    omitted = doubtful[cap:]
    doubtful = doubtful[:cap]
    return [sorted(fixed + [c for c, on in zip(doubtful, flags, strict=True) if on]) for flags in product([False, True], repeat=len(doubtful))], omitted


def support(panel: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    p = panel.copy()
    if candidate.support == "pre_last_adoption_prefix":
        last = int(p.cohort.max())
        p = p.loc[p.time_index < last].copy()
        # The last cohort is explicitly a comparison-only cohort within this scope.
        p.loc[p.cohort == last, ["cohort", "treated", "treat"]] = 0
        p.loc[p.cohort == 0, "rel_time"] = -1
    p = p.dropna(subset=["y", *candidate.controls])
    if candidate.endpoint_id == "paired_two_wave_att":
        p = p.loc[p.groupby("unit").time_index.transform("nunique") == 2]
    return p


def enumerate_candidates(h: Handoff, block: DidDesign, panel: pd.DataFrame, ledger: Ledger, context_id: str) -> tuple[list[Candidate], list[str]]:
    sets, omitted = control_sets(h, panel, ledger)
    aggregation = block.aggregation
    # Human cohort labels refer to the dataset calendar, never the engine's compressed index.
    labels = {
        label: int(row.time_index) for row in panel[["time", "time_index"]].drop_duplicates().itertuples() for label in {str(row.time), period_label(row.time)}
    }
    selected_cohorts = []
    for label in block.cohorts:
        if str(label) in labels:
            selected_cohorts.append(str(labels[str(label)]))
        else:
            raise ValueError(f"requested cohort {label!r} is not a dataset calendar value")
    shape = "wide" if block.before_column else "long"
    cohorts = int(panel.loc[panel.cohort > 0, "cohort"].nunique())
    never = bool((panel.cohort == 0).any())
    active = {e.endpoint_id for e in h.admission.endpoints if e.admitted} if h.admission else set()
    assumptions = [a for a, e in ledger.entries.items() if e.kind in {"parallel_trends", "selection", "spillover", "anticipation", "other_shock"}]
    required = {
        str(c) for e in ledger.entries.values() if e.kind == "parallel_trends" and isinstance(e.value, dict) for c in e.value.get("requires_controls", [])
    }
    out: list[Candidate] = []
    for endpoint in ENDPOINTS.endpoints:
        for covariates in sets:
            target = {
                "aggregation": aggregation,
                "population": "treated",
                "horizons": block.horizons,
                "cohorts": selected_cohorts,
                "outcome": h.outcome,
                "contrast": block.treated_group,
                "table_version": h.data_version,
            }
            cfg = {
                "controls": covariates,
                "target": target,
                "support": "pre_last_adoption_prefix"
                if block.prefix_scope_accepted and not never
                else endpoint.configuration.get("support", "admitted_window"),
            }
            cid = design_identity(endpoint.id, cfg, context_id)
            options = [e.name for e in load_estimators() if e.applies(endpoint=endpoint.id, aggregation=aggregation)]
            c = Candidate(
                id=cid, endpoint_id=endpoint.id, controls=covariates, target=target, estimator_options=options, support=cfg["support"], assumptions=assumptions
            )
            reason = ""
            if endpoint.id not in active:
                reason = "the desk has not admitted this endpoint"
            elif not options:
                reason = "endpoint does not support the requested target aggregation"
            elif not required <= set(covariates):
                reason = "the comparison account requires controls absent from this candidate"
            elif endpoint.id == "paired_two_wave_att" and (shape != "wide" or aggregation != "pooled" or covariates):
                reason = "paired endpoint requires two measurements; baseline controls are absorbed"
            elif endpoint.id == "common_adoption_projection_effect" and (shape != "long" or cohorts != 1 or not never):
                reason = "common adoption requires a single cohort and untreated comparison"
            elif endpoint.id == "untreated_imputation_att" and (shape != "long" or not never):
                reason = "full imputation endpoint requires never-treated support"
            elif endpoint.id == "untreated_prefix_imputation_att" and (shape != "long" or never or cohorts < 2 or not block.prefix_scope_accepted):
                reason = "prefix endpoint requires explicit accepted scope before last adoption"
            elif endpoint.id == "never_cohort_cell_att" and (shape != "long" or cohorts < 2 or not never or covariates):
                reason = "saturated endpoint requires multiple cohorts and never-treated units; covariates unsupported"
            elif endpoint.id == "clean_projection_att" and not never and not block.prefix_scope_accepted:
                reason = "no-never clean projection requires explicitly accepted prefix scope"
            elif endpoint.id == "clean_projection_att" and (shape != "long" or covariates or not regular_support(panel)):
                reason = "projection endpoint requires regular complete period blocks and no covariates"
            if not reason and c.endpoint_id in {"untreated_imputation_att", "untreated_prefix_imputation_att"} and not untreated_support(support(panel, c)):
                reason = "the untreated first-stage unit/time support is incomplete"
            if aggregation in {"pooled", "horizon", "horizon_profile"} and block.cohorts:
                reason = "cohort restriction requires a cohort or cohort_horizon target"
            if aggregation in {"cohort", "cohort_profile"} and block.horizons:
                reason = "horizon restriction within a cohort requires a cohort_horizon target"
            if aggregation in {"horizon", "cohort_horizon"} and len(block.horizons) != 1:
                reason = "a single requested horizon must be explicit"
            if aggregation in {"cohort", "cohort_horizon"} and len(block.cohorts) != 1:
                reason = "a single requested cohort must be explicit"
            if not reason and len([x for x in out if x.status == "proposed"]) >= MAX_DESIGNS:
                reason = "candidate budget exhausted; this alternative remains recorded but was not analysed"
            if reason:
                c = c.model_copy(update={"status": "excluded", "reason": reason})
            out.append(c)
    return out, omitted
