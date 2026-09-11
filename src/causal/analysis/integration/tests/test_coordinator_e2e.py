# The estimation coordinator end to end (T-026 §2; PRD-004 §7, §9, §17, §22, §26). Every
# fixture reaches `prepared` through the real PRD-003 coordinator over the real stores, so the
# twelve §4 entry conditions are satisfied by honest artifacts and nothing is stubbed but the
# ONE §16.2 claim-review model call.

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from scipy import stats

from causal.analysis.integration import harness as eh
from causal.analysis.integration import nodes
from causal.analysis.integration.tests.support.coordinator import (
    ATTRITION,
    BASE,
    CLUSTER,
    CUTOFF,
    JUMP,
    MULTI,
    NOW,
    PACK,
    PACK_IDS,
    PACK_STRUCTURE,
    POISON,
    THIN,
    Stack,
    evidence,
    matrix,
    primary,
    sandwich,
)
from causal.shared import handoff, persistence
from tests.infrastructure import MIGRATIONS, requires_docker
from tests.preparation.support import event

pytestmark = requires_docker

# One fixture per §9 shape the coordinator must carry end to end.
# D-089d: one honest fixture per registered pack, shaped like that adapter's own T-027/T-028
# table and prepared through the same real PRD-003 coordinator. The RCT pack is BASE above.
# The approved design facts each estimator cannot run without; the cutoff is deliberately not
# the adapter's own 0.0 fallback, so a plan that dropped it would estimate the wrong jump.


@pytest.fixture(scope="module")
def stack(postgres_dsn: str, minio_s3: dict[str, Any]) -> Iterator[Stack]:
    admin = psycopg.connect(postgres_dsn, autocommit=True)
    name = f"test_{uuid.uuid4().hex[:10]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    admin.close()
    conn = psycopg.connect(postgres_dsn.rsplit("/", 1)[0] + f"/{name}", autocommit=True)
    persistence.apply_migrations(conn, MIGRATIONS)
    yield Stack(conn, persistence.ObjectStore(minio_s3["client"], minio_s3["bucket"]))
    conn.close()


# -- the happy path (EV-P4-001, EV-P4-010) --------------------------------


def test_a_scripted_run_reaches_complete_and_opens_the_prd005_handoff(stack: Stack) -> None:
    """§22: a `complete` outcome, a committed bundle, and a T-006-clean five-id handoff."""
    run, gateway, deps = stack.runs[BASE]
    assert run.status == "complete", run.error_code
    assert run.overall_ceiling is None and run.handoff_id is not None
    # §16.2: the claim review is the only model call the whole stage makes.
    assert gateway.calls == []
    bundle = stack.payload(deps, stack.committed(BASE, "NumericalBundle")[0])
    assert len(bundle["evidence_bundles"]) == 2 and bundle["row_set_hash"] == run.row_set_hash
    opened = nodes.open_post_analysis_handoff(deps, BASE, str(run.outcome_artifact_id),
                                             "sr:presentation")
    assert len(opened.entries) == 3
    gate = handoff.HandoffGate(deps.objects, deps.products, handoff.HandoffStore(stack.conn),
                               deps.registry, deps.emitter)
    result = gate.accept(opened, "post-analysis", frozenset({"complete"}),
                         lambda verdict, codes: event(BASE, f"handoff.{verdict}"))
    assert result.accepted, result.error_codes
    assert stack.conn.execute("SELECT state FROM estimation.estimation_runs WHERE stage_run_id"
                              " = %s", (run.stage_run_id,)).fetchone() == ("completed",)


def test_every_diagnostic_sensitivity_and_support_measurement_is_committed(stack: Stack) -> None:
    """Numerical evidence persists independently of claim or figure production."""
    _, _, deps = stack.runs[BASE]
    diagnostics = evidence(stack, deps, BASE, "DiagnosticResult", "diagnostic_id")
    branches = evidence(stack, deps, BASE, "SensitivityResult", "branch_id")
    assert set(diagnostics) == set(PACK.severities())
    assert {row["execution_status"] for row in diagnostics.values()} == {"computed"}
    assert set(branches) == {row.branch_id for row in PACK.sensitivity_branches}
    supporting = stack.payload(deps, stack.committed(BASE, "AnalysisSupportingData")[0])
    assert supporting["measurements"] and supporting["primary_result"]
    assert not stack.committed(BASE, "ClaimJudgment")
    assert not stack.committed(BASE, "JudgmentCeiling")
    assert not stack.committed(BASE, "FigureDataArtifact")


# -- pyfixest parity (EV-P4-002) ------------------------------------------


def test_the_committed_itt_matches_a_hand_computed_ancova_with_robust_errors(
    stack: Stack
) -> None:
    """§9.1: the approved ANCOVA and its HC2 interval, pinned against an independent fit."""
    _, _, deps = stack.runs[BASE]
    design, outcome, kept = matrix(stack.view(deps, BASE))
    estimate, error = sandwich(design, outcome)
    item = primary(stack, deps, BASE)["primary_items"][0]
    assert item["estimate"] == pytest.approx(estimate, abs=1e-9)
    assert item["standard_error"] == pytest.approx(error, abs=1e-9)
    half = float(stats.t.ppf(0.975, kept.height - design.shape[1])) * error
    assert item["interval_lower"] == pytest.approx(estimate - half, abs=1e-9)
    assert item["interval_upper"] == pytest.approx(estimate + half, abs=1e-9)
    assert item["confidence_level"] == PACK.confidence_level
    assert item["contributing_counts"] == {"row": kept.height, "unit": kept.height}


def test_the_unadjusted_branch_matches_a_hand_computed_difference_in_means(
    stack: Stack
) -> None:
    """§9.4: the prespecified unadjusted branch is exactly the difference in means."""
    _, _, deps = stack.runs[BASE]
    design, outcome, _ = matrix(stack.view(deps, BASE), adjusted=False)
    estimate, error = sandwich(design, outcome)
    branch = evidence(stack, deps, BASE, "SensitivityResult", "branch_id")["unadjusted_itt"]
    assert branch["result"]["estimate"] == pytest.approx(estimate, abs=1e-9)
    assert branch["result"]["standard_error"] == pytest.approx(error, abs=1e-9)
    assert branch["comparison_result"] == "stable"


def test_a_cluster_randomized_fixture_uses_the_approved_cluster_robust_covariance(
    stack: Stack
) -> None:
    """§9.1: the approved cluster covariance, pinned against a hand-computed CRV1 sandwich."""
    run, _, deps = stack.runs[CLUSTER]
    assert run.status == "complete", run.error_code
    frame = stack.view(deps, CLUSTER)
    design, outcome, kept = matrix(frame)
    estimate, error = sandwich(design, outcome, kept["cluster"].to_numpy())
    item = primary(stack, deps, CLUSTER)["primary_items"][0]
    assert item["estimate"] == pytest.approx(estimate, abs=1e-9)
    assert item["standard_error"] == pytest.approx(error, abs=1e-9)
    assert item["method_quantities"]["clusters"] == 10
    counts = evidence(stack, deps, CLUSTER, "DiagnosticResult", "diagnostic_id")
    assert counts["covariance_cluster_adequacy"]["values"]["clusters"] == 10
    assert counts["covariance_cluster_adequacy"]["policy_result"] == "acceptable"


# -- multi-arm, multiplicity, and atomicity (EV-P4-002, §6.3) -------------


def test_two_confirmatory_contrasts_produce_ordered_items_and_a_holm_result(
    stack: Stack
) -> None:
    """§9.1: one item per approved contrast in one result, with the mandatory Holm adjustment."""
    run, _, deps = stack.runs[MULTI]
    assert run.status == "complete", run.error_code
    result = primary(stack, deps, MULTI)
    assert result["contrast_order"] == ["treated_vs_control", "boosted_vs_control"]
    assert [row["contrast_id"] for row in result["primary_items"]] == result["contrast_order"]
    assert result["complete"] and result["multiplicity_result"] is not None
    adjusted = stack.payload(deps, result["multiplicity_result"]["artifact_id"])
    assert adjusted["policy_id"] == "holm_step_down.v1"
    rows = adjusted["adjusted_by_contrast"]
    assert set(rows) == set(result["contrast_order"])
    ordered = sorted(rows.values(), key=lambda row: row["rank"])
    assert ordered[0]["adjusted_p_value"] == pytest.approx(2 * ordered[0]["p_value"])
    assert ordered[1]["adjusted_p_value"] >= ordered[0]["adjusted_p_value"]


def test_an_approved_contrast_without_observed_support_stops_before_fitting(stack: Stack) -> None:
    """An impossible approved contrast cannot produce a misleading partial result."""
    run, gateway, deps = stack.runs[POISON]
    assert (run.status, run.error_code) == (
        "not_estimable", "approved_contrast_without_support")
    assert gateway.calls == [] and run.handoff_id is None
    assert stack.committed(POISON, "PrimaryAnalysisResult") == []
    outcome = stack.payload(deps, str(run.outcome_artifact_id))
    assert outcome["status"] == "not_estimable" and outcome["estimation_bundle"] is None
    handoff = nodes.open_post_analysis_handoff(deps, POISON, str(run.outcome_artifact_id), "sr:next")
    assert handoff.originating_outcome == "not_estimable"
    assert [deps.products.load_envelope(entry.artifact_id).artifact_type for entry in handoff.entries] == [
        "EstimationOutcome", "EstimationContextManifest", "EstimationPlan"]


# -- §9.2 attrition and the qualification ceiling -------------------------


def test_attrition_is_visible_by_arm_and_never_leaves_the_denominator(stack: Stack) -> None:
    """§9.2: missing outcomes are reported by arm and the population denominator is unchanged."""
    run, _, deps = stack.runs[ATTRITION]
    assert run.status == "complete", run.error_code
    assert run.overall_ceiling is None
    rows = evidence(stack, deps, ATTRITION, "DiagnosticResult", "diagnostic_id")
    attrition = rows["outcome_attrition_by_arm"]
    assert attrition["denominators"] == {"row": 40, "assigned": 40, "unit": 40}
    assert attrition["values"]["attrition_treated"] == pytest.approx(0.25)
    assert attrition["values"]["attrition_control"] == pytest.approx(0.0)
    assert attrition["values"]["differential_attrition"] == pytest.approx(0.25)
    assert attrition["policy_result"] == "warning"
    # The mask declares the non-contributing rows; it never deletes them.
    mask = stack.payload(deps, stack.committed(ATTRITION, "AnalysisContributionMask")[0])
    assert mask["included_counts"]["row"] == 35 and mask["noncontributing_counts"]["row"] == 5
    assert primary(stack, deps, ATTRITION)["primary_items"][0][
        "contributing_counts"]["row"] == 35
    assert not stack.committed(ATTRITION, "ClaimJudgment")


def test_invalidating_diagnostics_are_handed_off_without_an_upstream_claim(stack: Stack) -> None:
    run, gateway, deps = stack.runs[THIN]
    assert (run.status, run.overall_ceiling) == ("complete", None)
    rows = evidence(stack, deps, THIN, "DiagnosticResult", "diagnostic_id")
    assert rows["covariance_cluster_adequacy"]["policy_result"] == "invalidating"
    assert gateway.calls == []
    assert stack.committed(THIN, "NumericalBundle")
    assert not stack.committed(THIN, "ClaimJudgment")
    assert nodes.open_post_analysis_handoff(deps, THIN, str(run.outcome_artifact_id), "sr:next")


# -- rerun replay (EV-P4-010) and the D-086 producer stamp ----------------


def test_a_rerun_replays_the_artifacts_and_lands_the_same_terminal(stack: Stack) -> None:
    """D-035: a new stage run recommits the deterministic artifacts as no-ops."""
    first = stack.runs[BASE][0]
    kinds = ("EstimationContextManifest", "EstimationPlan", "PrimaryAnalysisResult",
             "NumericalBundle", "AnalysisSupportingData")
    replayed = {kind: stack.committed(BASE, kind) for kind in kinds}
    second, gateway, _ = stack.estimate(BASE, revision=2)
    assert (second.status, second.row_set_hash) == (first.status, first.row_set_hash)
    assert second.stage_run_id != first.stage_run_id and len(gateway.calls) == 0
    assert {kind: stack.committed(BASE, kind) for kind in kinds} == replayed
    assert stack.conn.execute(
        "SELECT count(*) FROM estimation.estimation_runs WHERE analysis_id = %s"
        " AND state = 'completed'", (BASE,)).fetchone() == (2,)


@pytest.mark.parametrize("method", sorted(PACK_IDS))
def test_every_registered_pack_completes_opens_prd005_and_replays(stack: Stack,
                                                                  method: str) -> None:
    """D-089d: all four packs run through the one coordinator over the real four-adapter map."""
    analysis_id = PACK_IDS[method]
    run, gateway, deps = stack.runs[analysis_id]
    assert (run.status, len(gateway.calls)) == ("complete", 0), (
        run.error_code, [json.loads(line).get("safe_dimensions")
                         for line in stack.sink.getvalue().splitlines()
                         if analysis_id in line and "entry_validation_failed" in line])
    kinds = ("EstimationPlan", "PrimaryAnalysisResult", "NumericalBundle", "AnalysisSupportingData")
    replayed = {kind: stack.committed(analysis_id, kind) for kind in kinds}
    plan = stack.payload(deps, replayed["EstimationPlan"][0])
    assert plan["method_id"] == method and plan["estimator_parameters"] | PACK_STRUCTURE.get(
        method, {}) == plan["estimator_parameters"]
    opened = nodes.open_post_analysis_handoff(deps, analysis_id, str(run.outcome_artifact_id),
                                             "sr:presentation")
    store = handoff.HandoffStore(stack.conn)
    gate = handoff.HandoffGate(deps.objects, deps.products, store, deps.registry, deps.emitter)
    try:  # D-037: the RCT handoff is already recorded by the §22 test above, never re-recorded
        accepted = gate.accept(opened, "post-analysis", frozenset({"complete"}),
                               lambda verdict, codes: event(analysis_id, f"handoff.{verdict}"))
        assert accepted.accepted, accepted.error_codes
    except persistence.PersistenceError:
        assert store.load(opened.handoff_id).receiver_validation_result == "accepted"
    second, again, _ = stack.estimate(analysis_id, revision=4)  # BASE spends 1..3 above
    assert (second.status, second.row_set_hash) == (run.status, run.row_set_hash)
    assert len(again.calls) == 0 and second.stage_run_id != run.stage_run_id
    assert {kind: stack.committed(analysis_id, kind) for kind in kinds} == replayed


def test_the_approved_rdd_cutoff_reaches_the_plan_and_the_estimate(stack: Stack) -> None:
    """D-089b: the design's cutoff — never the adapter's 0.0 fallback — decides the estimate."""
    run, _, deps = stack.runs[PACK_IDS["sharp_rdd"]]
    plan = stack.payload(deps, stack.committed(PACK_IDS["sharp_rdd"], "EstimationPlan")[0])
    assert plan["estimator_parameters"]["cutoff"] == "5"
    item = primary(stack, deps, PACK_IDS["sharp_rdd"])["primary_items"][0]
    assert item["method_quantities"]["cutoff"] == pytest.approx(CUTOFF)
    assert item["estimate"] == pytest.approx(JUMP, abs=0.5), run.error_code


def test_the_aipw_run_commits_the_cross_fit_assignment_and_its_objects(stack: Stack) -> None:
    """D-089c: §10.2 — the coordinator commits what the adapter dealt, restricted objects too."""
    _, _, deps = stack.runs[PACK_IDS["aipw"]]
    dealt = stack.committed(PACK_IDS["aipw"], "CrossFitAssignment")
    assert len(dealt) == 1
    body = stack.payload(deps, dealt[0])
    assert body["fold_count"] == len(body["counts_by_fold"]) and body["nuisance_profile_id"]
    # The row-to-fold mapping is a restricted object: it is stored, never carried in a payload.
    mapping = json.loads(deps.objects.get(body["mapping_object"]["object_locator"]))
    assert mapping["fold_count"] == body["fold_count"] and mapping["fold_by_row_hex"]
    bundle = stack.payload(deps, stack.committed(PACK_IDS["aipw"], "NumericalBundle")[0])
    assert [ref["artifact_id"] for ref in bundle["cross_fit_assignments"]] == dealt
    assert not stack.committed(BASE, "CrossFitAssignment")


def test_an_estimation_produced_design_conflict_is_stamped_estimation_harness(
    stack: Stack
) -> None:
    """D-086: the registry admits a second producer, and the envelope names the real one."""
    _, _, deps = stack.runs[BASE]
    row = deps.registry.lookup("DesignConflict")
    assert row.producer_component == "preparation-harness"
    assert row.also_produced_by == ("estimation-harness",)
    base = eh.HarnessBase(deps)
    state = eh.new_state(BASE, f"es:{BASE}:3", 3, {})
    base.open_run(state, f"pr:{BASE}:1")
    held = stack.committed(BASE, "EstimationContextManifest")[0]
    state["artifacts"]["EstimationContextManifest"] = held
    state["hashes"][held] = deps.products.load_envelope(held).content_hash
    out = base.conflict(state, ec_conflict())
    envelope = deps.products.load_envelope(out["artifacts"]["DesignConflict"])
    assert envelope.producer_component == "estimation-harness"
    assert envelope.artifact_type == "DesignConflict"
    # A component the row never names still cannot claim the artifact.
    body = stack.payload(deps, envelope.artifact_id)
    foreign = persistence.build_envelope(
        deps.registry, "DesignConflict", body, analysis_id=BASE, stage_run_id=state[
            "stage_run_id"], producer_version="0.1.0", parents=(), created_at_utc=NOW,
        producer_component="presentation-coordinator")
    with pytest.raises(persistence.PersistenceError):
        deps.committer.commit(foreign, body, base.event(state, "artifact.committed",
                                                        eh.EVAL_STAGE))


def ec_conflict() -> Any:
    from causal.analysis.integration.plancompile import DesignConflictDraftV1

    return DesignConflictDraftV1(
        conflict_code="method_structure_changed", failed_rule_id="approved_unit_count_changed",
        affected_row_count=0, affected_unit_count=0, affected_dimension_counts={"contrasts": 9},
        evidence_artifact_ids=(), why_no_permitted_operation="the observed unit structure differs from the approved request",
        material_design_fields=("primary_contrasts",), recommended_action="revise_design")
