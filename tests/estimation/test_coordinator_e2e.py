# The estimation coordinator end to end (T-026 §2; PRD-004 §7, §9, §17, §22, §26). Every
# fixture reaches `prepared` through the real PRD-003 coordinator over the real stores, so the
# twelve §4 entry conditions are satisfied by honest artifacts and nothing is stubbed but the
# ONE §16.2 claim-review model call.

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psycopg
import pytest
from scipy import stats

from causal.estimation import contracts as ec
from causal.estimation import harness as eh
from causal.estimation import judge as ej
from causal.estimation import nodes
from causal.estimation import walls as ew
from causal.estimation.packs import load_estimation_packs
from causal.preparation import nodes as prep
from causal.runtime.failures import ADAPTERS
from causal.shared import events, frames, handoff, persistence
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.gateway import GatewayResultV1
from causal.shared.registry import load_artifact_type_registry
from tests.conftest import MIGRATIONS, requires_docker
from tests.preparation.test_coordinator_e2e import approved_design, event
from tests.preparation.test_coordinator_e2e import make_deps as preparation_deps

pytestmark = requires_docker

ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "registries"
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
METHOD, PACK_VERSION = "randomized_experiment", "randomized-experiment-pack.v1"
PACKS = load_estimation_packs(REGISTRIES / "method-pack-estimation.v1.json",
                              REGISTRIES / "method-packs.v1.json")
PACK = PACKS.get(METHOD, PACK_VERSION)
ARMS, THREE = ("control", "treated"), ("control", "treated", "boosted")
ONE_CONTRAST = ("treated_vs_control",)
ROLES = {"unit_id": "unit_identifier", "arm": "treatment", "y": "outcome",
         "x": "precision_covariate"}
CLUSTERED = ROLES | {"cluster": "cluster"}
BASE, CLUSTER, MULTI = "an-rct", "an-clu", "an-arms"
POISON, ATTRITION, THIN = "an-poison", "an-attr", "an-thin"
# One fixture per §9 shape the coordinator must carry end to end.
SPECS: dict[str, dict[str, Any]] = {
    BASE: {}, CLUSTER: {"cluster_size": 4},
    MULTI: {"units": 60, "arms": THREE,
            "contrasts": ("treated_vs_control", "boosted_vs_control")},
    POISON: {"contrasts": ("treated_vs_control", "absent_vs_control")},
    ATTRITION: {"attrition": {"treated": 4}}, THIN: {"units": 6}}
# D-089d: one honest fixture per registered pack, shaped like that adapter's own T-027/T-028
# table and prepared through the same real PRD-003 coordinator. The RCT pack is BASE above.
PACK_IDS: dict[str, str] = {"randomized_experiment": BASE, "aipw": "an-aipw", "did": "an-did",
                            "sharp_rdd": "an-rdd"}
PACK_COLUMNS: dict[str, tuple[str, ...]] = {
    "aipw": ("unit_id", "arm", "y", "x1"), "did": ("unit_id", "period", "grp", "adopt", "d", "y"),
    "sharp_rdd": ("unit_id", "score", "d", "y", "x")}
PACK_ROLES: dict[str, dict[str, str]] = {
    "aipw": {"unit_id": "unit_identifier", "arm": "treatment", "y": "outcome",
             "x1": "confounder_candidate"},
    "did": {"unit_id": "unit_identifier", "period": "time", "grp": "group",
            "adopt": "adoption_time", "d": "treatment", "y": "outcome"},
    "sharp_rdd": {"unit_id": "unit_identifier", "score": "running_variable", "d": "treatment",
                  "y": "outcome", "x": "precision_covariate"}}
PACK_CONTRASTS: dict[str, tuple[str, ...]] = {
    "aipw": ONE_CONTRAST, "did": ("adopters_vs_never_treated",),
    "sharp_rdd": ("above_vs_below_cutoff",)}
# The approved design facts each estimator cannot run without; the cutoff is deliberately not
# the adapter's own 0.0 fallback, so a plan that dropped it would estimate the wrong jump.
PACK_STRUCTURE: dict[str, dict[str, str]] = {
    "aipw": {}, "did": {"adoption_time": "4", "adoption_profile_id": "staggered"},
    "sharp_rdd": {"cutoff": "5", "assignment_direction": "above"}}
CUTOFF, JUMP = 5.0, 1.5


def trial(units: int = 40, arms: Sequence[str] = ARMS, *, cluster_size: int = 0,
          attrition: Mapping[str, int] = {}) -> bytes:
    """One frozen randomized table: assignment, outcome, baseline covariate, optional cluster.

    Every value is a deterministic function of the row number, so the parity numbers below are
    pinned by the fixture and not by a seed.
    """
    header = ["unit_id", "arm", "y", "x"] + (["cluster"] if cluster_size else [])
    rows = []
    for index in range(units):
        group = index // cluster_size if cluster_size else index // len(arms)
        arm = arms[(group if cluster_size else index) % len(arms)]
        baseline = 3.0 + float(index % 7)
        outcome = 10.0 + 4.0 * arms.index(arm) + 0.5 * baseline + float(index % 5) - 2.0
        dropped = arm in attrition and (index // len(arms)) % attrition[arm] == 0
        rows.append([f"u{index:03d}", arm, "" if dropped else f"{outcome:.3f}", f"{baseline:.1f}"]
                    + ([f"c{group}"] if cluster_size else []))
    return ("\n".join([",".join(header), *(",".join(row) for row in rows)]) + "\n").encode()


def pack_table(method: str) -> bytes:
    """One frozen table per pack: confounded AIPW, a staggered panel, a sharp cutoff at 5.0."""
    rng = np.random.default_rng(20260826)
    rows: list[list[str]] = []
    if method == "aipw":
        for unit in range(160):
            covariate = float(rng.normal())
            treated = bool(rng.random() < 1.0 / (1.0 + np.exp(-0.6 * covariate)))
            outcome = 1.0 + 0.5 * covariate + 2.0 * treated + float(rng.normal(scale=0.5))
            rows.append([f"u{unit:03d}", "treated" if treated else "control",
                         f"{outcome:.6f}", f"{covariate:.6f}"])
    elif method == "did":
        for unit in range(72):
            cohort, level = [4.0, 7.0, 0.0][unit % 3], float(rng.normal(0.0, 0.4))
            for period in range(1, 11):
                treated = bool(cohort) and period >= cohort
                outcome = level + 0.1 * period + (2.0 if treated else 0.0) + float(
                    rng.normal(0.0, 0.05))
                rows.append([f"u{unit:03d}", str(period), f"{cohort:.0f}", f"{cohort:.0f}",
                             str(int(treated)), f"{outcome:.6f}"])
    else:
        for unit in range(2000):
            score = float(rng.uniform(0.0, 10.0))
            outcome = 2.0 + 0.4 * score + JUMP * (score >= CUTOFF) + float(rng.normal(0.0, 0.4))
            rows.append([f"u{unit:04d}", f"{score:.6f}", str(int(score >= CUTOFF)),
                         f"{outcome:.6f}", f"{0.2 * score + float(rng.normal()):.6f}"])
    return ("\n".join([",".join(PACK_COLUMNS[method]),
                       *(",".join(row) for row in rows)]) + "\n").encode()


class Gateway:
    """The §16.2 claim-review receiver, answered from the bounded context it was handed.

    Every construction is counted, so "exactly one model call" is an assertion about this list
    and not about a mock's internals. The answer restates the frozen ceiling and the frozen
    numbers, which is all a compliant claim may do.
    """

    def __init__(self) -> None:
        self.calls: list[AgentTaskEnvelopeV1] = []

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        seen = dict(envelope.payload)
        cited = [str(seen["allowed_artifact_ids"][0])]
        items = [{
            "contrast_id": row["contrast_id"], "status": row["ceiling"], "ceiling": row["ceiling"],
            "estimate_units": row["estimate_units"] or seen["estimand_id"],
            "population_id": seen["population_id"], "timeframe_id": seen["timeframe_id"],
            "comparator_id": row["comparator_id"] or seen["population_id"],
            "cited_artifact_ids": cited, "estimate": row["estimate"],
            "interval_lower": row["interval_lower"], "interval_upper": row["interval_upper"],
            "confidence_level": row["confidence_level"],
            "effect_statement": "the approved contrast moved the approved outcome"}
            for row in seen["contrasts"]]
        payload = {
            "causal_question": seen["causal_question"], "estimand_id": seen["estimand_id"],
            "status": ec.most_restrictive(tuple(str(row["status"]) for row in items)),
            "overall_ceiling": seen["overall_ceiling"], "items": items,
            "qualifications": list(seen["required_qualifications"]), "cited_artifact_ids": cited}
        body = {"envelope_id": envelope.envelope_id, "schema_version": "agent-task-result.v1",
                "task_id": envelope.task_id, "status": "complete", "payload": payload,
                "artifact_type": ej.ARTIFACT_TYPE, "artifact_schema_version": ej.SCHEMA_VERSION,
                "parent_artifact_ids": [], "claims": [], "missing_requirements": [],
                "conflicts": [], "warnings": [], "evidence_ids": [], "tool_receipts": [],
                "output_hash": None, "validation_target": "v"}
        return GatewayResultV1(text=json.dumps(body), parsed=body, token_usage={}, attempts=1,
                               seed=1)


def estimation_deps(conn: Any, objects: Any, sink: io.StringIO,
                    gateway: Gateway) -> eh.EstimationDeps:
    """The production wiring over the docker stores, the frozen registries, and the RCT adapter."""
    registry = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
    emitter = events.EventEmitter(sink)
    products = persistence.ProductStore(conn)
    return eh.EstimationDeps(
        conn=conn, products=products, objects=objects, registry=registry, emitter=emitter,
        committer=persistence.ArtifactCommitter(objects, products, registry, emitter),
        clock=lambda: NOW, packs=PACKS, frames=frames.FrameStore(objects=objects),
        rules=ew.load_validation_rules(REGISTRIES / "estimation-validation-rules.v1.json"),
        registries=REGISTRIES, repo_root=ROOT, gateway=gateway,  # type: ignore[arg-type]
        adapters=ADAPTERS)  # D-089a: the runtime's own four-adapter factory map


def prepared_outcome(conn: Any, objects: Any, sink: io.StringIO, analysis_id: str,
                     spec: Mapping[str, Any]) -> str:
    """Drive the real PRD-003 coordinator to `prepared` and return its outcome artifact id."""
    deps = preparation_deps(conn, objects, sink)
    data = trial(**{key: value for key, value in spec.items() if key != "contrasts"})
    roles = CLUSTERED if spec.get("cluster_size") else ROLES
    design = approved_design(deps, analysis_id, METHOD, data, roles=roles,
                             contrasts=spec.get("contrasts", ONE_CONTRAST))
    run = prep.run_preparation(deps, analysis_id=analysis_id, design_outcome_artifact_id=design,
                               stage_run_id=f"pr:{analysis_id}:1")
    assert run.status == "prepared", run.error_code
    return str(run.outcome_artifact_id)


def pack_prepared(conn: Any, objects: Any, sink: io.StringIO, method: str) -> str:
    """The same real PRD-003 run for one non-RCT pack, over that pack's own frozen table."""
    deps = preparation_deps(conn, objects, sink)
    analysis_id = PACK_IDS[method]
    design = approved_design(deps, analysis_id, method, pack_table(method),
                             roles=PACK_ROLES[method], contrasts=PACK_CONTRASTS[method],
                             structure=PACK_STRUCTURE[method])
    run = prep.run_preparation(deps, analysis_id=analysis_id, design_outcome_artifact_id=design,
                               stage_run_id=f"pr:{analysis_id}:1")
    assert run.status == "prepared", (method, run.error_code, run.conflict_code)
    return str(run.outcome_artifact_id)


class Stack:
    """One database, one object store, and every fixture prepared and estimated once."""

    def __init__(self, conn: Any, objects: Any) -> None:
        self.conn, self.objects, self.sink = conn, objects, io.StringIO()
        self.prepared = {name: prepared_outcome(conn, objects, self.sink, name, spec)
                         for name, spec in SPECS.items()}
        self.prepared |= {PACK_IDS[method]: pack_prepared(conn, objects, self.sink, method)
                          for method in PACK_IDS if method != METHOD}
        self.runs = {name: self.estimate(name) for name in self.prepared}

    def estimate(self, analysis_id: str, revision: int = 1) -> tuple[Any, Gateway,
                                                                     eh.EstimationDeps]:
        gateway = Gateway()
        deps = estimation_deps(self.conn, self.objects, self.sink, gateway)
        found = nodes.run_estimation(
            deps, analysis_id=analysis_id, stage_run_id=f"es:{analysis_id}:{revision}",
            preparation_outcome_artifact_id=self.prepared[analysis_id],
            estimation_revision=revision)
        return found, gateway, deps

    def payload(self, deps: eh.EstimationDeps, artifact_id: str) -> dict[str, Any]:
        found = deps.products.load_envelope(artifact_id)
        return dict(json.loads(deps.objects.get(found.payload_locator)))

    def committed(self, analysis_id: str, kind: str) -> list[str]:
        return [str(row[0]) for row in self.conn.execute(
            "SELECT artifact_id FROM causal.artifacts WHERE analysis_id = %s"
            " AND artifact_type = %s ORDER BY artifact_id", (analysis_id, kind)).fetchall()]

    def view(self, deps: eh.EstimationDeps, analysis_id: str) -> pl.DataFrame:
        base = eh.HarnessBase(deps)
        bundle = base.payload(str(base.payload(
            self.prepared[analysis_id])["prepared_bundle"]["artifact_id"]))
        return base.frame(str(bundle["prepared_frame"]["artifact_id"]))


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


def sandwich(design: np.ndarray, outcome: np.ndarray,
             groups: np.ndarray | None = None) -> tuple[float, float]:
    """An independent least-squares fit and the robust covariance the plan asks pyfixest for.

    HC2 with the finite-sample factor for a unit-randomized fit, CRV1 over the approved cluster
    otherwise. Column one of `design` is the treated indicator.
    """
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ outcome
    residual = outcome - design @ beta
    rows, columns = design.shape
    if groups is None:
        leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
        scaled = residual / np.sqrt(1.0 - leverage)
        meat = design.T @ (design * (scaled**2)[:, None]) * (rows / (rows - columns))
    else:
        names = sorted(set(groups.tolist()))
        scores = np.array([design[groups == found].T @ residual[groups == found]
                           for found in names])
        meat = scores.T @ scores * (len(names) / (len(names) - 1)) * (
            (rows - 1) / (rows - columns))
    covariance = inverse @ meat @ inverse
    return float(beta[1]), float(np.sqrt(covariance[1, 1]))


def matrix(frame: pl.DataFrame, treated: str = "treated", comparator: str = "control", *,
           adjusted: bool = True) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    kept = frame.filter(pl.col("arm").is_in((treated, comparator))).drop_nulls("y")
    columns = [np.ones(kept.height), (kept["arm"] == treated).cast(pl.Float64).to_numpy()]
    if adjusted:
        columns.append(kept["x"].cast(pl.Float64).to_numpy())
    return np.column_stack(columns), kept["y"].cast(pl.Float64).to_numpy(), kept


def primary(stack: Stack, deps: eh.EstimationDeps, analysis_id: str) -> dict[str, Any]:
    found = stack.committed(analysis_id, "PrimaryAnalysisResult")
    return stack.payload(deps, found[0])


def evidence(stack: Stack, deps: eh.EstimationDeps, analysis_id: str,
             kind: str, key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in
            (stack.payload(deps, found) for found in stack.committed(analysis_id, kind))}


# -- the happy path (EV-P4-001, EV-P4-010) --------------------------------


def test_a_scripted_run_reaches_complete_and_opens_the_prd005_handoff(stack: Stack) -> None:
    """§22: a `complete` outcome, a committed bundle, and a T-006-clean five-id handoff."""
    run, gateway, deps = stack.runs[BASE]
    assert run.status == "complete", run.error_code
    assert run.overall_ceiling == "reportable" and run.handoff_id is not None
    # §16.2: the claim review is the only model call the whole stage makes.
    assert len(gateway.calls) == 1 and gateway.calls[0].task_kind == ej.TASK_KIND
    assert gateway.calls[0].allowed_tool_ids == ()
    bundle = stack.payload(deps, stack.committed(BASE, "EstimationBundle")[0])
    assert len(bundle["evidence_bundles"]) == 3 and bundle["row_set_hash"] == run.row_set_hash
    opened = nodes.open_presentation_handoff(deps, BASE, str(run.outcome_artifact_id),
                                             "sr:presentation")
    assert len(opened.entries) == 5
    gate = handoff.HandoffGate(deps.objects, deps.products, handoff.HandoffStore(stack.conn),
                               deps.registry, deps.emitter)
    result = gate.accept(opened, "presentation-coordinator", frozenset({"complete"}),
                         lambda verdict, codes: event(BASE, f"handoff.{verdict}"))
    assert result.accepted, result.error_codes
    assert stack.conn.execute("SELECT state FROM estimation.estimation_runs WHERE stage_run_id"
                              " = %s", (run.stage_run_id,)).fetchone() == ("completed",)


def test_every_required_diagnostic_sensitivity_and_figure_family_is_committed(
    stack: Stack
) -> None:
    """§14, §15, §17: every registered row reaches a visible terminal artifact."""
    _, _, deps = stack.runs[BASE]
    diagnostics = evidence(stack, deps, BASE, "DiagnosticResult", "diagnostic_id")
    branches = evidence(stack, deps, BASE, "SensitivityResult", "branch_id")
    figures = evidence(stack, deps, BASE, "FigureDataArtifact", "builder_id")
    assert set(diagnostics) == set(PACK.severities())
    assert {row["execution_status"] for row in diagnostics.values()} == {"computed"}
    assert set(branches) == {row.branch_id for row in PACK.sensitivity_branches}
    assert {row["execution_status"] for row in branches.values()} == {"computed"}
    assert set(figures) == set(PACK.figure_builder_ids)
    assert {row["disclosure_status"] for row in figures.values()} == {"reportable"}
    assert figures["primary_contrast_intervals"]["points"][0]["interval_lower"] is not None
    # PRD-005 §5 condition 8 (D-094): both axis roles carry an approved unit and label.
    assert all(set(row["units"]) == set(row["labels"]) == {"x", "y"} for row in figures.values())
    assert all(all(row["units"].values()) and all(row["labels"].values())
               for row in figures.values())


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


def test_one_unestimable_contrast_stops_the_run_and_keeps_the_computed_item(
    stack: Stack
) -> None:
    """§6.3: the primary result is atomic — but the contrast that did compute stays readable."""
    run, gateway, deps = stack.runs[POISON]
    assert (run.status, run.error_code) == ("not_estimable", "primary_result_incomplete")
    assert gateway.calls == [] and run.handoff_id is None
    result = primary(stack, deps, POISON)
    assert result["complete"] is False
    assert [row["contrast_id"] for row in result["primary_items"]] == ["treated_vs_control"]
    assert result["primary_items"][0]["estimate"] > 0
    outcome = stack.payload(deps, str(run.outcome_artifact_id))
    assert outcome["status"] == "not_estimable" and outcome["estimation_bundle"] is None
    with pytest.raises(persistence.PersistenceError):
        nodes.open_presentation_handoff(deps, POISON, str(run.outcome_artifact_id), "sr:next")


# -- §9.2 attrition and the qualification ceiling -------------------------


def test_attrition_is_visible_by_arm_and_never_leaves_the_denominator(stack: Stack) -> None:
    """§9.2: missing outcomes are reported by arm and the population denominator is unchanged."""
    run, _, deps = stack.runs[ATTRITION]
    assert run.status == "complete", run.error_code
    assert run.overall_ceiling == "reportable_with_qualifications"
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
    judgment = stack.payload(deps, stack.committed(ATTRITION, "ClaimJudgment")[0])
    assert judgment["status"] == "reportable_with_qualifications" and judgment["qualifications"]


# -- the invalidation ceiling (§16.1) -------------------------------------


def test_an_invalidating_diagnostic_caps_the_claim_and_blocks_the_handoff(
    stack: Stack
) -> None:
    """§14.2, §16.1: an invalidation guard caps the ceiling and no PRD-005 handoff opens."""
    run, _, deps = stack.runs[THIN]
    assert (run.status, run.overall_ceiling) == ("invalidated", "not_reportable")
    rows = evidence(stack, deps, THIN, "DiagnosticResult", "diagnostic_id")
    assert rows["covariance_cluster_adequacy"]["policy_result"] == "invalidating"
    assert rows["covariance_cluster_adequacy"]["severity"] == "invalidation_guard"
    ceiling = stack.payload(deps, stack.committed(THIN, "JudgmentCeiling")[0])
    assert ceiling["overall_ceiling"] == "not_reportable"
    assert stack.payload(deps, stack.committed(THIN, "ClaimJudgment")[0])["status"] == (
        "not_reportable")
    assert not stack.committed(THIN, "EstimationBundle")
    with pytest.raises(persistence.PersistenceError):
        nodes.open_presentation_handoff(deps, THIN, str(run.outcome_artifact_id), "sr:next")


# -- rerun replay (EV-P4-010) and the D-086 producer stamp ----------------


def test_a_rerun_replays_the_artifacts_and_lands_the_same_terminal(stack: Stack) -> None:
    """D-035: a new stage run recommits the deterministic artifacts as no-ops."""
    first = stack.runs[BASE][0]
    kinds = ("EstimationContextManifest", "EstimationPlan", "PrimaryAnalysisResult",
             "EstimationBundle", "ClaimJudgment")
    replayed = {kind: stack.committed(BASE, kind) for kind in kinds}
    second, gateway, _ = stack.estimate(BASE, revision=2)
    assert (second.status, second.row_set_hash) == (first.status, first.row_set_hash)
    assert second.stage_run_id != first.stage_run_id and len(gateway.calls) == 1
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
    assert (run.status, len(gateway.calls)) == ("complete", 1), run.error_code
    kinds = ("EstimationPlan", "PrimaryAnalysisResult", "EstimationBundle", "ClaimJudgment")
    replayed = {kind: stack.committed(analysis_id, kind) for kind in kinds}
    plan = stack.payload(deps, replayed["EstimationPlan"][0])
    assert plan["method_id"] == method and plan["estimator_parameters"] | PACK_STRUCTURE.get(
        method, {}) == plan["estimator_parameters"]
    opened = nodes.open_presentation_handoff(deps, analysis_id, str(run.outcome_artifact_id),
                                             "sr:presentation")
    store = handoff.HandoffStore(stack.conn)
    gate = handoff.HandoffGate(deps.objects, deps.products, store, deps.registry, deps.emitter)
    try:  # D-037: the RCT handoff is already recorded by the §22 test above, never re-recorded
        accepted = gate.accept(opened, "presentation-coordinator", frozenset({"complete"}),
                               lambda verdict, codes: event(analysis_id, f"handoff.{verdict}"))
        assert accepted.accepted, accepted.error_codes
    except persistence.PersistenceError:
        assert store.load(opened.handoff_id).receiver_validation_result == "accepted"
    second, again, _ = stack.estimate(analysis_id, revision=4)  # BASE spends 1..3 above
    assert (second.status, second.row_set_hash) == (run.status, run.row_set_hash)
    assert len(again.calls) == 1 and second.stage_run_id != run.stage_run_id
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
    bundle = stack.payload(deps, stack.committed(PACK_IDS["aipw"], "EstimationBundle")[0])
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
    from causal.estimation.plancompile import CAPACITY_EXCEEDED, DesignConflictDraftV1

    return DesignConflictDraftV1(
        conflict_code=CAPACITY_EXCEEDED, failed_rule_id="over_limit:panel:contrasts",
        affected_row_count=0, affected_unit_count=0, affected_dimension_counts={"contrasts": 9},
        evidence_artifact_ids=(), why_no_permitted_operation="the frozen cardinality no longer fits",
        material_design_fields=("primary_contrasts",), recommended_action="revise_design")
