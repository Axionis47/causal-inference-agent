"""Cross-stage integration fixtures. Uses shared Docker stores and the preparation coordinator; numerical tests must not import this module."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from causal.analysis.integration import RESOURCE_ROOT, nodes
from causal.analysis.integration import harness as eh
from causal.analysis.integration import walls as ew
from causal.analysis.integration.packs import load_estimation_packs
from causal.preparation import nodes as prep
from causal.runtime.failures import ADAPTERS
from causal.shared import events, frames, persistence
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.gateway import GatewayResultV1
from causal.shared.registry import load_artifact_type_registry
from tests.preparation.support import approved_design
from tests.preparation.support import make_deps as preparation_deps

ROOT = Path(__file__).resolve().parents[6]


REGISTRIES = ROOT / "registries"


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


METHOD, PACK_VERSION = "randomized_experiment", "randomized-experiment-pack.v1"


PACKS = load_estimation_packs(RESOURCE_ROOT / "method-pack-estimation.v1.json",
                              REGISTRIES / "method-packs.v1.json")


PACK = PACKS.get(METHOD, PACK_VERSION)


ARMS, THREE = ("control", "treated"), ("control", "treated", "boosted")


ONE_CONTRAST = ("treated_vs_control",)


ROLES = {"unit_id": "unit_identifier", "arm": "treatment", "y": "outcome",
         "x": "precision_covariate"}


CLUSTERED = ROLES | {"cluster": "cluster"}


BASE, CLUSTER, MULTI = "an-rct", "an-clu", "an-arms"


POISON, ATTRITION, THIN = "an-poison", "an-attr", "an-thin"


SPECS: dict[str, dict[str, Any]] = {
    BASE: {}, CLUSTER: {"cluster_size": 4},
    MULTI: {"units": 60, "arms": THREE,
            "contrasts": ("treated_vs_control", "boosted_vs_control")},
    POISON: {"contrasts": ("treated_vs_control", "absent_vs_control")},
    ATTRITION: {"attrition": {"treated": 4}}, THIN: {"units": 6}}


PACK_IDS: dict[str, str] = {"randomized_experiment": BASE, "aipw": "an-aipw", "did": "an-did",
                            "sharp_rdd": "an-rdd"}


PACK_COLUMNS: dict[str, tuple[str, ...]] = {
    "aipw": ("unit_id", "arm", "y", "x1"), "did": ("unit_id", "period", "grp", "adopt", "d", "y"),
    "sharp_rdd": ("unit_id", "score", "d", "y", "x")}


PACK_ROLES: dict[str, dict[str, str]] = {
    "aipw": {"unit_id": "unit_identifier", "arm": "treatment", "y": "outcome",
             "x1": "confounder_candidate"},
    "did": {"unit_id": "unit_identifier", "period": "time", "grp": "group",
            "d": "treatment", "y": "outcome"},
    "sharp_rdd": {"unit_id": "unit_identifier", "score": "running_variable", "d": "treatment",
                  "y": "outcome", "x": "precision_covariate"}}


PACK_CONTRASTS: dict[str, tuple[str, ...]] = {
    "aipw": ONE_CONTRAST, "did": ("adopters_vs_never_treated",),
    "sharp_rdd": ("above_vs_below_cutoff",)}


PACK_STRUCTURE: dict[str, dict[str, str]] = {
    "aipw": {}, "did": {"adoption_time": "4", "adoption_profile_id": "staggered"},
    "sharp_rdd": {"cutoff": "5", "assignment_direction": "above",
                  "treated_value": "1", "comparator_value": "0"}}


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
    """Fail if numerical execution attempts to call a reporting model."""

    def __init__(self) -> None:
        self.calls: list[AgentTaskEnvelopeV1] = []

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        raise AssertionError("numerical execution must not invoke an LLM")


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
        rules=ew.load_validation_rules(RESOURCE_ROOT / "estimation-validation-rules.v1.json"),
        registries=REGISTRIES, repo_root=ROOT,
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

