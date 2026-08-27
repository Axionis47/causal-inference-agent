# The presentation coordinator end to end (T-032 §2; PRD-005 §5, §6, §15, §16, §17.2, §18, §20).
# The fixture reaches `complete` through the real PRD-003 and PRD-004 coordinators over the real
# stores, so the thirteen §5 entry conditions are satisfied by honest artifacts and nothing is
# stubbed but the ONE §9 curator call.

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from causal.cli import render as cli_render
from causal.presentation import contracts as pc
from causal.presentation import harness as ph
from causal.presentation import nodes
from causal.presentation.catalog import load_visualization_catalog
from causal.shared import events, persistence
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.gateway import GatewayResultV1
from causal.shared.registry import load_artifact_type_registry
from tests.conftest import MIGRATIONS, requires_docker
from tests.estimation.test_coordinator_e2e import (
    ATTRITION,
    BASE,
    NOW,
    REGISTRIES,
    ROOT,
    SPECS,
    Gateway,
    estimation_deps,
    prepared_outcome,
)
from tests.estimation.test_coordinator_e2e import nodes as est_nodes

pytestmark = requires_docker

CATALOG = load_visualization_catalog(REGISTRIES / "visualization-catalog.v1.json")
UNITS = ("count", "risk_difference", "level")


class Curator:
    """The §9 visualization curator, answered from the bounded context it was handed.

    Every construction is counted, so "exactly one model call" is an assertion about this list.
    The plan below chooses only registered templates, enumerated choices, and the harness's own
    figure ids; it invents no evidence and reads no value.
    """

    def __init__(self, *, inability: str | None = None, template: str | None = None) -> None:
        self.calls: list[AgentTaskEnvelopeV1] = []
        self.inability, self.template = inability, template

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        seen = dict(envelope.payload)
        figures = [self._figure(row, seen) for row in seen["evidence"]]
        payload: dict[str, Any] = {"figures": [], "inability_code": self.inability,
                                   "implicated_evidence_ids": [], "summary": ""}
        if self.inability is None:
            payload = {"figures": figures, "summary": "one figure per approved question",
                       "inability_code": None, "implicated_evidence_ids": []}
        body = {"envelope_id": envelope.envelope_id, "schema_version": "agent-task-result.v1",
                "task_id": envelope.task_id, "status": "complete", "payload": payload,
                "artifact_type": "FigurePlan", "artifact_schema_version": "figure-plan.v1",
                "parent_artifact_ids": [], "claims": [], "missing_requirements": [],
                "conflicts": [], "warnings": [], "evidence_ids": [], "tool_receipts": [],
                "output_hash": None, "validation_target": "v"}
        return GatewayResultV1(text=json.dumps(body), parsed=body, token_usage={}, attempts=1,
                               seed=1)

    def _figure(self, row: Any, seen: dict[str, Any]) -> dict[str, Any]:
        name = str(row["visual_evidence_id"])
        template = next(t for t in seen["templates"]
                        if t["template_id"] == (self.template or row["compatible_template_ids"][0]))
        choices = {key: value[0] for key, value in template["choices"].items()}
        qualifications = [str(item) for item in seen["qualification_ids"]]
        units = sorted({str(unit) for unit in row["units"].values()})
        described = (f"A {choices['marks']} figure of {name} in {' and '.join(units)} with its"
                     f" interval, the {choices['reference_lines']} reference, and"
                     f" {' '.join(qualifications) or 'no qualification'}.")
        return {"figure_id": f"fig_{name}", "template_id": template["template_id"],
                "visual_evidence_ids": [name], "panel_groups": [[name]], "choices": choices,
                "annotation_ids": [], "qualification_ids": qualifications,
                "text": {"title": f"What does {name} show?",
                         "caption": f"Frozen {name} values. {' '.join(qualifications)}".strip(),
                         "accessible_description": described}}


def presentation_deps(conn: Any, objects: Any, sink: io.StringIO, gateway: Any,
                      root: Path) -> ph.PresentationDeps:
    """The production wiring over the docker stores, the frozen catalog, and one fake curator."""
    registry = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
    emitter = events.EventEmitter(sink)
    products = persistence.ProductStore(conn)
    return ph.PresentationDeps(
        conn=conn, products=products, objects=objects, registry=registry, emitter=emitter,
        committer=persistence.ArtifactCommitter(objects, products, registry, emitter),
        clock=lambda: NOW, catalog=CATALOG, repo_root=ROOT, render_root=root, gateway=gateway)


class Stack:
    """One database, one object store, and the BASE fixture prepared, estimated, and presented."""

    def __init__(self, conn: Any, objects: Any, root: Path) -> None:
        self.conn, self.objects, self.sink, self.root = conn, objects, io.StringIO(), root
        self.est = estimation_deps(conn, objects, self.sink, Gateway())
        self.runs = {name: self._estimate(name) for name in (BASE, ATTRITION)}
        self.result, self.curator, self.deps = self.present(BASE, 1)

    def _estimate(self, analysis_id: str) -> Any:
        prepared = prepared_outcome(self.conn, self.objects, self.sink, analysis_id,
                                    SPECS[analysis_id])
        found = est_nodes.run_estimation(
            self.est, analysis_id=analysis_id, stage_run_id=f"es:{analysis_id}:1",
            preparation_outcome_artifact_id=prepared, estimation_revision=1)
        assert found.status == "complete", found.error_code
        return found

    def view(self, analysis_id: str) -> Any:
        found = self.committed(analysis_id, "CausalGraphView")[0]
        return pc.ArtifactRef(artifact_id=found, content_hash=str(
            persistence.ProductStore(self.conn).find_artifact_hash(found)))

    def present(self, analysis_id: str, revision: int,
                **over: Any) -> tuple[Any, Curator, ph.PresentationDeps]:
        curator = Curator(**over)
        deps = presentation_deps(self.conn, self.objects, self.sink, curator, self.root)
        outcome, run_id = str(self.runs[analysis_id].outcome_artifact_id), (
            f"ps:{analysis_id}:{revision}")
        found = nodes.run_presentation(
            deps, analysis_id=analysis_id, stage_run_id=run_id, presentation_revision=revision,
            approved_graph_view=self.view(analysis_id),
            handoff_manifest=est_nodes.open_presentation_handoff(
                self.est, analysis_id, outcome, run_id),
            estimation_outcome=pc.ArtifactRef(
                artifact_id=outcome,
                content_hash=str(deps.products.find_artifact_hash(outcome))))
        return found, curator, deps

    def payload(self, artifact_id: str) -> dict[str, Any]:
        found = self.deps.products.load_envelope(artifact_id)
        return dict(json.loads(self.objects.get(found.payload_locator)))

    def committed(self, analysis_id: str, kind: str) -> list[str]:
        return [str(row[0]) for row in self.conn.execute(
            "SELECT artifact_id FROM causal.artifacts WHERE analysis_id = %s"
            " AND artifact_type = %s ORDER BY artifact_id", (analysis_id, kind)).fetchall()]


@pytest.fixture(scope="module")
def stack(postgres_dsn: str, minio_s3: dict[str, Any],
          tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    admin = psycopg.connect(postgres_dsn, autocommit=True)
    name = f"test_{uuid.uuid4().hex[:10]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    admin.close()
    conn = psycopg.connect(postgres_dsn.rsplit("/", 1)[0] + f"/{name}", autocommit=True)
    persistence.apply_migrations(conn, MIGRATIONS)
    yield Stack(conn, persistence.ObjectStore(minio_s3["client"], minio_s3["bucket"]),
                tmp_path_factory.mktemp("renders"))
    conn.close()


# -- the happy path (EV-P5-001, EV-P5-002, EV-P5-006) ---------------------


def test_a_scripted_run_reaches_a_delivered_status_and_commits_one_bundle(stack: Stack) -> None:
    """§6: entry, manifest, ONE curator call, compile, render, gate 5, one committed bundle."""
    assert stack.result.status in pc.DELIVERED_STATUSES, stack.result.detail_codes
    assert stack.result.destination is None and stack.result.error_code is None
    assert len(stack.curator.calls) == 1
    assert stack.curator.calls[0].allowed_tool_ids == ("resolve_registered_layout_facts",)
    assert len(stack.committed(BASE, "PresentationBundle")) == 1
    bundle = stack.payload(str(stack.result.presentation_bundle_id))
    assert bundle["causal_graph_view"]["artifact_id"] == stack.view(BASE).artifact_id
    assert len(bundle["specs"]) == len(bundle["renders"]) == 4
    assert bundle["accessible_tables"] == bundle["renders"]


def test_every_summary_sentence_cites_a_frozen_statement_or_artifact(stack: Stack) -> None:
    """§15/gate 5: no substantive sentence stands without an approved citation."""
    bundle = stack.payload(str(stack.result.presentation_bundle_id))
    book = stack.payload(str(bundle["context_manifest"]["artifact_id"]))
    allowed = frozenset(book["allowlists"]["artifact_ids"]) | set(book["statement_ids"]) | set(
        book["qualification_ids"]) | {bundle["plan"]["artifact_id"]}
    assert ph.summary_codes(str(bundle["summary"]), allowed) == ()
    assert str(bundle["summary"]).count("\n") >= 5


def test_the_run_row_is_terminal_and_carries_the_section_18_record(stack: Stack) -> None:
    """§18: one small row per revision, terminal on exit, artifacts remain the truth."""
    row = stack.conn.execute("SELECT state, bundle_artifact_id, run_record FROM presentation.runs"
                             " WHERE stage_run_id = %s", (f"ps:{BASE}:1",)).fetchone()
    assert row is not None and row[0] == stack.result.status
    assert row[1] == stack.result.presentation_bundle_id
    record = dict(row[2])
    assert set(record["upstream"]) == set(pc.ENTRY_KEYS)
    assert record["identities"]["catalog"] == CATALOG.catalog_version
    assert record["counters"]["tasks"] == 1 and record["counters"]["events"] > 0
    assert record["outcome"]["status"] == stack.result.status
    assert record["completed_at"] is not None


def test_a_mandatory_qualification_stands_beside_the_primary_result(stack: Stack) -> None:
    """§11.4/§15: an approved qualification reaches the primary figure, caption, and summary."""
    assert stack.result.status == "complete"  # the BASE claim carries no qualification at all
    found, _, _ = stack.present(ATTRITION, 1)
    assert found.status == "complete_with_qualifications"
    bundle = stack.payload(str(found.presentation_bundle_id))
    book = stack.payload(str(bundle["context_manifest"]["artifact_id"]))
    plan = stack.payload(str(bundle["plan"]["artifact_id"]))
    wanted = list(book["qualification_ids"])
    primary = [row for row in plan["figures"]
               if "primary_contrast_estimates" in row["visual_evidence_ids"]]
    assert wanted and len(primary) == 1
    assert all(name in primary[0]["qualification_ids"] for name in wanted)
    assert all(name in primary[0]["text"]["caption"] for name in wanted)
    assert all(f"qualification {name}" in bundle["summary"] for name in wanted)


# -- delivery (EV-P5-006, §20) --------------------------------------------


def _args(stack: Stack, out: Path | None, digest: str | None = None) -> Any:
    found = str(stack.result.presentation_bundle_id)
    return type("Args", (), {
        "bundle_id": found, "output_dir": out,
        "expected_bundle_hash": digest or str(stack.deps.products.find_artifact_hash(found))})()


class _Delivery:
    def __init__(self, stack: Stack) -> None:
        self.stack = stack

    def deliver(self, bundle_id: str, expected_hash: str) -> dict[str, Any]:
        from causal.runtime import presentation as pr
        deps = type("Deps", (), {"products": self.stack.deps.products,
                                 "objects": self.stack.objects})()
        return pr.bundle_view(deps, bundle_id, expected_hash)  # type: ignore[arg-type]


def test_delivery_exports_committed_bytes_and_verifies_every_hash(stack: Stack,
                                                                  tmp_path: Path) -> None:
    """§20: an absent target, committed bytes only, and every hash verified after the copy."""
    target = tmp_path / "export"
    found = cli_render.deliver(_Delivery(stack), BASE, _args(stack, target), "cli-1")
    assert found.status == "completed" and found.message_args["figures"] == 4
    assert (target / "summary.txt").read_text(encoding="utf-8").startswith("Estimand")
    assert len(list(target.glob("*.svg"))) == len(list(target.glob("*.png"))) == 4
    assert len(list(target.glob("*.table"))) == 4


def test_delivery_refuses_an_occupied_directory_and_a_wrong_expected_hash(stack: Stack,
                                                                         tmp_path: Path) -> None:
    """§20: a non-empty target and an unnamed bundle are blockers; the bundle is untouched."""
    (busy := tmp_path / "busy").mkdir()
    (busy / "already.txt").write_text("x", encoding="utf-8")
    with pytest.raises(cli_render.DeliveryError) as occupied:
        cli_render.deliver(_Delivery(stack), BASE, _args(stack, busy), "cli-2")
    assert occupied.value.code == cli_render.OCCUPIED
    with pytest.raises(persistence.PersistenceError) as wrong:
        cli_render.deliver(_Delivery(stack), BASE, _args(stack, None, "0" * 64), "cli-3")
    assert wrong.value.code == "presentation_bundle_unavailable"
    assert len(stack.committed(BASE, "PresentationBundle")) == 1


# -- typed non-delivering destinations (EV-P5-003, §17.2) -----------------


def test_a_curator_inability_is_a_typed_needs_template_terminal(stack: Stack) -> None:
    """§9.3/§17.2: repeated or declared inability becomes a catalog-owner status, not a plan."""
    found, _, _ = stack.present(BASE, 2, inability="no_honest_compatible_template")
    assert found.status == "needs_template" and found.presentation_bundle_id is None
    assert found.destination == "presentation-catalog-maintainer"
    assert found.error_code == "no_honest_compatible_template"
    row = stack.conn.execute("SELECT state FROM presentation.runs WHERE stage_run_id = %s",
                             (f"ps:{BASE}:2",)).fetchone()
    assert row == ("needs_template",)


def test_an_unreportable_claim_is_blocked_at_gate_one(stack: Stack) -> None:
    """§5 condition 2/§16 gate 1: an unreportable claim never reaches the curator."""
    deps = presentation_deps(stack.conn, stack.objects, stack.sink, Curator(), stack.root)
    outcome = str(stack.runs[BASE].outcome_artifact_id)
    opened = est_nodes.open_presentation_handoff(stack.est, BASE, outcome, f"ps:{BASE}:9")
    broken = opened.model_copy(update={"handoff_id": f"{opened.handoff_id}-x", "entries": (
        opened.entries[0], stack.view(BASE), *opened.entries[2:])})
    found = nodes.run_presentation(
        deps, analysis_id=BASE, stage_run_id=f"ps:{BASE}:9", presentation_revision=9,
        handoff_manifest=broken, approved_graph_view=stack.view(BASE),
        estimation_outcome=pc.ArtifactRef(
            artifact_id=outcome, content_hash=str(deps.products.find_artifact_hash(outcome))))
    assert found.status == "blocked" and found.destination == "stage-coordinator"
    assert found.presentation_bundle_id is None and found.error_code is not None


# -- replay and the runtime boundary (EV-P5-006, §17.2) -------------------


def test_a_rerun_replays_the_committed_artifacts_without_duplicating_them(stack: Stack) -> None:
    """D-035: a new stage_run_id at the next revision recommits the same deterministic ids."""
    before = {kind: stack.committed(BASE, kind) for kind in
              ("PresentationContextManifest", "FigurePlan", "FigureSpec", "PresentationBundle")}
    found, curator, _ = stack.present(BASE, 1)
    assert found.status == stack.result.status
    assert found.presentation_bundle_id == stack.result.presentation_bundle_id
    assert len(curator.calls) == 1
    assert {kind: stack.committed(BASE, kind) for kind in before} == before


def test_the_runtime_dispatch_reports_the_finished_revision_and_names_delivery(
    stack: Stack
) -> None:
    """§17.2/§20: `causal run` reports a finished revision and `causal status` names delivery."""
    from causal.runtime import presentation as pr
    deps = type("Deps", (), {"conn": stack.conn, "products": stack.deps.products,
                             "objects": stack.objects, "emitter": stack.deps.emitter,
                             "clock": stack.deps.clock})()
    row = pr.latest_stage_run(stack.conn, "presentation", BASE)
    found = pr.present(deps, stack.est, stack.deps, BASE, 1)  # type: ignore[arg-type]
    # D-035: the newest finished revision is reported from its own row, never re-run.
    assert row is not None and found.status == row.state and found.design_revision == 1
    assert found.stage_run_id == row.stage_run_id
    assert pr.latest_stage_run(stack.conn, "presentation", BASE).revision == row.revision
    assert pr.after(row.state, "presentation") == "presentation"
    assert pr.after("running", "presentation") == "run"
