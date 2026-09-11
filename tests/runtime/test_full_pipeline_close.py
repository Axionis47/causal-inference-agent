# Close verification (D-091): the composed `causal` chain on a realistic randomized table.
# The pinned five-row wall-10 test documents honest small-sample refusal; this one drives the
# SAME composed path — intake, design, approval, one `causal run` — on 200 rows with real
# signal, through PRD-003, PRD-004, and PRD-005 to one delivered bundle and its §20 export.

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from causal.cli.main import main
from causal.design.contracts import ApprovalDecision
from causal.runtime import composition, failures
from tests.design.test_graph import NOW, stub_renderer
from tests.infrastructure import requires_docker
from tests.intake.conftest import FILES_RESPONSE, README, FrozenKaggleClient
from tests.runtime.test_composition import PipelineGateway, binding, make_config, submission
from tests.shared.test_tracing import FakeTracer

pytestmark = requires_docker

PRESENTATION_ROW = "SELECT state, bundle_artifact_id FROM presentation.runs WHERE analysis_id = %s"


def trial(units: int = 200) -> bytes:
    # One frozen randomized table: a +2.0 treatment effect on earnings, over four sites. Every
    # value is a deterministic function of the row number, so the effect the chain recovers is
    # pinned by the fixture and not by a seed. Four per cent of the rows carry no earnings at
    # all, which is the outcome-missingness rule the approved design already declares.
    rows = []
    for index in range(units):
        treated = index % 2
        earnings = 10.0 + 2.0 * treated + 0.75 * (index % 7) + 0.1 * (index % 11)
        rows.append(f"{index + 1},{'' if index % 25 == 3 else f'{earnings:.3f}'},"
                    f"{'treated' if treated else 'control'},site-{index % 4}")
    return ("\n".join(["unit_id,earnings,group,site", *rows]) + "\n").encode()


TABLE = trial()
SITE = {"name": "site", "description": "randomisation stratum", "type": "string", "order": 3}
DECLARED: dict[str, Any] = {"files": [dict(FILES_RESPONSE["files"][0]) | {  # type: ignore[arg-type]
    "totalBytes": len(TABLE),
    "columns": [*FILES_RESPONSE["files"][0]["columns"], SITE]}]}  # type: ignore[index]


def test_the_composed_chain_delivers_one_bundle_on_a_realistic_table(
    conn: Any, minio_s3: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SC §1.1: one `causal run` carries a realistic analysis from an approved design to a
    # committed PresentationBundle, and `causal presentation` exports its frozen bytes.
    for pair in ("AWS_ACCESS_KEY_ID=causal", "AWS_SECRET_ACCESS_KEY=causal-test",
                 "AWS_DEFAULT_REGION=us-east-1"):
        monkeypatch.setenv(*pair.split("=", 1))
    stub_renderer(monkeypatch)
    gateway = PipelineGateway()
    built = composition.build_runtime(
        make_config(conn, minio_s3, tmp_path), model=gateway, strict_observability=False,
        clock=lambda: NOW, client_factory=lambda: FrozenKaggleClient(
            files={"nsw.csv": TABLE, "readme.md": README}, files_response=DECLARED))
    built.pres = replace(built.pres, require_tracing=False, tracer=FakeTracer(),
                         render_root=tmp_path / "renders")
    analysis_id = (made := built.new(submission())).analysis_id
    opened = built.run(analysis_id, expected_stage_run=made.stage_run_id, idempotency_key="k-run")
    assert built.approve_design(analysis_id, binding(opened), ApprovalDecision.APPROVED,
                                "k-approve").status == "approved"
    done = built.run(analysis_id, expected_stage_run=opened.stage_run_id, idempotency_key="k-2")
    prepared = failures.latest_preparation_run(conn, analysis_id)
    estimated = failures.latest_estimation_run(conn, analysis_id)
    assert prepared is not None and estimated is not None
    assert failures.payload(built.deps, str(prepared.outcome_artifact_id))["status"] == "prepared"
    body = failures.payload(built.deps, str(estimated.outcome_artifact_id))
    assert body["status"] == "complete", body.get("error_code")
    assert done.status == "complete", json.dumps(conn.execute(
        "SELECT run_record->'result' FROM presentation.runs WHERE analysis_id = %s",
        (analysis_id,)).fetchone(), indent=2)
    bundle = str(done.outcome_artifact_id)
    assert conn.execute(PRESENTATION_ROW, (analysis_id,)).fetchone() == (done.status, bundle)
    delivered = failures.payload(built.deps, bundle)
    report = failures.payload(built.deps, delivered["draft"]["artifact_id"])
    context = failures.payload(built.deps, delivered["context"]["artifact_id"])
    assert set(report["coverage"]) == set(context["required_evidence"])
    assert all(statement["citations"] for section in report["sections"]
               for statement in section["statements"])
    # The chain carried real signal: the delivered interval brackets the fixture's +2.0 effect.
    numerical = failures.payload(built.deps, body["estimation_bundle"]["artifact_id"])
    item = failures.payload(built.deps, numerical["primary_result"]["artifact_id"])["primary_items"][0]
    assert item["interval_lower"] < 2.0 < item["interval_upper"], item
    assert "claim_judgment" not in numerical and "figure_data_bundle" not in numerical
    exported = failures.payload(built.deps, delivered["export"]["artifact_id"])
    assert gateway.post.review_calls == 1
    assert gateway.post.preview_hashes == [exported["object_hashes"][key]
                                          for key in exported["preview_keys"]]
    # §20: the exact bundle and hash, into an absent directory, with every hash verified.
    target = tmp_path / "export"
    assert main(["presentation", analysis_id, "--bundle-id", bundle, "--expected-bundle-hash",
                 str(built.deps.products.find_artifact_hash(bundle)), "--output-dir",
                 str(target)], lambda: built) == 0
    assert "Causal interpretation remains conditional" in (target / "summary.txt").read_text()
    assert (target / "report.html").read_bytes() == built.pres.objects.get(exported["objects"]["html"])
    delivered_bytes = {path.read_bytes() for path in target.iterdir() if path.is_file()}
    assert all(built.pres.objects.get(locator) in delivered_bytes
               for locator in exported["objects"].values())
    assert any(path.suffix == ".svg" for path in target.iterdir())
    assert any(path.suffix == ".png" for path in target.iterdir())
    built.close()
