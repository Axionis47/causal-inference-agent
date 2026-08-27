# Close verification (D-091): the composed `causal` chain on a realistic randomized table.
# The pinned five-row wall-10 test documents honest small-sample refusal; this one drives the
# SAME composed path — intake, design, approval, one `causal run` — on 200 rows with real
# signal, through PRD-003, PRD-004, and PRD-005 to one delivered bundle and its §20 export.

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from causal.cli.main import main
from causal.design.contracts import ApprovalDecision
from causal.estimation import judge as ej
from causal.presentation import curate as cu
from causal.runtime import composition, failures
from tests.conftest import requires_docker
from tests.design.test_graph import NOW, ScriptedGateway, stub_renderer
from tests.estimation.test_coordinator_e2e import Gateway as ClaimGateway
from tests.intake.conftest import FILES_RESPONSE, README, FrozenKaggleClient
from tests.presentation.test_coordinator_e2e import Curator
from tests.runtime.test_composition import binding, make_config, submission

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


class Gateway(ScriptedGateway):
    # One receiver for the whole chain: the design script, the §16.2 judge, and the §9 curator.

    delegates: ClassVar[dict[str, Any]] = {ej.TASK_KIND: ClaimGateway(), cu.TASK_KIND: Curator()}

    def invoke(self, envelope: Any, prompt: str, schema: dict[str, object]) -> Any:
        found = self.delegates.get(envelope.task_kind)
        return (found or super()).invoke(envelope, prompt, schema)


def test_the_composed_chain_delivers_one_bundle_on_a_realistic_table(
    conn: Any, minio_s3: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SC §1.1: one `causal run` carries a realistic analysis from an approved design to a
    # committed PresentationBundle, and `causal presentation` exports its frozen bytes.
    for pair in ("AWS_ACCESS_KEY_ID=causal", "AWS_SECRET_ACCESS_KEY=causal-test",
                 "AWS_DEFAULT_REGION=us-east-1"):
        monkeypatch.setenv(*pair.split("=", 1))
    stub_renderer(monkeypatch)
    built = composition.build_runtime(
        make_config(conn, minio_s3, tmp_path), model=Gateway(), strict_observability=False,
        clock=lambda: NOW, client_factory=lambda: FrozenKaggleClient(
            files={"nsw.csv": TABLE, "readme.md": README}, files_response=DECLARED))
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
    bundle = str(done.outcome_artifact_id)
    assert conn.execute(PRESENTATION_ROW, (analysis_id,)).fetchone() == (done.status, bundle)
    assert done.status in ("complete", "complete_with_qualifications"), done.error_code
    # The chain carried real signal: the delivered interval brackets the fixture's +2.0 effect.
    item = failures.payload(built.deps, str(failures.payload(built.deps, body[
        "estimation_bundle"]["artifact_id"])["claim_judgment"]["artifact_id"]))["items"][0]
    assert item["interval_lower"] < 2.0 < item["interval_upper"], item
    # §20: the exact bundle and hash, into an absent directory, with every hash verified.
    target = tmp_path / "export"
    assert main(["presentation", analysis_id, "--bundle-id", bundle, "--expected-bundle-hash",
                 str(built.deps.products.find_artifact_hash(bundle)), "--output-dir",
                 str(target)], lambda: built) == 0
    assert (target / "summary.txt").read_text(encoding="utf-8").startswith("Estimand")
    assert len(list(target.glob("*.svg"))) == len(list(target.glob("*.png"))) == 4
    built.close()
