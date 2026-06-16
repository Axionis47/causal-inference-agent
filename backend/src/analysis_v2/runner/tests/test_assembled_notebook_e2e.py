"""End-to-end replay proof: an assembled (joined) frame survives the notebook.

A two-file bundle is joined by an AssemblyPlan; the report notebook's assemble
cell must reproduce the backend's joined frame so notebook_verify's lane re-run
matches the recorded estimate within 1%. This is the core replay contract for
multi-file datasets: the cell calls the same assemble_from the backend used,
resolving each file by its relative path.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.method_lane.lanes import LANES
from src.analysis_v2.agents.notebook_verify import NotebookVerificationAgent
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.agents.report.agent import ReportNotebookAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind
from src.analysis_v2.persistence import analysis_dir
from src.analysis_v2.spec import (
    AssemblyJoin,
    AssemblyPlan,
    CausalSpec,
    ClaimCritique,
    ClaimStrength,
    Confidence,
    DesignCandidate,
    MethodLane,
    MethodPlan,
    QuestionType,
    VariableRef,
)
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, write_manifest

JOB_ID = "job-assembled-e2e"


@pytest.fixture
def storage_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod
    from src.storage.local_storage import reset_local_storage_client

    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()
    yield tmp_path
    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()


def _stage_bundle() -> pd.DataFrame:
    """Facts table missing a region-level confounder that lives in a lookup."""
    rng = np.random.default_rng(0)
    n = 200
    region = rng.integers(0, 4, n)
    gdp = (region + 1).astype(float)
    treat = (rng.random(n) < 0.4).astype(int)
    y = 1.0 + 2.0 * treat + 0.5 * gdp + rng.normal(0, 0.3, n)
    facts = pd.DataFrame({"region_id": region, "treat": treat, "y": y})
    lookup = pd.DataFrame({"region_id": [0, 1, 2, 3], "gdp": [1.0, 2.0, 3.0, 4.0]})
    files = []
    for name, df in {"facts.csv": facts, "lookup.csv": lookup}.items():
        df.to_parquet(job_normalized_dir(JOB_ID) / f"{name}.parquet", index=False)
        files.append(
            ManifestFile(
                name=name, relative_path=f"raw/{name}", size_bytes=1, format="csv",
                sha256="0" * 64, used=(name == "facts.csv"),
                normalized_path=f"normalized/{name}.parquet", tabular=True,
                columns=list(df.columns), n_rows=len(df),
            )
        )
    write_manifest(
        JOB_ID,
        DatasetManifest(
            job_id=JOB_ID, kaggle_url="https://www.kaggle.com/x", raw_dir="raw",
            winner="facts.csv", files=files,
        ),
    )
    return facts.merge(lookup, on="region_id", how="left")


async def test_assembled_notebook_executes_and_reverifies_the_estimate(
    storage_dir, monkeypatch
):
    monkeypatch.setattr(
        "src.analysis_v2.agents.report.agent.get_llm_client", lambda: object()
    )
    joined = _stage_bundle()
    plan = AssemblyPlan(
        base_file="facts.csv",
        joins=[AssemblyJoin(right_file="lookup.csv", on=["region_id"], how="left")],
    )
    run = AnalysisRunState(
        job_id=JOB_ID, causal_question="Does treat raise y?", assembly_plan=plan
    )
    run.causal_spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="y"), treatment=VariableRef(column="treat"),
        candidate_confounders=["gdp"],
    )
    run.dataset_profile = build_profile_summary(joined)
    run.method_plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="y", treatment="treat", covariates=["gdp"],
    )
    # the recorded estimate IS the lane output on the backend's joined frame
    run.estimate_result = LANES[MethodLane.OBSERVATIONAL](
        joined, run.method_plan, run.causal_spec
    ).result
    run.selected_design = DesignCandidate(
        lane=MethodLane.OBSERVATIONAL, design_label="regression adjustment",
        confidence=Confidence.MEDIUM, rationale="adjust for gdp",
    )
    run.claim_critique = ClaimCritique(
        strength=ClaimStrength.MODERATE, allowed_language=["suggests"],
        forbidden_language=["proves"], limitations=["unmeasured confounding"],
        rationale="observational, adjusted for gdp",
    )

    ctx = AgentCtx(job_id=JOB_ID, run=run, frame=joined)
    ctx.add_artifact(
        agent="profiling", stage=AnalysisStage.S2_PROFILE_CREATED,
        artifact_id="profiling/dataset_profile", kind=ArtifactKind.JSON,
        title="profile", relative_path="profiling/dataset_profile.json",
        payload=run.dataset_profile.model_dump(mode="json"),
    )

    report = ReportNotebookAgent()
    report.commit(run, (await report.execute(ctx)).output)

    # the config drives replay: the plan + relative paths, no single dataset_path
    cfg = json.loads(
        (analysis_dir(JOB_ID) / "notebook" / "notebook_config.json").read_text()
    )
    assert cfg["assembly_plan"]["joins"][0]["right_file"] == "lookup.csv"
    assert cfg["assembly_paths"]["lookup.csv"] == "../normalized/lookup.csv.parquet"
    assert cfg["dataset_path"] is None

    verify = NotebookVerificationAgent()
    verify.commit(run, (await verify.execute(ctx)).output)

    # the notebook assembled the join, re-ran the lane, and matched within 1%
    assert run.notebook_verification.notebook_status.value == "verified_running"
    assert run.notebook_verification.executed_all_cells is True
