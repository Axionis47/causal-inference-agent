"""Notebook verification: a passing fixture, a repair, and a final failure."""
from __future__ import annotations

import json

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.notebook_verify import NotebookVerificationAgent
from src.analysis_v2.core import AnalysisRunState, ArtifactKind, AnalysisStage, GateStatus
from src.analysis_v2.persistence import analysis_dir
from src.analysis_v2.spec import NotebookBuildResult

import pandas as pd

JOB_ID = "job-verify"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def _ctx_with_notebook(cells) -> AgentCtx:
    run = AnalysisRunState(job_id=JOB_ID, causal_question="q?")
    run.notebook_build = NotebookBuildResult(
        notebook_artifact_id="notebook/causal_analysis",
        report_artifact_id="report/final_report",
        sections=["x"],
    )
    ctx = AgentCtx(job_id=JOB_ID, run=run, frame=pd.DataFrame({"a": [1]}))
    notebook = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    })
    ctx.add_artifact(
        agent="report_notebook", stage=AnalysisStage.S10_REPORT_NOTEBOOK_CREATED,
        artifact_id="notebook/causal_analysis", kind=ArtifactKind.NOTEBOOK,
        title="nb", relative_path="notebook/causal_analysis.ipynb",
        payload=nbformat.writes(notebook),
    )
    return ctx


async def test_a_clean_notebook_verifies_and_unlocks_the_download(data_dir):
    ctx = _ctx_with_notebook([new_code_cell("x = 2 + 2\nassert x == 4\nprint(x)")])
    agent = NotebookVerificationAgent()
    result = await agent.execute(ctx)
    agent.commit(ctx.run, result.output)

    assert result.gate.status == GateStatus.ADVANCE
    verification = ctx.run.notebook_verification
    assert verification.notebook_status.value == "verified_running"
    assert verification.executed_all_cells is True
    assert verification.attempts == 1
    base = analysis_dir(JOB_ID)
    assert (base / "notebook/causal_analysis_verified.ipynb").exists()
    assert (base / "notebook/notebook_execution_log.json").exists()
    assert (base / "notebook/notebook_preview.html").exists()


async def test_a_broken_notebook_fails_within_three_attempts_with_the_error(data_dir):
    ctx = _ctx_with_notebook(
        [new_code_cell("raise RuntimeError('cell exploded on purpose')")]
    )
    result = await NotebookVerificationAgent().execute(ctx)

    assert result.gate.status == GateStatus.FAIL
    assert "cell exploded on purpose" in result.gate.hard_failures[0]
    output = result.output
    assert output.notebook_status.value == "failed"
    assert output.attempts <= 3
    log = json.loads(
        (analysis_dir(JOB_ID) / "notebook/notebook_execution_log.json").read_text()
    )
    assert log["attempts"][0]["status"] == "failed"


async def test_a_wrong_dataset_path_is_repaired_and_the_rerun_passes(data_dir):
    from src.storage.job_data import job_normalized_dir

    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(
        job_normalized_dir(JOB_ID) / "data.csv.parquet", index=False
    )
    config_dir = analysis_dir(JOB_ID) / "notebook"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "notebook_config.json").write_text(
        json.dumps({"backend_dir": ".", "dataset_path": "../normalized/missing.parquet",
                    "ignored_columns": []})
    )
    ctx = _ctx_with_notebook(
        [
            new_code_cell(
                "import json\nfrom pathlib import Path\nimport pandas as pd\n"
                "CONFIG = json.loads(Path('notebook/notebook_config.json').read_text())\n"
                "df = pd.read_parquet(CONFIG['dataset_path'])\n"
                "print(len(df))"
            )
        ]
    )
    result = await NotebookVerificationAgent().execute(ctx)

    assert result.gate.status == GateStatus.ADVANCE
    assert result.output.attempts == 2
    assert any("dataset_path corrected" in r for r in result.output.repairs)
