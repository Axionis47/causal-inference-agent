"""The notebook load cell and config: single-file unchanged, assembly replayed."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.report.agent import ReportNotebookAgent
from src.analysis_v2.agents.report.load_cell import build_load_cells
from src.analysis_v2.core import AnalysisRunState
from src.analysis_v2.spec import AssemblyJoin, AssemblyPlan
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, write_manifest

JOB_ID = "job-load-cell"


def _run(plan: AssemblyPlan | None) -> AnalysisRunState:
    return AnalysisRunState(
        job_id=JOB_ID, causal_question="does x cause y?", assembly_plan=plan
    )


def _code(run: AnalysisRunState) -> str:
    return [c for c in build_load_cells(run) if c.cell_type == "code"][0].source


def test_single_file_load_cell_reads_one_parquet_and_does_not_assemble():
    code = _code(_run(None))
    assert "pd.read_parquet(CONFIG['dataset_path'])" in code
    assert "assemble_from" not in code


def test_trivial_plan_keeps_the_single_file_cell():
    code = _code(_run(AssemblyPlan.single_file("data.csv")))
    assert "pd.read_parquet(CONFIG['dataset_path'])" in code
    assert "assemble_from" not in code


def test_assembled_load_cell_replays_the_plan_through_assemble_from():
    plan = AssemblyPlan(
        base_file="facts.csv",
        joins=[AssemblyJoin(right_file="store.csv", on=["store_id"])],
    )
    code = _code(_run(plan))
    assert "from src.analysis_v2.runner.assembly import assemble_from" in code
    assert "AssemblyPlan.model_validate(CONFIG['assembly_plan'])" in code
    assert "assemble_from(_read, plan)" in code
    assert "pd.read_parquet(CONFIG['dataset_path'])" not in code


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


def _stage(names: list[str], winner: str) -> None:
    files = []
    for name in names:
        parquet = f"{name}.parquet"
        pd.DataFrame({"store_id": [1, 2]}).to_parquet(
            job_normalized_dir(JOB_ID) / parquet, index=False
        )
        files.append(
            ManifestFile(
                name=name, relative_path=f"raw/{name}", size_bytes=1, format="csv",
                sha256="0" * 64, used=(name == winner),
                normalized_path=f"normalized/{parquet}", tabular=True,
            )
        )
    write_manifest(
        JOB_ID,
        DatasetManifest(
            job_id=JOB_ID, kaggle_url="https://www.kaggle.com/x",
            raw_dir="raw", winner=winner, files=files,
        ),
    )


def test_config_emits_assembly_paths_for_a_non_trivial_plan(storage_dir):
    _stage(["facts.csv", "store.csv"], winner="facts.csv")
    plan = AssemblyPlan(
        base_file="facts.csv",
        joins=[AssemblyJoin(right_file="store.csv", on=["store_id"])],
    )
    config = ReportNotebookAgent._notebook_config(_run(plan))

    assert config["assembly_plan"]["base_file"] == "facts.csv"
    assert config["assembly_paths"] == {
        "facts.csv": "../normalized/facts.csv.parquet",
        "store.csv": "../normalized/store.csv.parquet",
    }
    assert config["dataset_path"] is None


def test_config_single_file_keeps_dataset_path_and_no_assembly(storage_dir):
    _stage(["data.csv"], winner="data.csv")
    config = ReportNotebookAgent._notebook_config(_run(None))

    assert config["dataset_path"] == "../normalized/data.csv.parquet"
    assert "assembly_plan" not in config
    assert "assembly_paths" not in config
