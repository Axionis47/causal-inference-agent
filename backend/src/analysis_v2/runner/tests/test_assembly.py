"""assemble_frame: build one analyzable frame from a multi-file bundle."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.core import AnalysisRunState
from src.analysis_v2.runner.assembly import DatasetUnavailable, assemble_frame
from src.analysis_v2.runner.graph import _rebuild_frame
from src.analysis_v2.runner.loader import load_analysis_frame
from src.analysis_v2.spec import AssemblyJoin, AssemblyPlan
from src.analysis_v2.state import AnalysisState, DatasetInfo, JobStatus
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, write_manifest

JOB_ID = "job-assembly"


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


def _stage(frames: dict[str, pd.DataFrame], winner: str) -> DatasetManifest:
    files = []
    for name, df in frames.items():
        parquet_name = f"{name}.parquet"
        df.to_parquet(job_normalized_dir(JOB_ID) / parquet_name, index=False)
        files.append(
            ManifestFile(
                name=name, relative_path=f"raw/{name}", size_bytes=1, format="csv",
                sha256="0" * 64, used=(name == winner),
                normalized_path=f"normalized/{parquet_name}", tabular=True,
                columns=list(df.columns), n_rows=len(df),
            )
        )
    manifest = DatasetManifest(
        job_id=JOB_ID, kaggle_url="https://www.kaggle.com/x", raw_dir="raw",
        winner=winner, files=files,
    )
    write_manifest(JOB_ID, manifest)
    return manifest


def _state() -> AnalysisState:
    return AnalysisState(
        job_id=JOB_ID,
        dataset_info=DatasetInfo(url="https://www.kaggle.com/x", name="x"),
        causal_question="does x cause y?",
        status=JobStatus.CONFIRMED,
    )


def test_single_file_plan_returns_the_base_frame_unchanged(storage_dir):
    base = pd.DataFrame({"id": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    manifest = _stage({"base.csv": base}, winner="base.csv")

    out = assemble_frame(JOB_ID, manifest, AssemblyPlan.single_file("base.csv"))

    pd.testing.assert_frame_equal(out, base)


def test_loader_default_equals_the_explicit_single_file_plan(storage_dir):
    # load_analysis_frame(plan=None) is today's behavior; it must equal the
    # explicit single_file plan, proving the refactor changed nothing.
    base = pd.DataFrame({"treatment": [1, 0, 1], "y": [1.0, 2.0, 3.0]})
    _stage({"base.csv": base}, winner="base.csv")

    default = load_analysis_frame(_state())
    explicit = load_analysis_frame(_state(), AssemblyPlan.single_file("base.csv"))

    pd.testing.assert_frame_equal(default, explicit)
    pd.testing.assert_frame_equal(default, base)


def test_left_join_brings_sibling_columns_without_dropping_base_rows(storage_dir):
    facts = pd.DataFrame({"store_id": [1, 1, 2], "sales": [10.0, 12.0, 9.0]})
    lookup = pd.DataFrame({"store_id": [1, 2], "store_type": ["a", "b"]})
    manifest = _stage({"facts.csv": facts, "store.csv": lookup}, winner="facts.csv")
    plan = AssemblyPlan(
        base_file="facts.csv",
        joins=[AssemblyJoin(right_file="store.csv", on=["store_id"], how="left")],
    )

    out = assemble_frame(JOB_ID, manifest, plan)

    assert list(out.columns) == ["store_id", "sales", "store_type"]
    assert len(out) == 3  # left join keeps every base row
    assert out["store_type"].tolist() == ["a", "a", "b"]


def test_same_schema_concat_stacks_the_shards(storage_dir):
    jan = pd.DataFrame({"id": [1, 2], "y": [1.0, 2.0]})
    feb = pd.DataFrame({"id": [3, 4], "y": [3.0, 4.0]})
    manifest = _stage({"jan.csv": jan, "feb.csv": feb}, winner="jan.csv")
    plan = AssemblyPlan(base_file="jan.csv", concat_files=["feb.csv"])

    out = assemble_frame(JOB_ID, manifest, plan)

    assert len(out) == 4
    assert out["id"].tolist() == [1, 2, 3, 4]


def test_drop_columns_trims_after_assembly_and_ignores_absent_names(storage_dir):
    base = pd.DataFrame({"id": [1, 2], "y": [1.0, 2.0], "notes": ["a", "b"]})
    manifest = _stage({"base.csv": base}, winner="base.csv")
    plan = AssemblyPlan(base_file="base.csv", drop_columns=["notes", "ghost"])

    out = assemble_frame(JOB_ID, manifest, plan)

    assert list(out.columns) == ["id", "y"]  # ghost ignored, notes dropped


def test_missing_join_key_raises_cleanly(storage_dir):
    facts = pd.DataFrame({"store_id": [1, 2], "sales": [10.0, 9.0]})
    lookup = pd.DataFrame({"other": [1, 2], "store_type": ["a", "b"]})
    manifest = _stage({"facts.csv": facts, "store.csv": lookup}, winner="facts.csv")
    plan = AssemblyPlan(
        base_file="facts.csv",
        joins=[AssemblyJoin(right_file="store.csv", on=["store_id"])],
    )

    with pytest.raises(DatasetUnavailable):
        assemble_frame(JOB_ID, manifest, plan)


def test_missing_file_raises_cleanly(storage_dir):
    base = pd.DataFrame({"id": [1], "y": [1.0]})
    manifest = _stage({"base.csv": base}, winner="base.csv")
    plan = AssemblyPlan(base_file="base.csv", concat_files=["ghost.csv"])

    with pytest.raises(DatasetUnavailable):
        assemble_frame(JOB_ID, manifest, plan)


def test_rebuild_frame_swaps_ctx_frame_to_the_assembled_one(storage_dir):
    # The documented highest risk: the runner must replace ctx.frame after S0A
    # so S1+ see the assembled data, not the provisional winner.
    facts = pd.DataFrame({"store_id": [1, 1, 2], "sales": [10.0, 12.0, 9.0]})
    lookup = pd.DataFrame({"store_id": [1, 2], "store_type": ["a", "b"]})
    _stage({"facts.csv": facts, "store.csv": lookup}, winner="facts.csv")
    run = AnalysisRunState(
        job_id=JOB_ID, causal_question="q?",
        assembly_plan=AssemblyPlan(
            base_file="facts.csv",
            joins=[AssemblyJoin(right_file="store.csv", on=["store_id"])],
        ),
    )
    ctx = AgentCtx(
        job_id=JOB_ID, run=run,
        frame=pd.DataFrame({"placeholder": [0]}), input_state=_state(),
    )

    _rebuild_frame(ctx)

    assert list(ctx.frame.columns) == ["store_id", "sales", "store_type"]
    assert len(ctx.frame) == 3


def test_rebuild_frame_leaves_a_trivial_plan_frame_untouched(storage_dir):
    base = pd.DataFrame({"id": [1], "y": [1.0]})
    _stage({"base.csv": base}, winner="base.csv")
    run = AnalysisRunState(
        job_id=JOB_ID, causal_question="q?",
        assembly_plan=AssemblyPlan.single_file("base.csv"),
    )
    sentinel = pd.DataFrame({"placeholder": [0]})
    ctx = AgentCtx(job_id=JOB_ID, run=run, frame=sentinel, input_state=_state())

    _rebuild_frame(ctx)

    assert ctx.frame is sentinel  # trivial plan -> no reload, frame left in place
