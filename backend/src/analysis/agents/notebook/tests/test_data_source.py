"""notebook_data_source picks raw when repairs ran, else the final dataframe.

The notebook agent (which bundles the file) and the data-loading renderer (which
writes the load cell) both call this, so the selection must be unambiguous: load
raw only when data_repair both applied repairs and left a raw snapshot on disk.
"""

import pandas as pd

from src.analysis.agents import AnalysisState, DatasetInfo
from src.analysis.agents.notebook.helpers import notebook_data_source


def _state() -> AnalysisState:
    return AnalysisState(
        job_id="ds-test",
        dataset_info=DatasetInfo(url="test://x", name="x"),
    )


def test_returns_final_dataframe_when_no_repairs(tmp_path):
    state = _state()
    final = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(final)
    state.dataframe_path = str(final)

    path, is_raw = notebook_data_source(state)

    assert path == str(final)
    assert is_raw is False


def test_returns_raw_snapshot_when_repairs_applied(tmp_path):
    state = _state()
    final = tmp_path / "data.parquet"
    raw = tmp_path / "data_raw.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(final)
    pd.DataFrame({"a": [1, None]}).to_parquet(raw)
    state.dataframe_path = str(final)
    state.raw_dataframe_path = str(raw)
    state.data_repairs = [{"type": "missing", "strategy": "median", "columns": ["a"]}]

    path, is_raw = notebook_data_source(state)

    assert path == str(raw)
    assert is_raw is True


def test_ignores_raw_snapshot_when_no_repairs_recorded(tmp_path):
    # A raw snapshot on disk without any recorded repairs is not a reason to
    # load raw: there is nothing for the cleaning section to reproduce.
    state = _state()
    final = tmp_path / "data.parquet"
    raw = tmp_path / "data_raw.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(final)
    pd.DataFrame({"a": [1, 2]}).to_parquet(raw)
    state.dataframe_path = str(final)
    state.raw_dataframe_path = str(raw)
    # state.data_repairs left empty

    path, is_raw = notebook_data_source(state)

    assert path == str(final)
    assert is_raw is False


def test_falls_back_to_final_when_raw_snapshot_is_missing(tmp_path):
    # raw_dataframe_path recorded but the file is gone: do not claim raw, or the
    # notebook would point its load cell at a file that was never bundled.
    state = _state()
    final = tmp_path / "data.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(final)
    state.dataframe_path = str(final)
    state.raw_dataframe_path = str(tmp_path / "gone_raw.parquet")
    state.data_repairs = [{"type": "missing", "strategy": "median", "columns": ["a"]}]

    path, is_raw = notebook_data_source(state)

    assert path == str(final)
    assert is_raw is False
