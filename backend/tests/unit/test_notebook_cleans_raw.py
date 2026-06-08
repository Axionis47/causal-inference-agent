"""Executes the data-loading + repairs cells on RAW data and asserts they clean it.

The notebook used to load the already-cleaned dataframe and emit repair cells that
ran on clean data and changed nothing (a note even said they "should produce 0
changes"). Slice 6 makes the notebook load the RAW dataframe so those cells
genuinely reproduce the cleaning. This test bundles raw data with injected NaNs and
an outlier, renders the loading + repairs cells, executes them, and asserts the
missing values are removed and the outlier clipped. If the cells were decorative,
the assertion cell raises and the test fails.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_notebook

from src.analysis.agents import AnalysisState, DatasetInfo
from src.analysis.agents.notebook.sections.data_loading import render_data_loading
from src.analysis.agents.notebook.sections.data_repairs import render_data_repairs


def test_repair_cells_clean_the_raw_dataset(tmp_path: Path):
    rng = np.random.default_rng(7)
    job_id = "clean-raw-001"
    n = 200

    df = pd.DataFrame(
        {
            "treat": rng.integers(0, 2, n),
            "outcome": rng.normal(0, 1, n),
            "age": rng.normal(40, 10, n),
            "income": rng.normal(50, 15, n),
        }
    )
    # Inject NaNs into the columns the median repair imputes; add an extreme
    # income value so the IQR clip cell has something to do post-impute.
    for col in ("age", "income"):
        idx = rng.choice(n, size=n // 10, replace=False)
        df.loc[idx, col] = np.nan
    df.loc[0, "income"] = 10_000.0

    # Bundle the RAW data under the name the loading cell reads from cwd.
    raw_path = tmp_path / f"data_{job_id}.parquet"
    df.to_parquet(raw_path)

    state = AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(url="test://synthetic", name="synthetic"),
        treatment_variable="treat",
        outcome_variable="outcome",
    )
    state.dataframe_path = str(raw_path)
    state.raw_dataframe_path = str(raw_path)
    state.data_repairs = [
        {"type": "missing", "strategy": "median", "columns": ["age", "income"]},
        {"type": "outliers", "strategy": "clip", "columns": ["income"]},
    ]

    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb.cells.append(new_code_cell("import numpy as np\nimport pandas as pd\n"))
    nb.cells.extend(render_data_loading(state))
    nb.cells.extend(render_data_repairs(state))
    # Regression guard: if the repair cells did nothing (loaded clean data or were
    # decorative), the injected NaNs and outlier survive and these asserts raise.
    nb.cells.append(
        new_code_cell(
            "assert df.isnull().sum().sum() == 0, "
            "f'raw not cleaned: {int(df.isnull().sum().sum())} missing remain'\n"
            "assert df['income'].max() < 10_000.0, 'outlier not clipped'\n"
            "assert len(df) == 200, 'median impute must not drop rows'\n"
        )
    )

    nbformat.write(nb, str(tmp_path / f"causal_analysis_{job_id}.ipynb"))
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()

    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.output_type == "error"
    ]
    assert not errors, "Cell errors:\n" + "\n---\n".join(
        f"{e['ename']}: {e['evalue']}\n" + "\n".join(e.get("traceback", []))
        for e in errors
    )
