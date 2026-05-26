"""Executes a generated notebook end-to-end on synthetic data with NaNs.

Section renderers used to silently emit cells that broke when the user opened
the notebook on data with missing values. Three concrete bugs hit in
practice: a string-literal "__file__" in the data-loader, np.std on a
pre-clean Y in the E-value cell, and an OLS placebo that fitted Y against a
permutation of T_binary whose lengths no longer matched. None of these
crashed notebook *generation*, so structure-only tests passed while the
shipped notebook was broken.

This test runs nbclient against the rendered cells and asserts no cell
errors. NaNs are deliberately injected so the verification path exercises
the dropna/index-alignment code that was buggy before.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_notebook

from src.analysis.agents import (
    AnalysisState,
    DatasetInfo,
    DataProfile,
    TreatmentEffectResult,
)
from src.analysis.agents.base.state import SensitivityResult
from src.analysis.agents.specialists.notebook.sections.data_loading import render_data_loading
from src.analysis.agents.specialists.notebook.sections.sensitivity import render_sensitivity
from src.analysis.agents.specialists.notebook.sections.treatment_effects import (
    render_treatment_effects,
)


def _build_dataset_with_nans(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    """Synthetic confounded dataset; ~5% of rows have a NaN in some column.

    The NaN injection is what catches the Y vs Y_clean and the T_binary
    length-mismatch bugs: without dropna kicking in, those code paths
    look identical in execution.
    """
    age = rng.normal(40, 10, n)
    income = rng.normal(50, 15, n)
    treat = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.05 * age + 0.02 * income - 4)))).astype(int)
    outcome = 2.0 * treat + 0.05 * age + 0.03 * income + rng.normal(0, 1, n)

    df = pd.DataFrame({"treat": treat, "outcome": outcome, "age": age, "income": income})

    # Drop ~5% of values from each column (treatment column kept clean to
    # exercise the path where T_binary has no NaNs but the covariate frame
    # does, which is the realistic case after the pipeline's repair stage).
    for col in ("outcome", "age", "income"):
        idx = rng.choice(n, size=max(1, n // 20), replace=False)
        df.loc[idx, col] = np.nan
    return df


def _build_state(job_id: str, n: int) -> AnalysisState:
    state = AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(url="test://synthetic", name="synthetic"),
        treatment_variable="treat",
        outcome_variable="outcome",
    )
    state.data_profile = DataProfile(
        n_samples=n,
        n_features=4,
        feature_names=["treat", "outcome", "age", "income"],
        feature_types={"treat": "binary", "outcome": "numeric", "age": "numeric", "income": "numeric"},
        missing_values={"treat": 0, "outcome": 20, "age": 20, "income": 20},
        numeric_stats={},
        categorical_stats={},
        potential_confounders=["age", "income"],
    )
    state.treatment_effects = [
        TreatmentEffectResult(
            method="OLS",
            estimand="ATE",
            estimate=2.0,
            std_error=0.1,
            ci_lower=1.8,
            ci_upper=2.2,
            p_value=0.0,
        )
    ]
    state.sensitivity_results = [
        SensitivityResult(
            method="E-value",
            robustness_value=2.5,
            interpretation="Moderate robustness to unmeasured confounding.",
            details={},
        )
    ]
    return state


def test_generated_notebook_executes_on_data_with_nans(tmp_path: Path):
    """Render data-loading + treatment-effects + sensitivity, execute, expect no errors."""
    rng = np.random.default_rng(42)
    job_id = "exec-test-001"
    df = _build_dataset_with_nans(rng)

    # Bundle data file under the filename the data-loading cell expects.
    data_path = tmp_path / f"data_{job_id}.parquet"
    df.to_parquet(data_path)

    state = _build_state(job_id=job_id, n=len(df))
    state.dataframe_path = str(data_path)

    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    # Imports; skip the pip-install cell to keep the test fast and offline.
    nb.cells.append(new_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import statsmodels.api as sm\n"
    ))
    nb.cells.extend(render_data_loading(state))
    nb.cells.extend(render_treatment_effects(state))
    nb.cells.extend(render_sensitivity(state))

    nb_path = tmp_path / f"causal_analysis_{job_id}.ipynb"
    nbformat.write(nb, str(nb_path))

    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()

    errors = [
        out for cell in nb.cells if cell.cell_type == "code"
        for out in cell.get("outputs", []) if out.output_type == "error"
    ]
    assert not errors, "Cell errors:\n" + "\n---\n".join(
        f"{e['ename']}: {e['evalue']}\n" + "\n".join(e.get("traceback", []))
        for e in errors
    )
