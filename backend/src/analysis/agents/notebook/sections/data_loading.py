"""Data loading section renderer."""

from pathlib import Path

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.analysis.agents.base import AnalysisState

from ..helpers import notebook_data_source


def render_data_loading(state: AnalysisState) -> list:
    """Generate data loading cells using portable relative path."""
    cells = []

    # The notebook agent bundles whichever file this returns under the same
    # name, so the load cell must read the same one. is_raw means repairs ran
    # and the cleaning section below transforms this raw data into the final set.
    data_source, is_raw = notebook_data_source(state)

    if is_raw:
        intro = (
            "## Data Loading\n\n"
            "Load the **raw** dataset bundled with this notebook. The data "
            "cleaning section below reproduces, in order, the repairs the "
            "pipeline applied, turning this raw data into the analysis-ready "
            "dataset used for everything that follows."
        )
    else:
        intro = (
            "## Data Loading\n\n"
            "Load the dataset bundled alongside this notebook."
        )
    cells.append(new_markdown_cell(intro))

    ext = Path(data_source).suffix if data_source else ".parquet"
    data_filename = f"data_{state.job_id}{ext}"

    # Kaggle source URL for provenance
    source_url = state.dataset_info.url if state.dataset_info else "unknown"

    if ext == ".parquet":
        read_call = f'df = pd.read_parquet(DATA_PATH)'
    else:
        read_call = f'df = pd.read_csv(DATA_PATH)'

    load_code = f'''# Dataset bundled with this notebook for reproducibility
# Original source: {source_url}
# Jupyter sets the kernel's working directory to the notebook folder by default,
# so the bundled data file resolves relative to cwd without filesystem gymnastics.
import os

DATA_FILENAME = "{data_filename}"

if not os.path.exists(DATA_FILENAME):
    raise FileNotFoundError(
        f"Data file '{{DATA_FILENAME}}' not found in the working directory "
        f"({{os.getcwd()!r}}). Place the bundled data file alongside this "
        f"notebook before running.\\nOriginal source: {source_url}"
    )

{read_call.replace("DATA_PATH", "DATA_FILENAME")}

print(f"Dataset shape: {{df.shape}}")
print(f"Columns ({{len(df.columns)}}): {{list(df.columns)}}")
df.head()'''

    cells.append(new_code_cell(load_code))
    return cells
