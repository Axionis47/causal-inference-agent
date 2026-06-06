"""Data loading section renderer."""

from pathlib import Path

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.analysis.agents.base import AnalysisState


def render_data_loading(state: AnalysisState) -> list:
    """Generate data loading cells using portable relative path."""
    cells = []
    cells.append(new_markdown_cell(
        "## Data Loading\n\n"
        "Load the dataset bundled alongside this notebook."
    ))

    # Determine the data filename that was bundled by helpers.save_notebook()
    data_source = state.dataframe_path or (
        state.dataset_info.local_path if state.dataset_info else None
    )
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
