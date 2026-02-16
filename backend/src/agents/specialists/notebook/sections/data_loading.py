"""Data loading section renderer."""

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState


def render_data_loading(state: AnalysisState) -> list:
    """Generate data loading cells using actual pipeline path."""
    cells = []
    cells.append(new_markdown_cell("## Data Loading\n\nLoad the dataset used in the analysis."))

    data_path = state.dataset_info.local_path
    if not data_path and hasattr(state, "dataframe_path") and state.dataframe_path:
        data_path = state.dataframe_path

    if data_path:
        if str(data_path).endswith(".parquet"):
            read_call = f'df = pd.read_parquet("{data_path}")'
        else:
            read_call = f'df = pd.read_csv("{data_path}")'

        load_code = f'''# Dataset path from pipeline
DATA_PATH = "{data_path}"
{read_call}

print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
df.head()'''
    else:
        url = state.dataset_info.url or "unknown"
        load_code = f'''# NOTE: Update DATA_PATH to your local copy of the dataset.
# Original source: {url}
DATA_PATH = "UPDATE_THIS_PATH.csv"

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
df.head()'''

    cells.append(new_code_cell(load_code))
    return cells
