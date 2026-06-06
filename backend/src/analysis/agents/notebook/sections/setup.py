"""Setup & imports section renderer."""

from nbformat.v4 import new_code_cell, new_markdown_cell


# Methods that require scikit-learn
_SKLEARN_METHODS = {
    "ipw", "inverse probability weighting",
    "aipw", "augmented ipw", "doubly robust",
    "psm", "propensity score matching", "matching",
    "double_ml", "double ml", "doubleml",
    "s_learner", "t_learner", "x_learner",
    "causal_forest", "causal forest",
}


def render_setup_cells(
    methods_used: list[str] | None = None,
    has_dag: bool = False,
) -> list:
    """Generate setup and import cells.

    Args:
        methods_used: Method names from pipeline results (for conditional deps).
        has_dag: Whether a causal DAG was produced (needs networkx).
    """
    cells = []
    cells.append(
        new_markdown_cell(
            "## Setup\n\n"
            "Install required packages and import libraries for analysis."
        )
    )

    # Determine required packages based on what the pipeline actually used
    packages = [
        "numpy", "pandas", "matplotlib", "seaborn",
        "scipy", "statsmodels",
    ]

    needs_sklearn = False
    if methods_used:
        method_set = {m.lower().replace("-", "_").replace(" ", "_") for m in methods_used}
        needs_sklearn = bool(method_set & {m.replace(" ", "_") for m in _SKLEARN_METHODS})

    if needs_sklearn:
        packages.append("scikit-learn")
    if has_dag:
        packages.append("networkx")

    install_code = f'''# Install required packages (run once, then restart kernel if needed)
import subprocess, sys
packages = {packages}
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + packages,
    stdout=subprocess.DEVNULL,
)
print(f"Installed {{len(packages)}} packages: {{', '.join(packages)}}")'''

    cells.append(new_code_cell(install_code))

    # Main imports cell
    imports = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
print("Setup complete!")"""

    cells.append(new_code_cell(imports))
    return cells
