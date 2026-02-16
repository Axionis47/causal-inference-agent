"""Setup & imports section renderer."""

from nbformat.v4 import new_code_cell, new_markdown_cell


def render_setup_cells() -> list:
    """Generate setup and import cells."""
    cells = []
    cells.append(
        new_markdown_cell("## Setup\n\nImport required libraries for visualizations.")
    )

    imports = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
print("Setup complete!")"""
    cells.append(new_code_cell(imports))
    return cells
