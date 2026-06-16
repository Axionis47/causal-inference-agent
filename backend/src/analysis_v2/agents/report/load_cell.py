"""The notebook's first code cell: load the analyzable frame and inputs.

Single-file runs read the one normalized parquet (today's behavior, kept
byte-for-byte). When an AssemblyPlan assembled the frame from several files,
the cell calls the same `assemble_from` the backend used, resolving each file
by its relative path from the config, so the notebook reproduces the backend
frame exactly and `notebook_verify`'s re-run holds.

The shared preamble also reads `analysis_inputs.json` and binds the frozen
`PLAN`, `SPEC`, and `DAG` (plain dicts) plus seeds numpy, so the later cells
can recompute the estimate, diagnostics, and figures from the same inputs the
backend used.
"""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

from src.analysis_v2.core import AnalysisRunState

# Imports, config, frozen inputs, seed, and backend path: shared by both loads.
_PREAMBLE = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "np.random.seed(7)\n\n"
    "CONFIG = json.loads(Path('notebook/notebook_config.json').read_text())\n"
    "INPUTS = json.loads(Path('notebook/analysis_inputs.json').read_text())\n"
    "PLAN, SPEC, DAG = INPUTS['plan'], INPUTS['spec'], INPUTS['dag']\n"
    "sys.path.insert(0, CONFIG['backend_dir'])\n"
)

_DROP_AND_SHOW_SINGLE = (
    "df = coerce_bool_columns(pd.read_parquet(CONFIG['dataset_path']))\n"
    "for col in CONFIG['ignored_columns']:\n"
    "    if col in df.columns:\n"
    "        df = df.drop(columns=[col])\n"
    "print(f'{len(df):,} rows x {len(df.columns)} columns')\n"
    "df.head()"
)

_SINGLE_FILE = (
    _PREAMBLE
    + "from src.analysis_v2.runner.loader import coerce_bool_columns\n\n"
    + _DROP_AND_SHOW_SINGLE
)

_ASSEMBLED = (
    _PREAMBLE
    + "from src.analysis_v2.runner.loader import coerce_bool_columns\n"
    + "from src.analysis_v2.runner.assembly import assemble_from\n"
    + "from src.analysis_v2.spec import AssemblyPlan\n\n"
    + "paths = CONFIG['assembly_paths']\n"
    + "def _read(name):\n"
    + "    p = Path(paths[name])\n"
    + "    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)\n\n"
    + "plan = AssemblyPlan.model_validate(CONFIG['assembly_plan'])\n"
    + "df = coerce_bool_columns(assemble_from(_read, plan))\n"
    + "for col in CONFIG['ignored_columns']:\n"
    + "    if col in df.columns:\n"
    + "        df = df.drop(columns=[col])\n"
    + "n_files = 1 + len(plan.concat_files) + len(plan.joins)\n"
    + "print(f'{len(df):,} rows x {len(df.columns)} columns "
    + "(assembled from {n_files} files)')\n"
    + "df.head()"
)


def build_load_cells(run: AnalysisRunState) -> list[nbformat.NotebookNode]:
    """The load section's cells. Assembled when the plan is non-trivial; the
    single-file template otherwise, unchanged from the original notebook."""
    plan = run.assembly_plan
    code = _ASSEMBLED if (plan is not None and not plan.is_trivial) else _SINGLE_FILE
    return [new_markdown_cell("## 1. Load dataset and config"), new_code_cell(code)]
