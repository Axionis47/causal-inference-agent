"""Data repairs section renderer.

Generates both a markdown summary of repairs applied by the pipeline
AND executable code cells that reproduce each repair step.
"""

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState


def _repair_code_cell(
    repair: dict,
    treatment_var: str,
    outcome_var: str,
) -> str | None:
    """Map a single repair record to executable Python code.

    Returns None if the repair type/strategy is unrecognised.
    """
    rtype = repair.get("type", "")
    strategy = repair.get("strategy", "")
    columns = repair.get("columns", [])

    if not columns:
        return None

    cols_repr = repr(columns)

    # ── Missing-value repairs ──────────────────────────────────
    if rtype == "missing":
        if strategy == "drop_rows":
            return (
                f"# Missing-value repair: drop rows with NaN in affected columns\n"
                f"_cols = {cols_repr}\n"
                f"_before = len(df)\n"
                f"df = df.dropna(subset=[c for c in _cols if c in df.columns])\n"
                f"print(f'Dropped {{_before - len(df)}} rows with missing values  →  {{len(df)}} rows remain')"
            )

        if strategy in ("median", "mean"):
            fn = strategy
            return (
                f"# Missing-value repair: impute with column {fn}\n"
                f"_cols = {cols_repr}\n"
                f"for _c in _cols:\n"
                f"    if _c in df.columns and df[_c].isna().any():\n"
                f"        _fill = df[_c].{fn}()\n"
                f"        _n = df[_c].isna().sum()\n"
                f"        df[_c] = df[_c].fillna(_fill)\n"
                f"        print(f'  {{_c}}: filled {{_n}} NaN with {fn}={{_fill:.4f}}')"
            )

        if strategy == "mode":
            return (
                f"# Missing-value repair: impute with column mode\n"
                f"_cols = {cols_repr}\n"
                f"for _c in _cols:\n"
                f"    if _c in df.columns and df[_c].isna().any():\n"
                f"        _mode = df[_c].mode().iloc[0]\n"
                f"        _n = df[_c].isna().sum()\n"
                f"        df[_c] = df[_c].fillna(_mode)\n"
                f"        print(f'  {{_c}}: filled {{_n}} NaN with mode={{_mode}}')"
            )

        if strategy == "iterative":
            return (
                f"# Missing-value repair: iterative imputation (MICE)\n"
                f"from sklearn.experimental import enable_iterative_imputer  # noqa\n"
                f"from sklearn.impute import IterativeImputer\n"
                f"\n"
                f"_cols = {cols_repr}\n"
                f"_num = [c for c in _cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]\n"
                f"if _num:\n"
                f"    _imp = IterativeImputer(max_iter=10, random_state=42)\n"
                f"    df[_num] = _imp.fit_transform(df[_num])\n"
                f"    print(f'Iteratively imputed {{len(_num)}} columns: {{_num}}')"
            )

        if strategy == "drop_columns":
            return (
                f"# Missing-value repair: drop columns with excessive missingness\n"
                f"_cols = {cols_repr}\n"
                f"_to_drop = [c for c in _cols if c in df.columns]\n"
                f"df = df.drop(columns=_to_drop)\n"
                f"print(f'Dropped {{len(_to_drop)}} columns: {{_to_drop}}')"
            )

    # ── Outlier repairs ────────────────────────────────────────
    if rtype == "outliers":
        if strategy == "winsorize":
            return (
                f"# Outlier repair: winsorize at 1st/99th percentiles\n"
                f"from scipy.stats.mstats import winsorize as _winsorize\n"
                f"\n"
                f"_cols = {cols_repr}\n"
                f"for _c in _cols:\n"
                f"    if _c in df.columns and pd.api.types.is_numeric_dtype(df[_c]):\n"
                f"        df[_c] = _winsorize(df[_c].values, limits=[0.01, 0.01])\n"
                f"        print(f'  {{_c}}: winsorized at 1st/99th percentiles')"
            )

        if strategy in ("clip", "iqr_removal"):
            return (
                f"# Outlier repair: IQR-based clipping\n"
                f"_cols = {cols_repr}\n"
                f"for _c in _cols:\n"
                f"    if _c in df.columns and pd.api.types.is_numeric_dtype(df[_c]):\n"
                f"        _q1 = df[_c].quantile(0.25)\n"
                f"        _q3 = df[_c].quantile(0.75)\n"
                f"        _iqr = _q3 - _q1\n"
                f"        _lo, _hi = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr\n"
                f"        _n = ((df[_c] < _lo) | (df[_c] > _hi)).sum()\n"
                f"        df[_c] = df[_c].clip(_lo, _hi)\n"
                f"        print(f'  {{_c}}: clipped {{_n}} outliers to [{{_lo:.2f}}, {{_hi:.2f}}]')"
            )

    # ── Collinearity repairs ───────────────────────────────────
    if rtype == "collinearity":
        # Protect treatment and outcome from being dropped
        return (
            f"# Collinearity repair: drop highly collinear columns\n"
            f"_cols = {cols_repr}\n"
            f"_protected = {{'{treatment_var}', '{outcome_var}'}}\n"
            f"_to_drop = [c for c in _cols if c in df.columns and c not in _protected]\n"
            f"df = df.drop(columns=_to_drop)\n"
            f"print(f'Dropped {{len(_to_drop)}} collinear columns: {{_to_drop}}')"
        )

    return None


def render_data_repairs(state: AnalysisState) -> list:
    """Report data repair agent findings AND emit executable repair cells."""
    cells = []
    repairs = state.data_repairs
    if not repairs:
        return cells

    # ── Markdown summary (existing) ────────────────────────────
    md = "## Data Preprocessing & Repairs\n\n"
    md += "*Findings from the Data Repair agent. "
    md += "These repairs were applied before causal analysis.*\n\n"

    md += "### Repairs Applied\n\n"
    md += "| # | Type | Strategy | Columns |\n"
    md += "|---|------|----------|---------|\n"
    for i, repair in enumerate(repairs, 1):
        rtype = repair.get("type", "unknown")
        strategy = repair.get("strategy", "unknown")
        columns = repair.get("columns", [])
        col_str = ", ".join(columns[:5])
        if len(columns) > 5:
            col_str += f" (+{len(columns) - 5} more)"
        md += f"| {i} | {rtype} | {strategy} | {col_str} |\n"
    md += "\n"

    # Detail each repair type
    missing_repairs = [r for r in repairs if r.get("type") == "missing"]
    outlier_repairs = [r for r in repairs if r.get("type") == "outliers"]
    collinearity_repairs = [r for r in repairs if r.get("type") == "collinearity"]

    if missing_repairs:
        md += "### Missing Value Handling\n\n"
        for r in missing_repairs:
            strategy = r.get("strategy", "unknown")
            columns = r.get("columns", [])
            md += f"- **Strategy**: {strategy}\n"
            md += f"- **Columns**: {', '.join(columns)}\n"
            if r.get("rows_dropped"):
                md += f"- **Rows dropped**: {r['rows_dropped']:,}\n"
            if r.get("before") is not None and r.get("after") is not None:
                md += f"- **Missing values**: {r['before']:,} → {r['after']:,}\n"
            md += "\n"

    if outlier_repairs:
        md += "### Outlier Treatment\n\n"
        for r in outlier_repairs:
            strategy = r.get("strategy", "unknown")
            columns = r.get("columns", [])
            md += f"- **Strategy**: {strategy}\n"
            md += f"- **Columns**: {', '.join(columns)}\n\n"

    if collinearity_repairs:
        md += "### Collinearity Resolution\n\n"
        for r in collinearity_repairs:
            strategy = r.get("strategy", "unknown")
            columns = r.get("columns", [])
            md += f"- **Strategy**: {strategy}\n"
            md += f"- **Columns removed/adjusted**: {', '.join(columns)}\n\n"

    # Quality assessment and cautions
    quality = None
    cautions = None
    for r in repairs:
        if "quality_assessment" in r:
            quality = r["quality_assessment"]
        if "cautions" in r:
            cautions = r["cautions"]

    if quality:
        md += f"### Data Quality Assessment\n\n{quality}\n\n"

    if cautions:
        md += "### Cautions\n\n"
        md += "*These caveats may affect the validity of downstream results.*\n\n"
        for c in cautions:
            md += f"- {c}\n"
        md += "\n"

    cells.append(new_markdown_cell(md))

    # ── Executable repair cells ────────────────────────────────
    repair_note = (
        "### Reproducible Repair Code\n\n"
        "> **Note**: The bundled dataset (`df`) has already been repaired by the pipeline. "
        "The cells below document the exact repairs applied. Running them on the "
        "already-repaired data should produce no changes (0 values repaired), serving "
        "as a verification that the bundled data matches expectations."
    )
    cells.append(new_markdown_cell(repair_note))

    treatment_var = state.treatment_variable or ""
    outcome_var = state.outcome_variable or ""

    emitted_any = False
    for i, repair in enumerate(repairs, 1):
        code = _repair_code_cell(repair, treatment_var, outcome_var)
        if code:
            rtype = repair.get("type", "unknown")
            strategy = repair.get("strategy", "unknown")
            header = f"# Repair {i}: {rtype} — {strategy}\n"
            cells.append(new_code_cell(header + code))
            emitted_any = True

    # Verification cell
    if emitted_any:
        verify_code = (
            "# Verify: data state after repairs\n"
            "print(f'DataFrame shape: {df.shape}')\n"
            "_missing = df.isnull().sum()\n"
            "_missing_total = _missing.sum()\n"
            "print(f'Total remaining missing values: {_missing_total}')\n"
            "if _missing_total > 0:\n"
            "    print('\\nColumns with missing values:')\n"
            "    print(_missing[_missing > 0].to_string())\n"
            "else:\n"
            "    print('No missing values — data is clean.')"
        )
        cells.append(new_code_cell(verify_code))

    return cells
