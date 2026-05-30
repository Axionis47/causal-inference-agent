"""Treatment binarization code-string used by every verification cell.

Returns a Python snippet (as a string) that, when executed inside a
notebook, defines `T` and `T_binary` consistent with how the pipeline
treated the column: explicit value mapping, stored threshold, or the
auto-detect fallback.
"""

from src.analysis.agents.base import AnalysisState


def binarization_code(state: AnalysisState) -> str:
    """Generate the binarization snippet matching the pipeline's encoding."""
    treatment = state.treatment_variable

    if state.treatment_encoding and state.treatment_encoding.value_mapping:
        mapping = state.treatment_encoding.value_mapping
        mapping_repr = repr(mapping)
        return (
            f"# Treatment encoding from pipeline (categorical → numeric)\n"
            f"_mapping = {mapping_repr}\n"
            f"T = df['{treatment}'].map(_mapping).values.astype(float)\n"
            f"_unmapped = np.isnan(T).sum()\n"
            f"if _unmapped > 0:\n"
            f"    print(f'Warning: {{_unmapped}} unmapped treatment values')\n"
            f"T_binary = T  # Already encoded\n"
            f"print(f'Treatment encoded via mapping: {{_mapping}}')\n"
        )

    if state.treatment_binarization_threshold is not None:
        thr = state.treatment_binarization_threshold
        return (
            f"# Binarize continuous treatment at pipeline threshold\n"
            f"T = df['{treatment}'].values.astype(float)\n"
            f"T_binary = (T > {thr}).astype(int)\n"
            f"print(f'Treatment binarized at pipeline threshold ({thr:.4f}): "
            f"{{T_binary.sum()}} treated, {{len(T_binary) - T_binary.sum()}} control')\n"
        )

    return (
        f"# Binarize treatment (auto-detect)\n"
        f"T = df['{treatment}'].values.astype(float)\n"
        f"if len(np.unique(T[~np.isnan(T)])) <= 2:\n"
        f"    T_binary = T.astype(int)\n"
        f"    print(f'Treatment already binary: {{T_binary.sum()}} treated, "
        f"{{len(T_binary) - T_binary.sum()}} control')\n"
        f"else:\n"
        f"    _median_t = np.median(T[~np.isnan(T)])\n"
        f"    T_binary = (T > _median_t).astype(int)\n"
        f"    print(f'Treatment binarized at median ({{_median_t:.4f}}): "
        f"{{T_binary.sum()}} treated, {{len(T_binary) - T_binary.sum()}} control')\n"
    )
