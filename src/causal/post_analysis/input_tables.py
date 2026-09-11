"""Display tables projected from exact numerical source records; no new statistics."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from causal.analysis.integration import contracts as ac
from causal.post_analysis.visualization.contracts import Column, DataTable, TableRowSource
from causal.shared.contracts import ArtifactRef


def _pointer(name: str) -> str:
    return name.replace("~", "~0").replace("/", "~1")


def _scalar_table(key: str, ref: ArtifactRef, selector: str,
                  values: Mapping[str, Any]) -> DataTable:
    numeric = all(value is None or (isinstance(value, int | float) and not isinstance(value, bool))
                  for value in values.values())
    return DataTable(table_id=key, source=ref, selector=selector, columns=(
        Column(name="quantity", label="Reported quantity", kind="nominal", quantity="quantity name",
               role="dimension"),
        Column(name="value", label="Reported value", kind="quantitative" if numeric else "nominal",
               quantity="supplied value; units are quantity-specific")),
        rows=tuple((name, value) for name, value in values.items()),
        row_sources=tuple(TableRowSource(source=ref, selector=f"{selector}/{_pointer(name)}")
                          for name in values))


def _estimate_table(key: str, source: ArtifactRef,
                    rows: list[tuple[str, ac.PrimaryContrastResultV1, ArtifactRef, str]],
                    ) -> DataTable:
    units = {row.estimate_units for _, row, _, _ in rows}
    unit = next(iter(units)) if len(units) == 1 else None
    return DataTable(table_id=key, source=source, selector="", columns=(
        Column(name="result", label="Result", kind="nominal", quantity="result identity", role="dimension"),
        Column(name="estimate", label="Estimate", kind="quantitative", quantity="effect estimate", units=unit),
        Column(name="interval_lower", label="Interval lower", kind="quantitative", quantity="effect estimate",
               units=unit, role="lower"),
        Column(name="interval_upper", label="Interval upper", kind="quantitative", quantity="effect estimate",
               units=unit, role="upper"),
        Column(name="confidence_level", label="Confidence level", kind="quantitative", quantity="confidence level",
               units="probability"),
        Column(name="estimate_units", label="Units", kind="nominal", quantity="effect scale", role="dimension"),
        Column(name="uncertainty_method", label="Uncertainty method", kind="nominal", quantity="uncertainty method",
               role="dimension")),
        rows=tuple((label, row.estimate, row.interval_lower, row.interval_upper, row.confidence_level,
                    row.estimate_units, row.uncertainty_method) for label, row, _, _ in rows),
        row_sources=tuple(TableRowSource(source=ref, selector=selector) for _, _, ref, selector in rows))



EvidenceRows = dict[str, tuple[ArtifactRef, ac.DiagnosticResultV1 | ac.SensitivityResultV1]]


def _evidence_tables(bundle: ac.NumericalBundleV1, bundle_ref: ArtifactRef,
                     primary: ac.PrimaryAnalysisResultV1, evidence: EvidenceRows,
                     support: ac.AnalysisSupportingDataV1,
                     ) -> tuple[dict[str, DataTable], list[str]]:
    """Project supplied records and compare only explicitly matching effect definitions."""
    tables: dict[str, DataTable] = {}
    primary_rows = [
        (row.contrast_id, row, bundle.primary_result, f"/primary_items/{index}")
        for index, row in enumerate(primary.primary_items) if row.convergence != "not_converged"]
    if primary_rows:
        tables["primary"] = _estimate_table("primary", bundle.primary_result, primary_rows)
    limitations = [("The legacy diagnostic/support measurements lack per-quantity unit descriptors; "
                    "their raw values may be shown in tables, without inventing units or comparable scales.")]
    if len(primary_rows) != len(primary.primary_items):
        limitations.append("Nonconverged primary estimates are retained for audit, not presented as successful fits.")
    branches: list[tuple[str, ac.PrimaryContrastResultV1, ArtifactRef, str]] = []
    for key, (ref, evidence_row) in evidence.items():
        if evidence_row.values:
            tables[key] = _scalar_table(key, ref, "/values", evidence_row.values)
        if isinstance(evidence_row, ac.SensitivityResultV1):
            limitations.append(
                f"No branch-specific diagnostic evidence was supplied for sensitivity {evidence_row.branch_id}.")
            if evidence_row.result is not None and evidence_row.result.convergence != "not_converged":
                projected = (evidence_row.branch_id, evidence_row.result, ref, "/result")
                branches.append(projected)
                tables[key] = _estimate_table(key, ref, [projected])
    for name, values in support.measurements.items():
        if values:
            key = f"support:{name}"
            tables[key] = _scalar_table(key, bundle.supporting_data,
                                        f"/measurements/{_pointer(name)}", values)
    for index, row in enumerate(primary.primary_items):
        if row.convergence == "not_converged":
            continue
        comparable = [branch for branch in branches if (
            branch[1].estimand_id, branch[1].contrast_id, branch[1].estimate_units,
            branch[1].comparator_id) == (
                row.estimand_id, row.contrast_id, row.estimate_units, row.comparator_id)]
        if comparable:
            key = f"comparison:{index}"
            tables[key] = _estimate_table(key, bundle_ref, [
                ("Primary", row, bundle.primary_result, f"/primary_items/{index}"), *comparable])
    return tables, limitations
