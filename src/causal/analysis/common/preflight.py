"""Observable data checks only. No estimator is called by this module."""
from __future__ import annotations

import hashlib
from typing import Any

from causal.analysis.common.catalog import method_module
from causal.analysis.common.models import Category, Issue
from causal.analysis.contracts import AnalysisSpecification, DatasetIdentity, Measurement


def identify_data(data: object, name: str) -> DatasetIdentity:
    """Hash the entire frame, including schema, row order, nulls and unused columns."""
    import polars as pl

    if not isinstance(data, pl.DataFrame):
        raise TypeError("data must be a Polars DataFrame")
    encoded = data.rechunk().write_ipc(None, compression="uncompressed")
    return DatasetIdentity(name=name, content_hash=hashlib.sha256(encoded.getvalue()).hexdigest())


def _data_issue(field: str, finding: str, requirement: str,
                category: Category = "incompatible_data") -> Issue:
    return Issue(category=category, field=field, finding=finding, requirement=requirement,
                 explanation="Resolve this data prerequisite before attempting estimation.",
                 resolutions=("Correct the data or propose a supported configuration.",))


def inspect_data(spec: AnalysisSpecification, data: object
                 ) -> tuple[tuple[Issue, ...], tuple[Measurement, ...]]:
    import polars as pl

    if not isinstance(data, pl.DataFrame):
        raise TypeError("data must be a Polars DataFrame")
    module = method_module(spec.configuration.method)
    issues: list[Issue] = []
    required = [(row.name, row.kind) for row in module.columns(spec.configuration)]
    required.append((spec.design.outcome.column, "numeric"))
    for name, kind in required:
        if name not in data.columns:
            issues.append(_data_issue(name, f"Column {name!r} is absent.",
                                      "Supply the declared column.", "missing_data"))
            continue
        series = data[name]
        numeric = series.dtype.is_numeric()
        if series.dtype.base_type() in (pl.List, pl.Array, pl.Struct, pl.Object, pl.Binary):
            issues.append(_data_issue(name, f"Column {name!r} has unsupported non-scalar type {series.dtype}.",
                                      "Scalar numerical, categorical or identifier values."))
        if kind in ("numeric", "binary", "time") and not numeric:
            issues.append(_data_issue(name, f"Column {name!r} has type {series.dtype}.",
                                      f"This configuration requires a {kind} column."))
        if numeric and series.cast(pl.Float64).is_infinite().any():
            issues.append(_data_issue(name, "Column contains infinite values.", "Finite values."))
        if series.dtype.is_float() and series.is_nan().any():
            issues.append(_data_issue(name, "Column contains NaN values.",
                                      "Finite values or explicitly supported null outcomes."))
        allow_null = name == spec.design.outcome.column and spec.configuration.method == "randomized"
        if series.null_count() and not allow_null:
            issues.append(_data_issue(name, "Column contains missing values.",
                                      "Complete values for this declared role."))
    outcome = spec.design.outcome.column
    if outcome in data.columns:
        values: Any = data[outcome].drop_nulls()
        if not len(values):
            issues.append(_data_issue(outcome, "No observed outcomes.", "Observed outcomes."))
        elif spec.design.outcome.kind == "binary" and not set(values.to_list()) <= {0, 1}:
            issues.append(_data_issue(outcome, "Outcome values contradict the binary measurement.",
                                      "Binary outcomes encoded as 0 and 1."))
    if not data.height:
        issues.append(_data_issue("dataset", "The frame is empty.", "Contributing observations."))
    if not issues:
        issues.extend(module.check_data(spec.configuration, data))
        if hasattr(module, "check_sensitivity_data"):
            issues.extend(module.check_sensitivity_data(spec.configuration, data, spec.sensitivities))
        if spec.configuration.method == "randomized":
            observed = data.filter(pl.col(outcome).is_not_null())
            issues.extend(module.check_data(spec.configuration, observed))
    observations = (Measurement(name="rows", value=data.height),
                    Measurement(name="columns", value=data.width))
    return tuple(issues), observations
