"""Delivery-capacity preflight against the frozen template registry (PRD-002 §13.5; SC §11)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import Field, ValidationError

from causal.design.contracts import _Row
from causal.design.frame import CAPACITY_DIMENSIONS, CapacityStatus, DeliveryCapacityCheckV1
from causal.design.packs import (
    INVALID_REGISTRY_FILE,
    MethodPackRegistry,
    MethodPackV1,
    PackRegistryError,
)
from causal.shared.contracts import Identity

__all__ = [
    "MAX_CONCURRENCY", "CapacityRegistryError", "CapacityRegistryV1",
    "VisualizationTemplateV1", "check_capacity", "load_capacity_registry",
]

UNKNOWN_DIMENSION: Final = "unknown_dimension"
MAX_CONCURRENCY: Final = 8
_LIMIT_FIELDS: Final = ("max_panels", "max_series", "max_labels", "max_annotations")
# Which template limits each cardinality dimension must fit inside (T-012 §5).
_DIMENSION_FIT: Final[dict[str, tuple[str, ...]]] = {
    "arms": ("max_series", "max_panels"), "contrasts": ("max_labels", "max_annotations"),
    "subgroups": ("max_panels",), "cohorts": ("max_panels",), "periods": ("max_series",),
    "event_times": ("max_series",), "cutoff_sides": ("max_panels",), "series": ("max_series",),
    "evidence_items": (),
}

_Limit = Annotated[int, Field(ge=1)]


class CapacityRegistryError(PackRegistryError):
    """A capacity registry operation failed; `code` is a stable contract value."""


class VisualizationTemplateV1(_Row):
    """One registered template: the evidence it can carry and its layout limits."""

    template_id: Identity
    visual_evidence_ids: Annotated[tuple[Identity, ...], Field(min_length=1)]
    max_panels: _Limit
    max_series: _Limit
    max_labels: _Limit
    max_annotations: _Limit


class CapacityRegistryV1(_Row):
    """The immutable visualization catalog and delivery-capacity registry (SC §11)."""

    visualization_catalog_version: Literal["visualization-catalog.v1"]
    capacity_registry_version: Literal["delivery-capacity.v1"]
    accessible_table_max_rows: _Limit
    max_concurrency: _Limit
    templates: Annotated[tuple[VisualizationTemplateV1, ...], Field(min_length=1)]


def load_capacity_registry(path: Path) -> CapacityRegistryV1:
    """Load the capacity registry; an unreadable, invalid, or repeated row fails closed."""
    try:
        registry = CapacityRegistryV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise CapacityRegistryError(
            f"invalid capacity registry {path}: {error}", INVALID_REGISTRY_FILE) from error
    ids = [template.template_id for template in registry.templates]
    if len(set(ids)) != len(ids):
        raise CapacityRegistryError("duplicate template id", INVALID_REGISTRY_FILE)
    return registry


# The closed dimension vector: an unnamed dimension is zero, an unknown one fails closed.
def _dimensions(cardinalities: Mapping[str, int]) -> dict[str, int]:
    if unknown := sorted(set(cardinalities) - set(CAPACITY_DIMENSIONS)):
        raise CapacityRegistryError(f"unknown cardinalities: {unknown}", UNKNOWN_DIMENSION)
    return {dimension: int(cardinalities.get(dimension, 0)) for dimension in CAPACITY_DIMENSIONS}


# One `over_limit` code per dimension the template cannot carry.
def _violations(template: VisualizationTemplateV1, dimensions: Mapping[str, int]) -> list[str]:
    return [
        f"over_limit:{template.template_id}:{dimension}"
        for dimension, fields in _DIMENSION_FIT.items()
        if any(dimensions[dimension] > int(getattr(template, field)) for field in fields)
    ]


def check_capacity(
    pack: MethodPackV1, cardinalities: Mapping[str, int],
    required_visual_evidence: tuple[str, ...], *, registry: CapacityRegistryV1,
) -> DeliveryCapacityCheckV1:
    """Prove one registered delivery path carries every required visual; never invent one."""
    dimensions = _dimensions(cardinalities)
    failures: list[str] = []
    compatible: list[str] = []
    limits: dict[str, int] = {}
    for evidence_id in required_visual_evidence:
        candidates = [t for t in registry.templates if evidence_id in t.visual_evidence_ids]
        if not candidates:
            failures.append(f"no_template:{evidence_id}")
            continue
        broken = {t.template_id: _violations(t, dimensions) for t in candidates}
        limits |= {f"{t.template_id}:{field}": int(getattr(t, field))
                   for t in candidates for field in _LIMIT_FIELDS}
        if fitting := [name for name, codes in broken.items() if not codes]:
            compatible.extend(fitting)
        else:
            failures.extend(code for codes in broken.values() for code in codes)
    if dimensions["evidence_items"] > registry.accessible_table_max_rows:
        failures.append("over_limit:accessible_table:evidence_items")
    concurrency = min(MAX_CONCURRENCY, registry.max_concurrency)
    return DeliveryCapacityCheckV1(
        method_id=pack.method_id, method_profile_id=pack.pack_version, cardinalities=dimensions,
        required_visual_evidence=tuple(required_visual_evidence),
        compatible_templates=tuple(dict.fromkeys(compatible)), template_limits=limits,
        accessible_table_capacity=registry.accessible_table_max_rows,
        execution_concurrency=concurrency, render_concurrency=concurrency,
        status=CapacityStatus.FAIL if failures else CapacityStatus.PASS,
        failure_codes=tuple(dict.fromkeys(failures)),
        visualization_catalog_version=registry.visualization_catalog_version,
        capacity_registry_version=registry.capacity_registry_version,
        method_registry_version=MethodPackRegistry.registry_version)
