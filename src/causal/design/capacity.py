"""Scientific cardinality measurements, with historical registry readers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError

from causal.design.contracts import _Row
from causal.design.packs import (
    INVALID_REGISTRY_FILE,
    MethodPackV1,
    PackRegistryError,
)
from causal.design.semantics import RoleName
from causal.design.v2 import (
    CapacityReportV2,
    CapacityValueV2,
    DesignFactSetV2,
)
from causal.shared.contracts import ArtifactRef, Identity

__all__ = [
    "CapacityRegistryError", "CapacityRegistryV1", "VisualizationTemplateV1",
    "compile_capacity_report",
    "load_capacity_registry",
]

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


def _cardinality(facts: DesignFactSetV2, profile: Mapping[str, Any],
                 role: RoleName) -> int | None:
    columns = facts.columns(role)
    if not columns:
        return None
    value = (profile.get("columns") or {}).get(columns[0], {}).get("cardinality")
    return int(value) if value is not None else None


def compile_capacity_report(
    *, pack: MethodPackV1, facts: DesignFactSetV2, profile: Mapping[str, Any],
    contrast_count: int, compiled_design: ArtifactRef, registry: CapacityRegistryV1,
) -> CapacityReportV2:
    """Record applicable scientific counts without prescribing or gating report layout.

    The registry argument preserves the existing caller contract. Its historical display
    limits have no authority over a new scientific design or post-analysis presentation.
    """
    treatment = _cardinality(facts, profile, RoleName.TREATMENT)
    groups = _cardinality(facts, profile, RoleName.GROUP)
    periods = _cardinality(facts, profile, RoleName.TIME)
    required = pack.required_roles
    na = (None, "not applicable")
    applicable: dict[str, tuple[int | None, str]] = {
        "arms": (treatment, "treatment cardinality"),
        "contrasts": (contrast_count, "compiled primary contrasts"),
        "subgroups": na,
        "cohorts": (groups, "group cardinality") if RoleName.GROUP in required else na,
        "periods": (periods, "time cardinality") if RoleName.TIME in required else na,
        "event_times": na,
        "cutoff_sides": (2, "compiled cutoff partition") if RoleName.RUNNING_VARIABLE in required else na,
        "series": na,
        "evidence_items": na,
    }
    relevant = {name for name, (_, source) in applicable.items() if source != "not applicable"}
    dimensions = tuple(CapacityValueV2(
        dimension=name, value=value, applicability=(
            "applicable" if value is not None else
            "unknown" if name in relevant else "not_applicable"), source=source)
        for name, (value, source) in applicable.items())
    return CapacityReportV2(
        compiled_design=compiled_design, dimensions=dimensions,
        compatible_template_ids=(), issues=(), status="pass")
