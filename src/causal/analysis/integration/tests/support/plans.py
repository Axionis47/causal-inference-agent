"""Prepared handoff and legacy execution-plan builders for integration contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from causal.analysis.integration import RESOURCE_ROOT
from causal.analysis.integration import plancompile as pc
from causal.analysis.integration.contracts import ESTIMATION_REGISTRY_KEYS
from causal.analysis.integration.packs import load_estimation_packs
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[6] / "registries"


CAPACITY_PATH = REGISTRIES / "delivery-capacity.v1.json"


REGISTRY = load_estimation_packs(RESOURCE_ROOT / "method-pack-estimation.v1.json",
                                 REGISTRIES / "method-packs.v1.json")


PACK = REGISTRY.get("randomized_experiment", "randomized-experiment-pack.v1")


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=digest(name))


DECLARED = {key: ref(key) for key in pc.ENTRY_KEYS}


ROW_HASH = digest("rows")


CARDINALITIES = {"arms": 2, "contrasts": 1, "subgroups": 0, "cohorts": 0, "periods": 0,
                 "event_times": 0, "cutoff_sides": 0, "series": 2, "evidence_items": 12}


VERSIONS = dict.fromkeys(ESTIMATION_REGISTRY_KEYS, "pinned.v1") | {
    "capacity": "delivery-capacity.v1", "visualization_catalog": "visualization-catalog.v1"}


def payloads() -> dict[str, dict[str, Any]]:
    return {
        "outcome": {"status": "prepared",
                    "prepared_bundle": DECLARED["prepared_bundle"].model_dump()},
        "bundle": {"compiled_design": DECLARED["compiled_design"].model_dump(),
                   "capacity_report": DECLARED["capacity_report"].model_dump(),
                   "row_set_hash": ROW_HASH, "stabilized_frame_row_set_hash": ROW_HASH,
                   "prepared_frame_row_set_hash": ROW_HASH},
        "design": {"method_id": PACK.method_id, "method_pack_version": PACK.pack_version,
                   "estimand": "att", "comparator": "control", "unit": "participant",
                   "frame": {"population": "enrolled", "timeframe": "wave_one",
                             "outcome": "completion", "treatment": "offer"},
                   "primary_contrasts": ["arm_b_vs_control"],
                   "required_visual_evidence": ["primary_contrast_estimates"],
                   "preparation": {
                       "estimator_input_schema_id": "prepared-frame.v1",
                       "required_roles": ["treatment", "outcome"],
                       "required_final_diagnostic_ids": ["baseline_balance"],
                       "method_structure": {}},
                   "registry_versions": {"schema": "prepared-frame.v1"},
                   "estimator": {"estimand": "att", "parameters": {"estimand": "att"}}},
        "capacity": {"status": "pass",
                     "compiled_design": DECLARED["compiled_design"].model_dump(),
                     "dimensions": [
                         {"dimension": name, "value": value,
                          "applicability": "applicable", "source": "approved design"}
                         for name, value in CARDINALITIES.items()]},
        "record": {"freeze": {"row_set_hash": ROW_HASH},
                   "dispositions": {"counts": [{"disposition": "retained", "row_count": 100}]},
                   "source_row_index": {"row_count": 100}}}


def inputs(**over: Any) -> pc.EntryInputs:
    fields: dict[str, Any] = payloads() | {
        "declared": dict(DECLARED), "committed": dict(DECLARED),
        "estimator_input_types": {"treatment": "categorical", "outcome": "numeric", "group": "categorical"},
        "role_columns": {"treatment": "arm", "outcome": "finished", "group": "cohort"},
        "postrepair_statuses": {"baseline_balance": "pass"}}
    return pc.EntryInputs(**(fields | over))


def policy(**over: Any) -> pc.EntryPolicy:
    fields: dict[str, Any] = {"pack": PACK, "registry_versions": dict(VERSIONS),
                              "numerical_tolerances": {"absolute": 1e-9}}
    return pc.EntryPolicy(**(fields | over))


def manifest() -> Any:
    return pc.compile_context_manifest(inputs(), policy())


def plan(**over: Any) -> Any:
    return pc.compile_plan(manifest(), PACK, ref("context_manifest"), **over)


def structure(**over: Any) -> pc.PreparedStructureV1:
    fields: dict[str, Any] = {"cardinalities": dict(CARDINALITIES),
                              "required_visual_evidence": ("primary_contrast_estimates",),
                              "registry_path": CAPACITY_PATH}
    return pc.PreparedStructureV1(**(fields | over))

