# The visualization-catalog loader and accessors, and the thirteen-condition PRD-005 §5 entry
# gate over a T-026-shaped estimation handoff (T-030 §2; PRD-005 §5, §8, §12).

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from causal.estimation.contracts import FigureDataArtifactV1, FigureDataPointV1
from causal.presentation import catalog as vc
from causal.presentation import contracts as pc
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
CATALOG = vc.load_visualization_catalog(REGISTRIES / "visualization-catalog.v1.json")
PROFILE = vc.profile_for_method(CATALOG, "randomized_experiment")
# PRD-005 §12: the evidence question each V1 method must be able to answer honestly.
METHODS = ("randomized_experiment", "aipw", "did", "sharp_rdd")


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=digest(name))


DECLARED = {key: ref(key) for key in pc.ENTRY_KEYS}
GRAPH_VIEW = ref("causal_graph_view")


# One committed FigureDataArtifact, dumped the way the object store holds it (T-026 shape).
def figure_data(evidence_id: str, **over: Any) -> dict[str, Any]:
    point = FigureDataPointV1(series_id="arm_b", category="arm_b", x_value=0.12, y_value=1.0,
                              interval_lower=0.01, interval_upper=0.23, denominator=100)
    payload = FigureDataArtifactV1(
        parents=(ref("estimation_plan"),), versions={"schema": "figure-data-artifact.v1"},
        visual_evidence_id=evidence_id, builder_id=f"{evidence_id}_builder",
        builder_version="figure-builder.v1", points=(point,), units={"y": "rate"},
        labels={"arm_b": "Arm B"}, rule_ids=("suppression.v1",),
        contributing_counts={"arm_b": 100}, contribution_mask_hash=None,
        disclosure_status="reportable").model_dump(mode="json")
    return payload | over


# The five §5 handoff payloads of a passing T-026 handoff, read the way the store holds them.
def payloads() -> dict[str, dict[str, Any]]:
    return {
        "outcome": {"status": "complete",
                    "estimation_bundle": DECLARED["estimation_bundle"].model_dump(mode="json")},
        "bundle": {"primary_result": ref("primary_result").model_dump(mode="json"),
                   "claim_judgment": DECLARED["claim_judgment"].model_dump(mode="json"),
                   "capacity_check": DECLARED["capacity_check"].model_dump(mode="json"),
                   "evidence_bundles": [ref(kind).model_dump(mode="json")
                                        for kind in ("diagnostic", "sensitivity", "figure_data")]},
        "judgment": {"status": "reportable_with_qualifications", "estimand_id": "att",
                     "overall_ceiling": "reportable_with_qualifications",
                     "qualifications": ("attrition above the prespecified threshold",),
                     "items": [{"contrast_id": "arm_b_vs_control", "ceiling": "reportable",
                                "status": "reportable_with_qualifications",
                                "confidence_level": 0.95,
                                "cited_artifact_ids": ["primary_result"]}]},
        "design": {"method_id": "randomized_experiment", "estimand": "att",
                   "visualization_catalog_version": CATALOG.catalog_version},
        "capacity": {"status": "pass", "method_id": "randomized_experiment",
                     "visualization_catalog_version": CATALOG.catalog_version,
                     "cardinalities": {"evidence_families": 4, "primary_result_items": 2}},
        "graph_view": {"artifact_id": GRAPH_VIEW.artifact_id,
                       "content_hash": GRAPH_VIEW.content_hash}}


def entry_inputs(**over: Any) -> vc.EntryInputs:
    data = payloads() | {key: value for key, value in over.items() if key in payloads()}
    return vc.EntryInputs(
        outcome=data["outcome"], bundle=data["bundle"], judgment=data["judgment"],
        design=data["design"], capacity=data["capacity"], graph_view=data["graph_view"],
        declared=over.get("declared", DECLARED), committed=over.get("committed", DECLARED),
        figure_data=over.get("figure_data", {name: figure_data(name)
                                             for name in PROFILE.required_evidence_ids}))


def policy(**over: Any) -> vc.EntryPolicy:
    return vc.EntryPolicy(**({"catalog": CATALOG, "profile": PROFILE,
                              "approved_graph_view": GRAPH_VIEW,
                              "observability_ready": True} | over))


def patched(key: str, **fields: Any) -> vc.EntryInputs:
    return entry_inputs(**{key: payloads()[key] | fields})


class TestCatalogLoader:
    def test_the_registered_catalog_loads_and_covers_every_v1_method(self) -> None:
        assert CATALOG.catalog_version == "visualization-catalog.v1"
        assert tuple(row.method_id for row in CATALOG.profiles) == METHODS

    @pytest.mark.parametrize("method_id", METHODS)
    def test_every_required_evidence_question_has_a_compatible_template(self,
                                                                        method_id: str) -> None:
        profile = vc.profile_for_method(CATALOG, method_id)
        for evidence_id in profile.questions():
            rows = vc.templates_for_evidence(CATALOG, profile, evidence_id)
            assert all(profile.profile_id in row.allowed_profile_ids
                       and evidence_id in row.visual_evidence_ids for row in rows)

    def test_an_unknown_method_resolves_to_no_profile(self) -> None:
        with pytest.raises(pc.PresentationError) as excinfo:
            vc.profile_for_method(CATALOG, "synthetic_control")
        assert excinfo.value.code == vc.UNKNOWN_PROFILE

    def test_an_unregistered_evidence_question_has_no_template(self) -> None:
        with pytest.raises(pc.PresentationError) as excinfo:
            vc.templates_for_evidence(CATALOG, PROFILE, "event_time_evidence")
        assert excinfo.value.code == vc.NO_COMPATIBLE_TEMPLATE

    @pytest.mark.parametrize("text", ["{not json", '{"schema_version": "visualization-catalog.v1"}',
                                      '{"schema_version": "visualization-catalog.v2"}'])
    def test_a_malformed_catalog_fails_closed(self, tmp_path: Path, text: str) -> None:
        broken = tmp_path / "catalog.json"
        broken.write_text(text, encoding="utf-8")
        with pytest.raises(pc.PresentationError) as excinfo:
            vc.load_visualization_catalog(broken)
        assert excinfo.value.code == vc.INVALID_CATALOG_FILE

    def test_a_missing_catalog_file_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(pc.PresentationError):
            vc.load_visualization_catalog(tmp_path / "absent.json")

    def test_capacity_codes_name_every_exceeded_ceiling(self) -> None:
        assert vc.capacity_codes(CATALOG, PROFILE, {"evidence_families": 4}) == ()
        assert vc.capacity_codes(CATALOG, PROFILE, {"evidence_families": 9, "figures": 7}) == (
            f"over_limit:{PROFILE.profile_id}:evidence_families",
            f"over_limit:{PROFILE.profile_id}:figures")


class TestEntryGate:
    def test_a_passing_t026_handoff_opens_the_stage(self) -> None:
        assert vc.entry_codes(entry_inputs(), policy()) == ()

    @pytest.mark.parametrize(("key", "patch", "code"), [
        ("outcome", {"status": "invalidated"}, vc.ESTIMATION_NOT_COMPLETE),
        ("judgment", {"status": "not_reportable"}, vc.CLAIM_NOT_REPORTABLE),
        ("judgment", {"estimand_id": "atu"}, vc.AMBIGUOUS_SELECTION),
        ("judgment", {"overall_ceiling": "not_reportable"}, vc.CLAIM_EXCEEDS_CEILING),
        ("judgment", {"items": []}, vc.INCOMPLETE_PRIMARY),
        ("bundle", {"primary_result": None}, vc.INCOMPLETE_PRIMARY),
        ("design", {"method_id": "aipw"}, vc.AMBIGUOUS_SELECTION),
        ("design", {"visualization_catalog_version": "visualization-catalog.v9"},
         vc.UNSUPPORTED_CATALOG_VERSION),
        ("capacity", {"status": "fail"}, vc.CAPACITY_NOT_PASS),
        ("capacity", {"cardinalities": {"evidence_families": 99}}, vc.CAPACITY_BINDING),
        ("capacity", {"method_id": "did"}, vc.CAPACITY_BINDING),
        ("graph_view", {"content_hash": digest("revised_graph_view")}, vc.GRAPH_VIEW_MISMATCH),
    ])
    def test_each_violated_condition_returns_its_stable_code(self, key: str, patch: dict[str, Any],
                                                             code: str) -> None:
        assert code in vc.entry_codes(patched(key, **patch), policy())

    def test_an_uncommitted_entry_artifact_is_missing(self) -> None:
        held = dict(DECLARED) | {"claim_judgment": None}
        assert vc.MISSING_ARTIFACT in vc.entry_codes(entry_inputs(committed=held), policy())

    def test_an_entry_artifact_that_does_not_hash_true_is_refused(self) -> None:
        held = dict(DECLARED) | {"figure_data_bundle": ref("other_figure_data_bundle")}
        assert vc.ENTRY_HASH_MISMATCH in vc.entry_codes(entry_inputs(committed=held), policy())

    def test_a_claim_citing_unfrozen_evidence_is_refused(self) -> None:
        items = [dict(payloads()["judgment"]["items"][0]) | {"cited_artifact_ids": ["scratch"]}]
        assert vc.UNRESOLVED_EVIDENCE in vc.entry_codes(patched("judgment", items=items), policy())

    @pytest.mark.parametrize(("skip", "patch", "code"), [
        (True, {}, vc.MISSING_FIGURE_DATA),
        (False, {"labels": {}}, vc.INCOMPLETE_FIGURE_DATA),
        (False, {"rows": [{"unit_id": 1}]}, vc.UNAPPROVED_FIGURE_CONTENT)])
    def test_unpresentable_figure_data_is_refused(self, skip: bool, patch: dict[str, Any],
                                                  code: str) -> None:
        wanted = PROFILE.required_evidence_ids[1:] if skip else PROFILE.required_evidence_ids
        held = {name: figure_data(name, **patch) for name in wanted}
        assert code in vc.entry_codes(entry_inputs(figure_data=held), policy())

    def test_a_failed_observability_preflight_closes_the_gate(self) -> None:
        codes = vc.entry_codes(entry_inputs(), policy(observability_ready=False))
        assert codes == (vc.OBSERVABILITY_PREFLIGHT,)

    def test_the_gate_reports_every_failing_condition_at_once(self) -> None:
        codes = vc.entry_codes(entry_inputs(outcome={"status": "failed"},
                                            committed={}, graph_view={}),
                               policy(observability_ready=False))
        assert set(codes) >= {vc.ESTIMATION_NOT_COMPLETE, vc.MISSING_ARTIFACT,
                              vc.GRAPH_VIEW_MISMATCH, vc.OBSERVABILITY_PREFLIGHT}
        assert list(codes) == sorted(codes)
