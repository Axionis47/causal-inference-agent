"""CausalSpec: taxonomy completeness and variable references."""
from __future__ import annotations

from src.analysis_v2.spec import CausalSpec, Confidence, QuestionType, VariableRef

MANIFEST_TAXONOMY = [
    "simple_effect",
    "binary_treatment",
    "dose_response",
    "multi_factor",
    "interaction",
    "mediation",
    "did",
    "rdd",
    "iv",
    "time_series_intervention",
    "survival",
    "before_after",
    "heterogeneous_effects",
    "driver_analysis",
    "no_effect",
    "mechanism_search",
]


def test_question_type_matches_the_manifest_taxonomy_exactly():
    assert [t.value for t in QuestionType] == MANIFEST_TAXONOMY


def test_variable_ref_is_resolved_only_when_a_column_is_set():
    assert not VariableRef().resolved
    assert not VariableRef(candidates=["re78", "re75"], clue="earnings").resolved
    assert VariableRef(column="re78").resolved


def test_a_realistic_did_spec_round_trips_json():
    spec = CausalSpec(
        question_type=QuestionType.DID,
        type_candidates=[QuestionType.DID, QuestionType.BEFORE_AFTER],
        confidence=Confidence.MEDIUM,
        outcome=VariableRef(column="fte_employment"),
        treatment=VariableRef(derived=True, clue="NJ stores after the wage rise"),
        time_column=VariableRef(column="period"),
        group_column=VariableRef(column="state"),
        candidate_confounders=["chain", "co_owned"],
        missing_info=["post-period start needs confirmation"],
    )
    loaded = CausalSpec.model_validate(spec.model_dump(mode="json"))
    assert loaded.question_type == QuestionType.DID
    assert loaded.treatment.derived and not loaded.treatment.resolved
    assert loaded.missing_info == ["post-period start needs confirmation"]
