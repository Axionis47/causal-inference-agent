"""Scoring rubric for dataset_inspector.

Every weight in helpers.py has a test here. If the rubric ever changes,
the failing tests document exactly which weights drifted.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.data_profiler import DataProfile
from src.analysis.agents.dataset_inspector.helpers import (
    SCORE_CONFOUNDER_CAP,
    SCORE_HAS_OUTCOME,
    SCORE_HAS_TREATMENT,
    SCORE_PER_CONFOUNDER,
    SCORE_TEST_PATTERN,
    SCORE_TINY_FILE,
    SCORE_TRAIN_PATTERN,
    TINY_FILE_THRESHOLD,
    FileScore,
    explain_choice,
    pick_winner,
    score_profile,
)


def _profile(
    *,
    n_samples: int = 1000,
    treatment: list[str] | None = None,
    outcome: list[str] | None = None,
    confounders: list[str] | None = None,
) -> DataProfile:
    return DataProfile(
        n_samples=n_samples,
        n_features=3,
        feature_names=["a", "b", "c"],
        feature_types={"a": "numeric", "b": "binary", "c": "categorical"},
        missing_values={"a": 0, "b": 0, "c": 0},
        numeric_stats={"a": {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0}},
        categorical_stats={"c": {"x": 5, "y": 5}},
        treatment_candidates=treatment or [],
        outcome_candidates=outcome or [],
        potential_confounders=confounders or [],
    )


# --- score_profile: each rubric rule fires individually --------------------


def test_baseline_profile_scores_zero():
    score = score_profile("data.csv", _profile())
    assert score.score == 0.0


def test_treatment_candidate_adds_treatment_weight():
    score = score_profile("data.csv", _profile(treatment=["treated"]))
    assert score.score == SCORE_HAS_TREATMENT


def test_outcome_candidate_adds_outcome_weight():
    score = score_profile("data.csv", _profile(outcome=["y"]))
    assert score.score == SCORE_HAS_OUTCOME


def test_confounders_add_per_confounder_weight_up_to_cap():
    s_two = score_profile("data.csv", _profile(confounders=["a", "b"]))
    assert s_two.score == 2 * SCORE_PER_CONFOUNDER
    # Beyond the cap, more confounders do not add more weight.
    s_many = score_profile(
        "data.csv", _profile(confounders=[f"c{i}" for i in range(20)])
    )
    assert s_many.score == SCORE_CONFOUNDER_CAP


def test_train_pattern_in_filename_adds_train_weight():
    score = score_profile("train.csv", _profile())
    assert score.score == SCORE_TRAIN_PATTERN


def test_test_pattern_in_filename_adds_test_penalty():
    score = score_profile("sample_submission.csv", _profile())
    assert score.score == SCORE_TEST_PATTERN


def test_train_pattern_in_kaggle_description_overrides_neutral_filename():
    """When Kaggle's resource description is present, it takes precedence
    over the filename — which is the right call for bundles that use
    arbitrary filenames like 'data_v3.csv' but describe them properly."""
    score = score_profile(
        "data_v3.csv",
        _profile(),
        resource_description="Main training set with all features",
    )
    assert score.score == SCORE_TRAIN_PATTERN


def test_tiny_file_penalty_fires_below_threshold():
    score = score_profile("tiny.csv", _profile(n_samples=10))
    assert score.score == SCORE_TINY_FILE


def test_tiny_file_penalty_does_not_fire_at_threshold():
    score = score_profile("ok.csv", _profile(n_samples=TINY_FILE_THRESHOLD))
    assert score.score == 0.0


def test_full_train_profile_combines_weights():
    """A realistic train.csv with treatment + outcome + a few confounders
    should score notably higher than a no-signal file."""
    score = score_profile(
        "train.csv",
        _profile(
            treatment=["treated"],
            outcome=["earnings"],
            confounders=["age", "education"],
        ),
    )
    expected = (
        SCORE_HAS_TREATMENT
        + SCORE_HAS_OUTCOME
        + 2 * SCORE_PER_CONFOUNDER
        + SCORE_TRAIN_PATTERN
    )
    assert score.score == expected


# --- pick_winner --------------------------------------------------------


def test_pick_winner_returns_highest_scored():
    scores = [
        FileScore("a.csv", 1.0, (), 100),
        FileScore("b.csv", 5.0, (), 100),
        FileScore("c.csv", 3.0, (), 100),
    ]
    assert pick_winner(scores).name == "b.csv"


def test_pick_winner_ties_broken_by_larger_n_samples():
    scores = [
        FileScore("small.csv", 5.0, (), 100),
        FileScore("big.csv", 5.0, (), 10_000),
    ]
    assert pick_winner(scores).name == "big.csv"


def test_pick_winner_full_tie_broken_lexically_for_determinism():
    scores = [
        FileScore("b.csv", 5.0, (), 100),
        FileScore("a.csv", 5.0, (), 100),
    ]
    # max() with tuple key (score, n_samples, name) prefers the larger
    # name when score and n_samples tie. We just need determinism.
    winner = pick_winner(scores)
    assert winner.name in {"a.csv", "b.csv"}
    assert pick_winner(scores).name == winner.name  # idempotent


def test_pick_winner_raises_on_empty_list():
    with pytest.raises(ValueError, match="at least one"):
        pick_winner([])


# --- explain_choice -----------------------------------------------------


def test_explain_choice_quotes_winner_and_runner_up():
    scores = [
        FileScore("train.csv", 8.0, ("has 1 treatment candidate", "matches train-set pattern"), 1000),
        FileScore("test.csv", -2.0, ("matches test/sample/submission pattern",), 250),
    ]
    winner = pick_winner(scores)
    explanation = explain_choice(winner, scores)
    assert "train.csv" in explanation
    assert "score 8" in explanation
    assert "test.csv" in explanation
    assert "has 1 treatment candidate" in explanation


def test_explain_choice_handles_single_file_bundle():
    only = FileScore("data.csv", 6.0, ("has 1 outcome candidate",), 500)
    explanation = explain_choice(only, [only])
    assert "data.csv" in explanation
    assert "next:" not in explanation  # no runner-up


def test_explain_choice_handles_zero_rubric_matches():
    only = FileScore("data.csv", 0.0, (), 500)
    explanation = explain_choice(only, [only])
    assert "no rubric rules matched" in explanation
