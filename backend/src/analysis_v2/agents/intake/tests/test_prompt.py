"""Prompt construction: labeled channels and bounded size."""
from __future__ import annotations

from src.analysis_v2.agents.intake.prompt import MAX_COLUMNS, build_prompt


def _cols(n: int) -> list[tuple[str, str, str | None]]:
    return [(f"col_{i}", "numeric", f"description of column {i}") for i in range(n)]


def test_both_context_channels_appear_separately_labeled():
    prompt = build_prompt(
        causal_question="Does training increase earnings?",
        columns=_cols(3),
        user_context="The analyst suspects prior earnings confound this.",
        kaggle_description="NSW job training study uploaded from MatchIt.",
        dataset_name="lalonde",
    )
    assert "[user_context]" in prompt
    assert "[kaggle_description]" in prompt
    # the channels stay in their own blocks, separated by content
    assert prompt.index("[user_context]") < prompt.index("suspects prior earnings")
    assert prompt.index("[kaggle_description]") < prompt.index("MatchIt")


def test_missing_context_is_surfaced_not_invented():
    prompt = build_prompt(
        causal_question="Does X cause Y?",
        columns=_cols(2),
        user_context=None,
        kaggle_description="  ",
        dataset_name=None,
    )
    assert "No analyst notes and no dataset description" in prompt
    assert "[user_context]" not in prompt


def test_column_listing_is_capped_with_a_more_marker():
    prompt = build_prompt(
        causal_question="Does X cause Y?",
        columns=_cols(MAX_COLUMNS + 25),
        user_context=None,
        kaggle_description=None,
        dataset_name=None,
    )
    assert f"(+25 more columns omitted)" in prompt
    assert f"col_{MAX_COLUMNS - 1}" in prompt
    assert f"col_{MAX_COLUMNS}" not in prompt


def test_prompt_size_stays_bounded_on_a_pathological_dataset():
    huge = [
        (f"column_with_a_rather_long_name_{i}", "categorical", "x" * 5000)
        for i in range(500)
    ]
    prompt = build_prompt(
        causal_question="Q?" * 200,
        columns=huge,
        user_context="u" * 50_000,
        kaggle_description="k" * 50_000,
        dataset_name="wide",
    )
    # regression bound: capped columns + clipped descriptions + clipped channels
    assert len(prompt) < 25_000
