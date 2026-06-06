"""Tests for the EDA notebook section's caption rendering.

The EDA agent writes per-plot captions into state.eda_result.plot_captions
during finalize_eda. The section renderer must emit each caption as a
markdown cell immediately above the matching plot code cell; when a
caption is missing or blank, the plot stands on its own.
"""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.eda import EDAResult
from src.analysis.agents.notebook.sections.eda import render_eda_report


def _state(eda: EDAResult | None) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
    )
    state.treatment_variable = "treat"
    state.outcome_variable = "re78"
    state.eda_result = eda
    return state


def _cell_sources(cells: list, kind: str) -> list[str]:
    return [c["source"] for c in cells if c["cell_type"] == kind]


def _caption_above_code(cells: list, caption_text: str, code_substr: str) -> bool:
    """True iff a markdown cell containing `caption_text` is followed (eventually)
    by a code cell containing `code_substr`, with no intervening code cell."""
    in_window = False
    for c in cells:
        if c["cell_type"] == "markdown" and caption_text in c["source"]:
            in_window = True
            continue
        if c["cell_type"] == "code":
            if in_window and code_substr in c["source"]:
                return True
            in_window = False
    return False


# --- no captions: section still renders, no caption cells -------------------


def test_renders_without_captions_when_eda_result_has_none():
    """Back-compat: empty plot_captions emits no extra markdown cells."""
    eda = EDAResult(
        data_quality_score=80.0,
        correlation_matrix={"a": {"a": 1.0, "b": 0.4}, "b": {"a": 0.4, "b": 1.0}},
        covariate_balance={"age": {"smd": 0.05, "is_balanced": True}},
    )
    cells = render_eda_report(_state(eda))
    md = "\n".join(_cell_sources(cells, "markdown"))
    assert "*" not in md or "*Findings" in md
    assert "treat" in "\n".join(_cell_sources(cells, "code"))


# --- with captions: each caption renders above its plot ---------------------


def test_distribution_caption_renders_above_distribution_plot():
    eda = EDAResult(
        plot_captions={"distribution": "Treatment is binary with 30% prevalence."}
    )
    cells = render_eda_report(_state(eda))
    assert _caption_above_code(
        cells,
        "Treatment is binary with 30% prevalence.",
        "Distribution plots for treatment and outcome",
    )


def test_outcome_by_group_caption_renders_above_group_plot():
    eda = EDAResult(
        plot_captions={"outcome_by_group": "Treated group has higher median."}
    )
    cells = render_eda_report(_state(eda))
    assert _caption_above_code(
        cells,
        "Treated group has higher median.",
        "Outcome by treatment group",
    )


def test_correlation_caption_renders_above_heatmap():
    eda = EDAResult(
        correlation_matrix={"a": {"a": 1.0, "b": 0.4}, "b": {"a": 0.4, "b": 1.0}},
        plot_captions={"correlation_heatmap": "Highest |r| is 0.42 (a, b)."},
    )
    cells = render_eda_report(_state(eda))
    assert _caption_above_code(
        cells,
        "Highest |r| is 0.42 (a, b).",
        "Correlation matrix from pipeline",
    )


def test_love_plot_caption_renders_above_love_plot():
    eda = EDAResult(
        covariate_balance={
            "age": {"smd": 0.25, "is_balanced": False},
            "educ": {"smd": 0.05, "is_balanced": True},
        },
        plot_captions={"love_plot": "Age is imbalanced at SMD 0.25; educ is fine."},
    )
    cells = render_eda_report(_state(eda))
    assert _caption_above_code(
        cells,
        "Age is imbalanced at SMD 0.25; educ is fine.",
        "Love plot: Standardized Mean Differences",
    )


# --- blank captions are ignored --------------------------------------------


def test_blank_caption_is_not_rendered():
    """A whitespace-only caption value emits no markdown cell."""
    eda = EDAResult(plot_captions={"distribution": "   "})
    cells = render_eda_report(_state(eda))
    md = "\n".join(_cell_sources(cells, "markdown"))
    # The whitespace caption must not appear wrapped in italics.
    assert "*   *" not in md


# --- missing key is silently skipped ----------------------------------------


def test_partial_captions_only_render_for_provided_keys():
    """Providing only one caption renders one caption cell, no placeholders."""
    eda = EDAResult(
        correlation_matrix={"a": {"a": 1.0, "b": 0.4}, "b": {"a": 0.4, "b": 1.0}},
        covariate_balance={"age": {"smd": 0.2, "is_balanced": False}},
        plot_captions={"love_plot": "Age imbalanced."},
    )
    cells = render_eda_report(_state(eda))
    md_blocks = _cell_sources(cells, "markdown")
    italics = [s for s in md_blocks if s.startswith("*") and s.endswith("*")]
    assert any("Age imbalanced." in s for s in italics)
    assert len(italics) == 1
