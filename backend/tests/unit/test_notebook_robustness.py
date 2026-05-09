"""Tests that notebook section renderers degrade gracefully when state is missing.

The notebook generator is the user-facing artifact. If a stage of the
pipeline failed or was skipped, the corresponding section should appear
in the output with a clear placeholder, not crash and not silently emit
an empty header.
"""

from __future__ import annotations

from src.agents import AnalysisState, DatasetInfo, TreatmentEffectResult
from src.agents.specialists.notebook.sections._skip import render_skipped_cell
from src.agents.specialists.notebook.sections.causal_structure import (
    render_causal_structure,
)
from src.agents.specialists.notebook.sections.eda import render_eda_report
from src.agents.specialists.notebook.sections.treatment_effects import (
    render_treatment_effects,
)


def _empty_state() -> AnalysisState:
    return AnalysisState(
        job_id="t",
        dataset_info=DatasetInfo(url="u", name="n"),
        treatment_variable="treat",
        outcome_variable="outcome",
    )


def _is_skipped_cell(cell) -> bool:
    """Heuristic: the placeholder cell contains the literal 'Section skipped' marker."""
    if cell.get("cell_type") != "markdown":
        return False
    return "Section skipped" in cell.get("source", "")


class TestSkipHelper:
    """The shared placeholder helper produces well-formed markdown."""

    def test_renders_one_markdown_cell(self):
        cells = render_skipped_cell(
            "Test Section",
            reason="Because reasons.",
            upstream_agent="some_agent",
        )
        assert len(cells) == 1
        assert cells[0]["cell_type"] == "markdown"

    def test_includes_section_name_reason_and_agent(self):
        cells = render_skipped_cell(
            "Test Section", reason="Because reasons.", upstream_agent="some_agent"
        )
        text = cells[0]["source"]
        assert "Test Section" in text
        assert "Because reasons." in text
        assert "some_agent" in text
        assert "Section skipped" in text


class TestEdaSkippedWhenMissing:
    """render_eda_report emits a placeholder when state.eda_result is None."""

    def test_skip_placeholder(self):
        cells = render_eda_report(_empty_state())
        assert len(cells) == 1
        assert _is_skipped_cell(cells[0])
        assert "EDA agent" not in cells[0]["source"] or "did not" in cells[0]["source"].lower()


class TestTreatmentEffectsSkippedWhenEmpty:
    """render_treatment_effects emits a placeholder when no estimates exist."""

    def test_skip_placeholder_on_empty(self):
        cells = render_treatment_effects(_empty_state())
        assert len(cells) == 1
        assert _is_skipped_cell(cells[0])

    def test_renders_normally_when_effects_present(self):
        state = _empty_state()
        state.treatment_effects = [
            TreatmentEffectResult(
                method="OLS", estimand="ATE", estimate=1.5, std_error=0.2,
                ci_lower=1.1, ci_upper=1.9, p_value=0.001,
            )
        ]
        cells = render_treatment_effects(state)
        # Multiple cells expected (header, caveat, table, code, ...)
        assert len(cells) > 1
        assert not _is_skipped_cell(cells[0])


class TestCausalStructureSkippedWhenMissing:
    """render_causal_structure emits a placeholder when no DAG is set."""

    def test_skip_placeholder(self):
        cells = render_causal_structure(_empty_state())
        assert len(cells) == 1
        assert _is_skipped_cell(cells[0])
