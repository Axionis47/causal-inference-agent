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
from src.agents.specialists.notebook.sections.confounder_analysis import (
    render_confounder_analysis,
)
from src.agents.specialists.notebook.sections.critique import render_critique_section
from src.agents.specialists.notebook.sections.data_profile import (
    render_data_profile_report,
)
from src.agents.specialists.notebook.sections.data_repairs import render_data_repairs
from src.agents.specialists.notebook.sections.decisions import render_decisions
from src.agents.specialists.notebook.sections.domain_knowledge import (
    render_domain_knowledge,
)
from src.agents.specialists.notebook.sections.eda import render_eda_report
from src.agents.specialists.notebook.sections.ps_diagnostics import (
    render_ps_diagnostics,
)
from src.agents.specialists.notebook.sections.sensitivity import render_sensitivity
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


class TestEverySectionEmitsSkipPlaceholderOnEmptyState:
    """Every renderer must produce a skip placeholder on an empty state.

    Centralising the skip logic in the renderers (rather than at the call
    site in agent.py) means the notebook outline is stable regardless of
    which agents ran. This test enforces that contract for each renderer.
    """

    def _assert_single_skip(self, cells, label: str) -> None:
        assert len(cells) == 1, f"{label}: expected 1 cell, got {len(cells)}"
        assert _is_skipped_cell(cells[0]), f"{label}: cell is not a skip placeholder"

    def test_data_profile(self):
        self._assert_single_skip(render_data_profile_report(_empty_state()), "data_profile")

    def test_data_repairs(self):
        self._assert_single_skip(render_data_repairs(_empty_state()), "data_repairs")

    def test_confounder_analysis(self):
        self._assert_single_skip(
            render_confounder_analysis(_empty_state()), "confounder_analysis"
        )

    def test_ps_diagnostics(self):
        self._assert_single_skip(render_ps_diagnostics(_empty_state()), "ps_diagnostics")

    def test_sensitivity(self):
        self._assert_single_skip(render_sensitivity(_empty_state()), "sensitivity")

    def test_decisions(self):
        self._assert_single_skip(render_decisions(_empty_state()), "decisions")

    def test_critique(self):
        self._assert_single_skip(render_critique_section(_empty_state()), "critique")

    def test_domain_knowledge(self):
        self._assert_single_skip(
            render_domain_knowledge(_empty_state()), "domain_knowledge"
        )
