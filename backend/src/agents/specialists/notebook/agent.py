"""Notebook Generator Agent — thin shell that orchestrates section renderers.

Decomposed from monolithic notebook_generator.py. Each section is now
a standalone renderer function in the sections/ subpackage.
"""

import time

from nbformat.v4 import new_notebook

from src.agents.base import AnalysisState, BaseAgent, JobStatus
from src.agents.registry import register_agent
from src.logging_config.structured import get_logger

from .helpers import save_notebook
from .sections import (
    render_causal_structure,
    render_conclusions,
    render_confounder_analysis,
    render_critique_section,
    render_data_loading,
    render_data_profile_report,
    render_data_repairs,
    render_domain_knowledge,
    render_eda_report,
    render_introduction,
    render_ps_diagnostics,
    render_sensitivity,
    render_setup_cells,
    render_treatment_effects,
)

logger = get_logger(__name__)


@register_agent("notebook_generator")
class NotebookGeneratorAgent(BaseAgent):
    """Agent that generates reproducible Jupyter notebooks reporting pipeline results.

    Each notebook section maps 1:1 to a pipeline agent and presents
    that agent's actual outputs, reasoning, and findings.
    """

    AGENT_NAME = "notebook_generator"

    SYSTEM_PROMPT = """You are an expert at creating clear, reproducible Jupyter notebooks
for causal inference analysis.

When generating narratives:
1. Be specific to the dataset - never write generic placeholder text
2. Explain the choice of methods in context of the data characteristics
3. Present results with proper interpretation and caveats
4. Discuss limitations honestly
5. Write in clear academic style accessible to data scientists"""

    TOOLS = []  # No tools needed - uses LLM for narrative only

    REQUIRED_STATE_FIELDS = ["data_profile"]
    WRITES_STATE_FIELDS = ["notebook_path"]
    JOB_STATUS = "generating_notebook"
    PROGRESS_WEIGHT = 95

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Generate the analysis notebook reporting all pipeline findings."""
        self.logger.info("notebook_generation_start", job_id=state.job_id)
        state.status = JobStatus.GENERATING_NOTEBOOK
        start_time = time.time()

        try:
            nb = new_notebook()
            nb.metadata = {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            }

            cells: list = []

            # 1. Title & Introduction (async — uses LLM)
            cells.extend(await render_introduction(
                state, llm=self.llm, system_prompt=self.SYSTEM_PROMPT
            ))

            # 2. Domain Knowledge (if agent ran)
            if state.domain_knowledge:
                cells.extend(render_domain_knowledge(state))

            # 3. Setup & Imports
            cells.extend(render_setup_cells())

            # 4. Data Loading
            cells.extend(render_data_loading(state))

            # 5. Data Profile (profiler agent findings)
            if state.data_profile:
                cells.extend(render_data_profile_report(state))

            # 6. Data Repairs (data repair agent)
            if state.data_repairs:
                cells.extend(render_data_repairs(state))

            # 7. EDA (EDA agent findings)
            cells.extend(render_eda_report(state))

            # 8. Causal Structure (discovery agent)
            if state.proposed_dag:
                cells.extend(render_causal_structure(state))

            # 9. Confounder Analysis
            if state.confounder_discovery or (
                state.data_profile and state.data_profile.potential_confounders
            ):
                cells.extend(render_confounder_analysis(state))

            # 10. PS Diagnostics (propensity score diagnostics agent)
            if state.ps_diagnostics:
                cells.extend(render_ps_diagnostics(state))

            # 11. Treatment Effect Estimation (effect estimator)
            cells.extend(render_treatment_effects(state))

            # 12. Sensitivity Analysis (sensitivity analyst)
            if state.sensitivity_results:
                cells.extend(render_sensitivity(state))

            # 13. Analysis Quality & Critique
            if state.critique_history:
                cells.extend(render_critique_section(state))

            # 14. Conclusions (async — uses LLM)
            cells.extend(await render_conclusions(
                state, llm=self.llm, system_prompt=self.SYSTEM_PROMPT
            ))

            nb.cells = cells
            notebook_path = save_notebook(nb, state.job_id)
            state.notebook_path = notebook_path

            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="notebook_generated",
                reasoning="Generated pipeline report notebook",
                outputs={"path": notebook_path, "n_cells": len(cells)},
                duration_ms=duration_ms,
            )
            state.add_trace(trace)
            self.logger.info(
                "notebook_generation_complete",
                path=notebook_path,
                n_cells=len(cells),
            )

        except Exception as e:
            self.logger.error("notebook_generation_failed", error=str(e))
            state.mark_failed(f"Notebook generation failed: {str(e)}", self.AGENT_NAME)

        return state
