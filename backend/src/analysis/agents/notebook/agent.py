"""Notebook Generator Agent — thin shell that orchestrates section renderers.

Decomposed from monolithic notebook_generator.py. Each section is now
a standalone renderer function in the sections/ subpackage.
"""

import time

from nbformat.v4 import new_notebook

from src.analysis.agents.base import AnalysisState, BaseAgent, JobStatus
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from .brief import CAPABILITY as NB_CAPABILITY, build_brief, preflight
from .helpers import notebook_data_source, save_notebook_async
from .sections import (
    render_causal_structure,
    render_conclusions,
    render_confounder_analysis,
    render_critique_section,
    render_data_loading,
    render_decisions,
    render_data_profile_report,
    render_data_repairs,
    render_domain_knowledge,
    render_eda_report,
    render_introduction,
    render_pipeline_briefs,
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
    CAPABILITY = NB_CAPABILITY

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Generate the analysis notebook reporting all pipeline findings."""
        refusal = preflight(state)
        if refusal is not None:
            state.agent_briefs[refusal.agent] = refusal
            self.logger.info(
                "notebook_refused",
                flag=refusal.flags[0].value,
                headline=refusal.headline,
            )
            return state

        self.logger.info("notebook_generation_start", job_id=state.job_id)
        # Status set by orchestrator
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

            # Every section renders unconditionally. Each renderer emits a
            # `Section skipped` placeholder when its upstream stage produced
            # nothing, so the notebook's outline is stable regardless of
            # which agents ran. Centralising the skip logic in the renderers
            # rather than the call site means a section header can never be
            # silently dropped.
            cells.extend(await render_introduction(
                state, llm=self.llm, system_prompt=self.SYSTEM_PROMPT
            ))
            cells.extend(render_domain_knowledge(state))

            methods_used = [e.method for e in state.treatment_effects]
            cells.extend(render_setup_cells(
                methods_used=methods_used,
                has_dag=bool(state.proposed_dag),
            ))
            cells.extend(render_data_loading(state))
            cells.extend(render_data_profile_report(state))
            cells.extend(render_data_repairs(state))
            cells.extend(render_eda_report(state))
            cells.extend(render_causal_structure(state))
            cells.extend(render_confounder_analysis(state))
            cells.extend(render_ps_diagnostics(state))
            cells.extend(render_treatment_effects(state))
            cells.extend(render_sensitivity(state))
            cells.extend(render_decisions(state))
            cells.extend(render_pipeline_briefs(state))
            cells.extend(render_critique_section(state))
            cells.extend(await render_conclusions(
                state, llm=self.llm, system_prompt=self.SYSTEM_PROMPT
            ))

            nb.cells = cells
            # Bundle raw data when repairs ran so the cleaning section can
            # reproduce them; otherwise bundle the final dataframe directly.
            data_source, _ = notebook_data_source(state)
            notebook_path = await save_notebook_async(
                nb, state.job_id, data_source_path=data_source
            )
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
        finally:
            if "notebook_generator" not in state.agent_briefs:
                brief = build_brief(state)
                state.agent_briefs[brief.agent] = brief
                self.logger.info(
                    "notebook_brief",
                    status=brief.status,
                    flags=[f.value for f in brief.flags],
                )

        return state
