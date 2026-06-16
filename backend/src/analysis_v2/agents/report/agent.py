"""ReportNotebookAgent (S10): report, dashboard payload, and the notebook.

When the LLM has tool support, a ReAct loop lets the model read the results
and write each section's connective narrative, guided by the causal DAG; the
prose is woven into the deterministic execution order (so the notebook still
runs and self-verifies) and the chosen sections are recorded for replay. With
no tool support the agent degrades to the deterministic build. Either path
reads only existing artifacts and run-state slots; it computes nothing new and
changes no estimate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat
import structlog
from nbformat.v4 import new_markdown_cell

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent, react_loop
from src.analysis_v2.core import (
    AnalysisRunState,
    AnalysisStage,
    ArtifactKind,
    GateResult,
    TokenUsage,
)
from src.analysis_v2.spec import NotebookBuildResult, ReportToolCall, ReportToolTrace
from src.llm import get_llm_client
from src.storage.job_data import job_normalized_dir, read_manifest

from .guard import forbidden_hit
from .notebook import SECTIONS, build_notebook
from .prompt import MAX_NARRATIVE_TOOLS, MAX_TURNS, SYSTEM_PROMPT, build_mission
from .report import build_dashboard_payload, build_report_markdown, headline_lines
from .study import Section, ordered_study
from .tools import ReportLedger, build_report_tools, collect_trace

logger = structlog.get_logger(__name__)

_REQUIRED_SLOTS = (
    "causal_spec", "dataset_profile", "method_plan",
    "estimate_result", "claim_critique",
)


class ReportNotebookAgent(AnalysisAgent):
    name = "report_notebook"
    stage = AnalysisStage.S10_REPORT_NOTEBOOK_CREATED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        missing = [s for s in _REQUIRED_SLOTS if getattr(run, s) is None]
        if missing:
            return AgentResult(
                gate=GateResult.fail([f"report needs upstream slots: {missing}"]),
                public_summary="Upstream results are missing; the spine is out of order.",
            )

        llm = get_llm_client()
        ledger, tools = build_report_tools(run)
        prose, exec_summary, trace, tokens, agentic = await self._narrate(
            ctx, run, llm, ledger, tools
        )

        ids: list[str] = []

        def _add(artifact_id, kind, title, path, payload, summary=None):
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id=artifact_id,
                kind=kind, title=title, relative_path=path, payload=payload,
                summary=summary,
            )
            ids.append(artifact_id)

        # artifact index first: the notebook's last section reads it
        _add(
            "report/artifacts_index", ArtifactKind.JSON, "Artifact index",
            "report/artifacts_index.json",
            {
                "artifacts": [
                    {"artifact_id": a.artifact_id, "kind": a.kind.value,
                     "agent": a.agent, "title": a.title, "path": a.path}
                    for a in run.artifact_registry.artifacts
                ]
            },
        )

        report_md = (
            self._render_report(run, prose, exec_summary)
            if agentic else build_report_markdown(run)
        )
        _add("report/final_report", ArtifactKind.MARKDOWN, "Final report",
             "report/final_report.md", report_md,
             summary=f"claim strength {run.claim_critique.strength.value}")
        _add("report/final_report_json", ArtifactKind.JSON, "Final report (json)",
             "report/final_report.json",
             {
                 "question": run.causal_question,
                 "claim_strength": run.claim_critique.strength.value,
                 "primary": run.estimate_result.primary.model_dump(mode="json"),
                 "limitations": run.claim_critique.limitations,
             })
        _add("report/dashboard_payload", ArtifactKind.JSON, "Dashboard payload",
             "report/dashboard_payload.json", build_dashboard_payload(run))

        config = self._notebook_config(run)
        _add("notebook/config", ArtifactKind.JSON, "Notebook config",
             "notebook/notebook_config.json", config)
        _add("notebook/analysis_inputs", ArtifactKind.JSON, "Analysis inputs",
             "notebook/analysis_inputs.json", self._analysis_inputs(run))
        assembled = self._assemble(run, prose, exec_summary) if agentic else None
        notebook = build_notebook(run, assembled)
        _add("notebook/causal_analysis", ArtifactKind.NOTEBOOK,
             "Causal analysis notebook", "notebook/causal_analysis.ipynb",
             nbformat.writes(notebook))

        sections = [s.title for s in assembled] if assembled is not None else list(SECTIONS)
        output = NotebookBuildResult(
            notebook_artifact_id="notebook/causal_analysis",
            report_artifact_id="report/final_report",
            dashboard_artifact_id="report/dashboard_payload",
            sections=sections,
            referenced_artifact_ids=[a.artifact_id for a in run.artifact_registry.artifacts],
            report_tool_trace=trace,
        )
        return AgentResult(
            gate=GateResult.advance(),
            output=output,
            public_summary=(
                f"Report written with claim strength "
                f"{run.claim_critique.strength.value}; the "
                f"{'agentic ' if agentic else ''}notebook is built and awaits "
                "execution verification."
            ),
            artifact_ids=ids,
            tokens=tokens,
        )

    async def _narrate(
        self, ctx: AgentCtx, run: AnalysisRunState, llm: Any,
        ledger: ReportLedger, tools: list,
    ) -> tuple[dict[str, str], str, ReportToolTrace | None, TokenUsage, bool]:
        """Run the agentic loop; fall back to deterministic on no tool support,
        a provider that cannot run tools, or a model that wrote nothing."""
        none_result: tuple[dict[str, str], str, ReportToolTrace | None, TokenUsage, bool] = (
            {}, "", None, TokenUsage(), False
        )
        if not tools or not hasattr(llm, "chat_with_tools"):
            return none_result
        try:
            loop = await react_loop(
                llm, prompt=build_mission(run, tools), system_instruction=SYSTEM_PROMPT,
                tools=tools, ctx=ctx, max_tool_calls=MAX_NARRATIVE_TOOLS, max_turns=MAX_TURNS,
            )
        except NotImplementedError:
            return none_result  # provider has no tool support
        except Exception as exc:
            # The narrative is the last mile; a provider/loop hiccup must degrade
            # to the deterministic report, never sink a completed analysis.
            logger.warning("report_loop_failed", job_id=run.job_id, error=str(exc))
            return none_result
        if not ledger.prose and not ledger.executive_summary:
            return ({}, "", None, loop.tokens, False)  # never ship a model-empty narrative
        return (
            ledger.prose, ledger.executive_summary,
            collect_trace(ledger, exhausted=loop.exhausted), loop.tokens, True,
        )

    def _assemble(
        self, run: AnalysisRunState, prose: dict[str, str], exec_summary: str
    ) -> list[Section]:
        """Weave the model prose into the deterministic spine: the executive
        summary leads, each narrated section's header carries the model prose,
        and every section's data cells (and the verification cell) are kept in
        their fixed execution order, so coverage and reproducibility hold."""
        sections: list[Section] = []
        if exec_summary:
            sections.append(Section(
                "summary", "Executive summary",
                [new_markdown_cell(f"## Executive summary\n\n{exec_summary}")],
            ))
        for s in ordered_study(run):
            if s.section_id in prose:
                header = new_markdown_cell(f"## {s.title}\n\n{prose[s.section_id]}")
                sections.append(Section(s.section_id, s.title, [header, *s.cells[1:]]))
            else:
                sections.append(s)
        return sections

    def _render_report(
        self, run: AnalysisRunState, prose: dict[str, str], exec_summary: str
    ) -> str:
        """The same narrative rendered to markdown, so report and notebook do
        not drift. Per-section prose is already guarded; a final check falls
        back to the deterministic report if any forbidden phrase slipped."""
        critique = run.claim_critique
        lines = ["# Causal analysis report", "", f"**Question.** {run.causal_question}", ""]
        if exec_summary:
            lines += [exec_summary, ""]
        lines += [*headline_lines(run), ""]
        for s in ordered_study(run):
            if s.section_id in prose:
                lines += [f"## {s.title}", "", prose[s.section_id], ""]
        lines += ["## Limitations", ""]
        lines += [f"- {limit}" for limit in critique.limitations]
        text = "\n".join(lines)
        if forbidden_hit(text, critique) is not None:
            return build_report_markdown(run)
        return text

    @staticmethod
    def _notebook_config(run: AnalysisRunState) -> dict:
        """Relative file paths + backend dir; cwd at execution = analysis dir.
        A non-trivial assembly plan emits the plan plus a name -> relative-path
        map the notebook replays through assemble_from; otherwise a single
        dataset_path, unchanged from before."""
        backend_dir = str(Path(__file__).resolve().parents[5])
        manifest = read_manifest(run.job_id)
        config: dict = {
            "job_id": run.job_id,
            "dataset_path": None,
            "ignored_columns": run.ignored_columns,
            "backend_dir": backend_dir,
            "question": run.causal_question,
        }
        if manifest is None:
            return config

        def _parquet_rel(name: str) -> str | None:
            entry = next((f for f in manifest.files if f.name == name), None)
            if entry is not None and entry.normalized_path:
                pname = Path(entry.normalized_path).name
                if (job_normalized_dir(run.job_id) / pname).exists():
                    return f"../normalized/{pname}"
            return None

        plan = run.assembly_plan
        if plan is not None and not plan.is_trivial:
            names = [
                plan.base_file, *plan.concat_files,
                *(j.right_file for j in plan.joins),
            ]
            config["assembly_plan"] = plan.model_dump(mode="json")
            config["assembly_paths"] = {
                n: (_parquet_rel(n) or f"../raw/{n}") for n in names
            }
        elif manifest.winner is not None:
            config["dataset_path"] = _parquet_rel(manifest.winner)
        return config

    @staticmethod
    def _analysis_inputs(run: AnalysisRunState) -> dict:
        """The frozen plan/spec/dag the notebook deserializes to recompute the
        estimate, diagnostics, and figures inline. Kept out of notebook_config
        so notebook_verify's repair, which only touches the config keys, never
        sees these blobs."""
        return {
            "plan": run.method_plan.model_dump(mode="json"),
            "spec": run.causal_spec.model_dump(mode="json"),
            "dag": run.causal_dag.model_dump(mode="json") if run.causal_dag else None,
        }

    def commit(self, run: AnalysisRunState, output: NotebookBuildResult) -> None:
        run.notebook_build = output
        # The replayable trace: the agentic path carries the model's ordered
        # tool calls; the deterministic path derives it from the sections that
        # rendered (mirrors targeted_eda's commit synthesizing from checks).
        run.report_tool_trace = output.report_tool_trace or ReportToolTrace(
            calls=[
                ReportToolCall(name=title, section_id=title, kind="section", status="ok")
                for title in output.sections
            ]
        )
