"""ReportNotebookAgent (S10): report, dashboard payload, and the notebook.

Reads only existing artifacts and run-state slots; computes nothing new
and changes no estimate. The notebook config records the relative
dataset path and the backend dir so execution needs no web-app state.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
import structlog

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.spec import NotebookBuildResult
from src.storage.job_data import job_normalized_dir
from src.storage.job_data import read_manifest

from .notebook import SECTIONS, build_notebook
from .report import build_dashboard_payload, build_report_markdown

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

        report_md = build_report_markdown(run)
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
        notebook = build_notebook(run)
        _add("notebook/causal_analysis", ArtifactKind.NOTEBOOK,
             "Causal analysis notebook", "notebook/causal_analysis.ipynb",
             nbformat.writes(notebook))

        output = NotebookBuildResult(
            notebook_artifact_id="notebook/causal_analysis",
            report_artifact_id="report/final_report",
            dashboard_artifact_id="report/dashboard_payload",
            sections=list(SECTIONS),
            referenced_artifact_ids=[a.artifact_id for a in run.artifact_registry.artifacts],
        )
        return AgentResult(
            gate=GateResult.advance(),
            output=output,
            public_summary=(
                f"Report written with claim strength "
                f"{run.claim_critique.strength.value}; the {len(SECTIONS)}-section "
                "notebook is built and awaits execution verification."
            ),
            artifact_ids=ids,
        )

    @staticmethod
    def _notebook_config(run: AnalysisRunState) -> dict:
        """Relative dataset path + backend dir; cwd at execution = analysis dir."""
        backend_dir = str(Path(__file__).resolve().parents[5])
        dataset_rel = None
        manifest = read_manifest(run.job_id)
        if manifest is not None and manifest.winner is not None:
            entry = next(
                (f for f in manifest.files if f.name == manifest.winner), None
            )
            if entry is not None and entry.normalized_path:
                name = Path(entry.normalized_path).name
                if (job_normalized_dir(run.job_id) / name).exists():
                    dataset_rel = f"../normalized/{name}"
        return {
            "job_id": run.job_id,
            "dataset_path": dataset_rel,
            "ignored_columns": run.ignored_columns,
            "backend_dir": backend_dir,
            "question": run.causal_question,
        }

    def commit(self, run: AnalysisRunState, output: NotebookBuildResult) -> None:
        run.notebook_build = output
