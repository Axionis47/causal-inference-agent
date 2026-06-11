"""NotebookVerificationAgent (S11): execute the notebook or say why not.

Deterministic executor with a bounded repair loop (max 3 attempts).
Repairs may only fix reproducibility plumbing (a wrong backend dir or
dataset path in the config); conclusions and estimates are never
touched. The download card unlocks only on verified_running.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
import structlog

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, ArtifactKind, GateResult
from src.analysis_v2.persistence import analysis_dir
from src.analysis_v2.spec import NotebookStatus, NotebookVerificationResult
from src.storage.job_data import job_normalized_dir

logger = structlog.get_logger(__name__)

MAX_ATTEMPTS = 3
CELL_TIMEOUT_SECONDS = 300


class NotebookVerificationAgent(AnalysisAgent):
    name = "notebook_verification"
    stage = AnalysisStage.S11_NOTEBOOK_VERIFIED

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        run = ctx.run
        build = run.notebook_build
        if build is None:
            return AgentResult(
                gate=GateResult.fail(["nothing to verify: the notebook was not built"]),
                public_summary="The report stage must build the notebook first.",
            )
        base = analysis_dir(ctx.job_id)
        nb_path = base / "notebook" / "causal_analysis.ipynb"
        config_path = base / "notebook" / "notebook_config.json"

        attempts: list[dict] = []
        executed = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            notebook = nbformat.read(nb_path, as_version=4)
            error = await self._execute(notebook, base)
            if error is None:
                executed = notebook
                attempts.append({"attempt": attempt, "status": "passed"})
                break
            attempts.append({"attempt": attempt, "status": "failed", "error": error[:1500]})
            repair = self._repair(error, config_path, ctx)
            if repair is None:
                break
            attempts[-1]["repair"] = repair

        ids: list[str] = []

        def _add(artifact_id, kind, title, path, payload):
            ctx.add_artifact(
                agent=self.name, stage=self.stage, artifact_id=artifact_id,
                kind=kind, title=title, relative_path=path, payload=payload,
            )
            ids.append(artifact_id)

        _add("notebook/execution_log", ArtifactKind.JSON, "Notebook execution log",
             "notebook/notebook_execution_log.json", {"attempts": attempts})

        if executed is None:
            errors = [a.get("error", "unknown") for a in attempts if a["status"] == "failed"]
            output = NotebookVerificationResult(
                notebook_status=NotebookStatus.FAILED,
                executed_all_cells=False,
                errors=errors,
                attempts=len(attempts),
                repairs=[a["repair"] for a in attempts if "repair" in a],
                execution_log_artifact_id="notebook/execution_log",
            )
            return AgentResult(
                gate=GateResult.fail(
                    [f"notebook failed execution after {len(attempts)} attempts: "
                     f"{errors[-1][:200]}"]
                ),
                output=output,
                public_summary="The notebook did not execute cleanly; the download "
                "stays locked. See the execution log.",
                artifact_ids=ids,
            )

        _add("notebook/verified", ArtifactKind.NOTEBOOK, "Verified notebook",
             "notebook/causal_analysis_verified.ipynb", nbformat.writes(executed))
        html = self._html_preview(executed)
        if html is not None:
            _add("notebook/html_preview", ArtifactKind.HTML, "Notebook preview",
                 "notebook/notebook_preview.html", html)

        output = NotebookVerificationResult(
            notebook_status=NotebookStatus.VERIFIED_RUNNING,
            executed_all_cells=True,
            errors=[],
            attempts=len(attempts),
            repairs=[a["repair"] for a in attempts if "repair" in a],
            verified_notebook_artifact_id="notebook/verified",
            execution_log_artifact_id="notebook/execution_log",
            html_preview_artifact_id="notebook/html_preview" if html else None,
        )
        return AgentResult(
            gate=GateResult.advance(),
            output=output,
            public_summary=(
                f"The notebook executed top to bottom on attempt {len(attempts)}; "
                "every cell ran, including the estimate re-verification. "
                "Download unlocked."
            ),
            artifact_ids=ids,
        )

    async def _execute(self, notebook, cwd: Path) -> str | None:
        """Run all cells; returns None on success or the error text."""
        import asyncio

        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError

        client = NotebookClient(
            notebook,
            timeout=CELL_TIMEOUT_SECONDS,
            kernel_name="python3",
            resources={"metadata": {"path": str(cwd)}},
        )

        def _run() -> str | None:
            try:
                client.execute()
                return None
            except CellExecutionError as exc:
                return str(exc)
            except Exception as exc:  # kernel/setup failures
                return f"{type(exc).__name__}: {exc}"

        return await asyncio.to_thread(_run)

    def _repair(self, error: str, config_path: Path, ctx: AgentCtx) -> str | None:
        """Reproducibility-only repairs to the notebook config."""
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            return None
        if "ModuleNotFoundError" in error and "src" in error:
            fixed = str(Path(__file__).resolve().parents[5])
            if config.get("backend_dir") != fixed:
                config["backend_dir"] = fixed
                config_path.write_text(json.dumps(config, indent=2))
                return f"backend_dir corrected to {fixed}"
            return None
        if "FileNotFoundError" in error or "No such file" in error:
            normalized = job_normalized_dir(ctx.job_id)
            parquets = sorted(normalized.glob("*.parquet"))
            if parquets:
                absolute = str(parquets[0])
                if config.get("dataset_path") != absolute:
                    config["dataset_path"] = absolute
                    config_path.write_text(json.dumps(config, indent=2))
                    return f"dataset_path corrected to {absolute}"
        return None

    def _html_preview(self, executed) -> str | None:
        try:
            from nbconvert import HTMLExporter

            html, _ = HTMLExporter().from_notebook_node(executed)
            return html
        except Exception as exc:
            logger.warning("notebook_html_preview_failed", error=str(exc))
            return None

    def commit(self, run: AnalysisRunState, output: NotebookVerificationResult) -> None:
        run.notebook_verification = output
