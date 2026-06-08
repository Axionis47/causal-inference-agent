"""Shared helpers for notebook generation.

Contains LLM narrative generation, deduplication utilities,
and notebook save logic used across section renderers.
"""

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nbformat
import numpy as np

from src.logging_config.structured import get_logger

if TYPE_CHECKING:
    from src.analysis.agents.base import AnalysisState

logger = get_logger(__name__)


def notebook_data_source(state: "AnalysisState") -> tuple[str | None, bool]:
    """Pick which dataframe the notebook bundles and loads. Returns (path, is_raw).

    When data_repair applied repairs and saved a raw snapshot, the notebook loads
    the RAW data and the repairs section reproduces the cleaning on it. Otherwise
    it loads the final dataframe directly (no repairs means raw == final, nothing
    to reproduce). Both the notebook agent (which bundles the file) and the
    data-loading renderer (which writes the load cell) call this, so they never
    disagree on which file ships beside the notebook.
    """
    raw = getattr(state, "raw_dataframe_path", None)
    if state.data_repairs and raw and Path(raw).exists():
        return raw, True
    final = state.dataframe_path or (
        state.dataset_info.local_path if state.dataset_info else None
    )
    return final, False


async def generate_llm_narrative(
    llm, system_prompt: str, section: str, context: dict[str, Any]
) -> str:
    """Generate LLM-driven narrative for a notebook section.

    Falls back to empty string on any failure so notebook generation
    never breaks due to LLM issues.

    Args:
        llm: The LLM client instance.
        system_prompt: System prompt for the LLM.
        section: Name of the section being generated.
        context: Key-value pairs of context for the LLM.
    """
    context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
    prompt = f"""Generate a clear, concise narrative for the "{section}" section
of a causal inference analysis notebook.

Context:
{context_str}

Write in clear academic style. Be specific to this dataset.
Output markdown only, no code fences. 2-3 paragraphs maximum."""

    try:
        response = await llm.generate(
            prompt=prompt,
            system_instruction=system_prompt,
        )
        text = response.text
        # Strip leading markdown headings — the caller adds its own
        lines = text.lstrip().split("\n")
        while lines and lines[0].startswith("#"):
            lines.pop(0)
        return "\n".join(lines).strip()
    except Exception as e:
        logger.warning("llm_narrative_failed", section=section, error=str(e))
        return ""


def deduplicate_effects(effects: list) -> list:
    """Deduplicate treatment effects by (method, treatment_variable, outcome_variable).

    The TreatmentEffectResult schema fields are *_variable; the previous
    `getattr(effect, "treatment", "")` masked the rename by silently
    falling back to the empty string, which collapsed every effect with
    the same method to the same key and dropped rows.
    """
    seen: dict[tuple, Any] = {}
    for effect in effects:
        key = (
            effect.method.lower().strip(),
            effect.treatment_variable or "",
            effect.outcome_variable or "",
        )
        seen[key] = effect
    return list(seen.values())


def deduplicate_sensitivity(results: list) -> list:
    """Deduplicate sensitivity results by method name, keeping last occurrence."""
    seen: dict[str, Any] = {}
    for result in results:
        seen[result.method] = result
    return list(seen.values())


def save_notebook(
    nb: nbformat.NotebookNode,
    job_id: str,
    data_source_path: str | None = None,
) -> str:
    """Save the notebook and bundle data alongside it.

    Args:
        nb: The notebook object to save.
        job_id: Job identifier for filename.
        data_source_path: Path to the dataset file (parquet/csv) to bundle.

    Returns:
        Path to the saved notebook file.
    """
    output_dir = Path(tempfile.gettempdir()) / "causal_orchestrator" / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"causal_analysis_{job_id}.ipynb"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    # Bundle data file alongside notebook for reproducibility
    if data_source_path:
        src = Path(data_source_path)
        if src.exists():
            ext = src.suffix or ".parquet"
            data_dest = output_dir / f"data_{job_id}{ext}"
            try:
                shutil.copy2(str(src), str(data_dest))
                logger.info("notebook_data_bundled", src=str(src), dest=str(data_dest))
            except Exception as e:
                logger.warning("notebook_data_bundle_failed", error=str(e))

    return str(filepath)


async def save_notebook_async(
    nb: nbformat.NotebookNode,
    job_id: str,
    data_source_path: str | None = None,
) -> str:
    """Save the notebook to disk without blocking the event loop."""
    import asyncio

    return await asyncio.to_thread(save_notebook, nb, job_id, data_source_path)
