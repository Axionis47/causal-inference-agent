"""AgentCtx: everything an agent may touch during one dispatch.

Agents read the run state and the loaded frame, write artifacts through
`add_artifact` (which persists the file AND registers it, so nothing
displayable can exist unregistered), and emit SSE events through `emit`.
They do not construct storage clients, mutate run-state slots, or reach
around the context.
"""
from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.analysis_v2.core import (
    AnalysisRunState,
    AnalysisStage,
    Artifact,
    ArtifactKind,
    default_media_type,
)
from src.analysis_v2.persistence import write_artifact_bytes, write_artifact_json
from src.analysis_v2.state import AnalysisState


def _noop_emit(event_type: str, data: dict) -> None:  # pragma: no cover
    return None


@dataclass
class AgentCtx:
    job_id: str
    run: AnalysisRunState
    frame: pd.DataFrame
    # The picked-up input record; carries the two context channels
    # (user_provided_context authoritative, kaggle_description informative).
    input_state: AnalysisState | None = None
    emit: Callable[[str, dict], None] = field(default=_noop_emit)

    def add_artifact(
        self,
        *,
        agent: str,
        stage: AnalysisStage,
        artifact_id: str,
        kind: ArtifactKind,
        title: str,
        relative_path: str,
        payload: Any,
        summary: str | None = None,
    ) -> Artifact:
        """Persist one artifact file and register it on the run state.

        `payload` is bytes for binary kinds, str for text kinds, or a
        jsonable object for JSON.
        """
        if kind == ArtifactKind.JSON and not isinstance(payload, (bytes, str)):
            write_artifact_json(self.job_id, relative_path, payload)
        elif isinstance(payload, bytes):
            write_artifact_bytes(self.job_id, relative_path, payload)
        elif isinstance(payload, str):
            write_artifact_bytes(self.job_id, relative_path, payload.encode("utf-8"))
        else:
            raise TypeError(f"unsupported payload type {type(payload)!r} for {kind}")
        artifact = Artifact(
            artifact_id=artifact_id,
            kind=kind,
            stage=stage,
            agent=agent,
            title=title,
            path=relative_path,
            media_type=default_media_type(kind),
            summary=summary,
        )
        self.run.register_artifact(artifact)
        self.emit(
            "analysis_artifact_emitted",
            {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind.value,
                "title": artifact.title,
                "stage": stage.value,
                "headline": f"artifact: {title}",
            },
        )
        return artifact

    def add_table_artifact(
        self,
        *,
        agent: str,
        stage: AnalysisStage,
        artifact_id: str,
        title: str,
        relative_path: str,
        frame: pd.DataFrame,
        summary: str | None = None,
    ) -> Artifact:
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        return self.add_artifact(
            agent=agent,
            stage=stage,
            artifact_id=artifact_id,
            kind=ArtifactKind.TABLE,
            title=title,
            relative_path=relative_path,
            payload=buf.getvalue(),
            summary=summary,
        )
