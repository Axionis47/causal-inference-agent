"""Artifacts: every displayable output an agent emits, with its registry.

An artifact is a file under {LOCAL_STORAGE_PATH}/{job_id}/analysis/ plus
one registry entry on the run state. The registry is what the frontend
tiles list and what the report/notebook stage indexes. Paths are stored
relative to the job's analysis dir so records survive a data-dir move.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from .stages import AnalysisStage


class ArtifactKind(StrEnum):
    JSON = "json"
    TABLE = "table"  # csv or parquet
    PLOT = "plot"  # png
    MARKDOWN = "markdown"
    NOTEBOOK = "notebook"
    HTML = "html"


_MEDIA_TYPES: dict[ArtifactKind, str] = {
    ArtifactKind.JSON: "application/json",
    ArtifactKind.TABLE: "text/csv",
    ArtifactKind.PLOT: "image/png",
    ArtifactKind.MARKDOWN: "text/markdown",
    ArtifactKind.NOTEBOOK: "application/x-ipynb+json",
    ArtifactKind.HTML: "text/html",
}


def default_media_type(kind: ArtifactKind) -> str:
    return _MEDIA_TYPES[kind]


class Artifact(BaseModel):
    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    stage: AnalysisStage
    agent: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    # Relative to the job's analysis dir; never absolute, never escaping.
    path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    summary: str | None = Field(default=None, max_length=500)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("path")
    @classmethod
    def _relative_and_contained(cls, v: str) -> str:
        if v.startswith("/") or v.startswith("\\"):
            raise ValueError("artifact path must be relative to the analysis dir")
        parts = v.replace("\\", "/").split("/")
        if ".." in parts:
            raise ValueError("artifact path must not traverse upward")
        return v


class ArtifactRegistry(BaseModel):
    artifacts: list[Artifact] = Field(default_factory=list)

    def register(self, artifact: Artifact) -> Artifact:
        if self.get(artifact.artifact_id) is not None:
            raise ValueError(f"duplicate artifact_id {artifact.artifact_id!r}")
        self.artifacts.append(artifact)
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        for a in self.artifacts:
            if a.artifact_id == artifact_id:
                return a
        return None

    def by_stage(self, stage: AnalysisStage) -> list[Artifact]:
        return [a for a in self.artifacts if a.stage == stage]

    def by_agent(self, agent: str) -> list[Artifact]:
        return [a for a in self.artifacts if a.agent == agent]

    def ids(self) -> list[str]:
        return [a.artifact_id for a in self.artifacts]
