"""Artifact registry: ids are unique, paths stay inside the analysis dir."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis_v2.core import (
    AnalysisStage,
    Artifact,
    ArtifactKind,
    ArtifactRegistry,
    default_media_type,
)


def _plot(artifact_id: str = "eda/outcome_hist", agent: str = "targeted_eda") -> Artifact:
    return Artifact(
        artifact_id=artifact_id,
        kind=ArtifactKind.PLOT,
        stage=AnalysisStage.S4_TARGETED_EDA_COMPLETE,
        agent=agent,
        title="Outcome distribution",
        path="eda/outcome_hist.png",
        media_type="image/png",
    )


def test_registry_rejects_duplicate_artifact_ids():
    reg = ArtifactRegistry()
    reg.register(_plot())
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(_plot())
    assert len(reg.artifacts) == 1


def test_artifact_path_rejects_absolute_and_traversal_paths():
    for bad in ["/etc/passwd", "..", "a/../../b", "..\\windows", "plots/../../x.png"]:
        with pytest.raises(ValidationError):
            Artifact.model_validate({**_plot().model_dump(), "path": bad})


def test_registry_filters_by_stage_and_agent():
    reg = ArtifactRegistry()
    reg.register(_plot("eda/a"))
    reg.register(_plot("eda/b"))
    reg.register(
        Artifact(
            artifact_id="intake/spec",
            kind=ArtifactKind.JSON,
            stage=AnalysisStage.S1_INTAKE_PARSED,
            agent="intake",
            title="Causal spec draft",
            path="intake/causal_spec.json",
            media_type="application/json",
        )
    )
    assert [a.artifact_id for a in reg.by_agent("targeted_eda")] == ["eda/a", "eda/b"]
    assert [a.artifact_id for a in reg.by_stage(AnalysisStage.S1_INTAKE_PARSED)] == [
        "intake/spec"
    ]
    assert reg.get("missing") is None
    assert reg.ids() == ["eda/a", "eda/b", "intake/spec"]


def test_every_artifact_kind_has_a_default_media_type():
    for kind in ArtifactKind:
        assert "/" in default_media_type(kind)
