"""Bounded list projections. Full evidence is loaded through the owned catalogue."""

from causal_agent.memory.artifacts import Artifact
from causal_agent.server.models import ArtifactSummary

MEDIA = {".png": "image/png", ".csv": "text/csv", ".json": "application/json", ".txt": "text/plain", ".py": "text/plain"}


def view(name: str, artifact: Artifact) -> ArtifactSummary:
    return ArtifactSummary(
        id=artifact.id,
        ask=artifact.ask,
        origin=artifact.origin,
        format=artifact.format,
        summary=artifact.summary,
        conversation_id=artifact.conversation_id,
        dataset_version_id=artifact.dataset_version_id,
        design=artifact.design,
        made_at=artifact.made_at,
        url=f"/api/conversations/{artifact.conversation_id}/artifacts/{artifact.id}/files/figure.png" if artifact.figure else None,
    )
