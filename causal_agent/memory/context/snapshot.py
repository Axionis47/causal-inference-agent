"""An immutable local snapshot and a compact, integrity-checked lane reference."""

import hashlib
from pathlib import Path

from causal_agent.common import trace
from causal_agent.common.contracts import Handoff
from causal_agent.memory.context.contracts import Contract


class SnapshotReference(Contract):
    revision_id: str
    context_id: str
    sha256: str
    family: str


def publish(directory: Path, handoff: Handoff, revision_id: str) -> Path:
    content = handoff.model_dump_json().encode()
    reference = SnapshotReference(revision_id=revision_id, context_id=handoff.context_id, sha256=hashlib.sha256(content).hexdigest(), family=handoff.family)
    body = directory / "snapshot.json"
    path = directory / "authorization.json"
    if body.exists() and body.read_bytes() != content:
        raise ValueError("An authorized snapshot cannot be replaced.")
    if path.exists() and path.read_text() != reference.model_dump_json():
        raise ValueError("An authorization cannot be replaced.")
    trace.write(body, content)
    trace.write(path, reference.model_dump_json())
    return path


def resolve(path: Path) -> Handoff:
    reference = SnapshotReference.model_validate_json(path.read_text())
    content = (path.parent / "snapshot.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise ValueError("Authorized snapshot integrity check failed.")
    handoff = Handoff.model_validate_json(content)
    if handoff.context_id != reference.context_id or handoff.family != reference.family:
        raise ValueError("Authorized snapshot identity changed.")
    return handoff
