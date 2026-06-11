"""Analysis-owned projections for a conversation's turn-bound discussion."""

from copy import deepcopy
from hashlib import sha256
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.analysis import artifacts, packets
from backend.context.service import fingerprint
from backend.handoffs import AnalysisBinding, AnalysisFile, AnalysisMaterial
from backend.models import AnalysisRun


def capture(db: Session, conversation_id: UUID) -> dict[str, AnalysisBinding]:
    runs = db.scalars(select(AnalysisRun).where(
        AnalysisRun.conversation_id == conversation_id,
        or_(AnalysisRun.result.is_not(None), AnalysisRun.snapshot.is_not(None),
            AnalysisRun.inputs["frozen_context"].astext.is_not(None)),
    ))
    return {str(run.id): artifacts.binding(run) for run in runs}


def views(db: Session, conversation_id: UUID, identities: list[UUID],
          bindings: dict[str, AnalysisBinding] | None) -> list[AnalysisMaterial]:
    result = []
    for run in db.scalars(select(AnalysisRun).where(
        AnalysisRun.conversation_id == conversation_id, AnalysisRun.id.in_(identities),
    )):
        # Historical turns selected only already completed analyses.
        bound = deepcopy(bindings[str(run.id)]) if bindings is not None else artifacts.binding(run)
        if bound["snapshot_hash"] is not None and bound["snapshot_hash"] != run.snapshot_hash:
            raise ValueError("The frozen analysis differs from this turn's binding.")
        scientific = packets.science({key: value for key, value in (run.snapshot or {}).items() if key != "program"}) if bound["snapshot_hash"] is not None else None
        computed = None
        if bound["result_hash"] is not None:
            if fingerprint(run.result) != bound["result_hash"]:
                raise ValueError("The saved result differs from this turn's binding.")
            computed = deepcopy(run.result)
        result.append(AnalysisMaterial(run.id, run.question, bound, scientific, computed))
    return result


def file(db: Session, conversation_id: UUID, material: AnalysisMaterial,
         name: str, *, expected_hash: str | None = None) -> AnalysisFile:
    run = db.scalar(select(AnalysisRun).where(
        AnalysisRun.id == material.id, AnalysisRun.conversation_id == conversation_id,
    ))
    if run is None or name not in material.binding["files"]:
        raise ValueError("Selected analysis artifact is outside the turn's bound analysis records.")
    raw = artifacts.read(run, name)
    digest = sha256(raw).hexdigest()
    if digest != material.binding["files"][name]:
        raise ValueError("Selected artifact differs from its turn-bound checksum.")
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("A bound analysis artifact changed.")
    return AnalysisFile(raw, digest, artifacts.numerical_facts(run, name))
