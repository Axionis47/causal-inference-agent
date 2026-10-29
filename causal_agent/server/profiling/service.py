"""Generate, atomically publish and independently retrieve owned profile facts."""

import logging
from tempfile import TemporaryFile

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from causal_agent.profile import inspection
from causal_agent.profile.report import Check, ColumnResult, ColumnSummary, ComputedProfile, DatasetResult, ProfileView, SavedColumn, SavedProfile
from causal_agent.server.ingestion.lookup import owned
from causal_agent.server.ingestion.service import Missing, uid
from causal_agent.server.ingestion.validation import validate
from causal_agent.server.profiling import schema as S

log = logging.getLogger(__name__)


def matching(owner, info):
    return (
        S.profiles.c.owner_id == owner,
        S.profiles.c.version_id == info["version_id"],
        S.profiles.c.source_checksum == info["sha256"],
        S.profiles.c.profiler_version == inspection.VERSION,
        S.profiles.c.settings_fingerprint == inspection.FINGERPRINT,
    )


def view(storage, owner, dataset_id):
    info = owned(storage, owner, dataset_id)
    with storage.engine.connect().execution_options(isolation_level="REPEATABLE READ") as db:
        row = db.execute(select(S.profiles).where(*matching(owner, info))).mappings().first()
        if row is None:
            return ProfileView(status="not_profiled")
        cols = (
            db.execute(
                select(
                    S.columns.c.id.label("column_id"),
                    S.columns.c.position,
                    S.columns.c.name,
                    S.columns.c.key,
                    S.column_results.c.status,
                    S.column_results.c.result["suggested_dtype"].astext.label("suggested_dtype"),
                    func.jsonb_array_length(S.column_results.c.result["issues"]).label("issue_count"),
                )
                .select_from(S.columns.join(S.column_results, S.column_results.c.column_id == S.columns.c.id))
                .where(S.column_results.c.profile_id == row["id"], S.column_results.c.owner_id == owner)
                .order_by(S.columns.c.position)
            )
            .mappings()
            .all()
        )
        profile = SavedProfile(
            **{k: v for k, v in row.items() if k not in {"id", "owner_id"}},
            profile_id=row["id"],
            dataset_id=dataset_id,
            columns=[ColumnSummary.model_validate(dict(c)) for c in cols],
        )
    return ProfileView(status=profile.status, profile=profile)


def column(storage, owner, dataset_id, profile_id, column_id):
    info = owned(storage, owner, dataset_id)
    with storage.engine.connect() as db:
        row = (
            db.execute(
                select(S.column_results.c.result, S.profiles.c.source_checksum)
                .select_from(S.column_results.join(S.profiles, S.profiles.c.id == S.column_results.c.profile_id))
                .where(
                    S.column_results.c.profile_id == profile_id,
                    S.column_results.c.column_id == column_id,
                    S.column_results.c.owner_id == owner,
                    S.column_results.c.version_id == info["version_id"],
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        raise Missing("Dataset not found.")
    return SavedColumn(
        profile_id=profile_id,
        version_id=info["version_id"],
        column_id=column_id,
        source_checksum=row["source_checksum"],
        result=ColumnResult.model_validate(row["result"]),
    )


def publish(db, owner, info, result):
    """One transaction includes the dataset summary, column identities and results."""
    existing = db.execute(select(S.profiles).where(*matching(owner, info))).mappings().first()
    profile_id = existing["id"] if existing else uid()
    values = dict(status=result.status, dataset=result.dataset.model_dump(mode="json"), updated_at=func.now())
    if existing:
        db.execute(update(S.profiles).where(S.profiles.c.id == profile_id).values(**values))
        db.execute(delete(S.column_results).where(S.column_results.c.profile_id == profile_id))
    else:
        db.execute(
            S.profiles.insert().values(
                id=profile_id,
                owner_id=owner,
                version_id=info["version_id"],
                source_checksum=info["sha256"],
                profiler_version=inspection.VERSION,
                settings_fingerprint=inspection.FINGERPRINT,
                **values,
            )
        )
    for col in result.columns:
        column_id = db.scalar(select(S.columns.c.id).where(S.columns.c.version_id == info["version_id"], S.columns.c.position == col.position))
        if column_id is None:
            db.execute(
                insert(S.columns)
                .values(id=uid(), owner_id=owner, version_id=info["version_id"], position=col.position, name=col.name, key=col.key)
                .on_conflict_do_nothing(index_elements=["version_id", "position"])
            )
            column_id = db.scalar(select(S.columns.c.id).where(S.columns.c.version_id == info["version_id"], S.columns.c.position == col.position))
        db.execute(
            S.column_results.insert().values(
                profile_id=profile_id, column_id=column_id, owner_id=owner, version_id=info["version_id"], status=col.status, result=col.model_dump(mode="json")
            )
        )


def generate(storage, owner, dataset_id):
    info = owned(storage, owner, dataset_id)
    lock_key = f"profile:{info['version_id']}:{inspection.VERSION}:{inspection.FINGERPRINT}"
    with storage.locked(owner, lock_key) as db:
        current = view(storage, owner, dataset_id)
        if current.status == "complete":
            return current
        # Infrastructure errors propagate as 503 and preserve the previous report.
        with TemporaryFile(mode="w+b") as temp:
            body = storage.objects.open(info["object_key"])
            try:
                size = 0
                for chunk in body.iter_chunks(64 * 1024):
                    size += len(chunk)
                    if size > min(storage.settings.max_bytes, info["size_bytes"]):
                        raise OSError("Source exceeds declared size")
                    temp.write(chunk)
            finally:
                body.close()
            digest, size = validate(temp, info["filename"], storage.settings)
            if size != info["size_bytes"] or digest != info["sha256"]:
                raise OSError("Source checksum mismatch")
            try:
                result = inspection.inspect(temp)
            except Exception:
                log.exception("Profile computation failed")
                result = ComputedProfile(
                    status="failed",
                    dataset=DatasetResult(
                        checks=[Check(name="source", status="failed", reason="Could not inspect the verified CSV.")],
                        issues=[inspection.issue("source", "inspection_failed", "Could not inspect the verified CSV.", None)],
                    ),
                )
        # Retrying cannot destroy useful results with a total computation failure.
        if result.status == "failed" and current.profile is not None:
            raise OSError("Retry failed; previous report retained")
        publish(db, owner, info, result)
        db.commit()
    return view(storage, owner, dataset_id)
