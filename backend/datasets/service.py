import csv
import hashlib
import io
from itertools import islice
from uuid import UUID, uuid5

from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from backend.contracts import DatasetList, DatasetView, Preview, VersionView
from backend.datasets.csv_data import inspect_csv
from backend.models import Dataset, DatasetVersion
from backend.storage import LocalFiles


def view(dataset: Dataset, version: DatasetVersion) -> DatasetView:
    return DatasetView(
        id=dataset.id,
        title=dataset.title,
        created_at=dataset.created_at,
        version=VersionView.model_validate(version),
    )


def owned(db: Session, owner: UUID, dataset_id: UUID) -> tuple[Dataset, DatasetVersion]:
    row = db.execute(
        select(Dataset, DatasetVersion)
        .join(DatasetVersion)
        .where(
            Dataset.id == dataset_id,
            Dataset.owner_id == owner,
            DatasetVersion.number == 1,
        )
    ).first()
    if row is None:
        raise HTTPException(404, "Dataset not found.")
    return row[0], row[1]


def list_owned(db: Session, owner: UUID, page: int, page_size: int, search: str) -> DatasetList:
    condition = [Dataset.owner_id == owner]
    if search:
        condition.append(Dataset.title.icontains(search, autoescape=True))
    total = db.scalar(select(func.count()).select_from(Dataset).where(*condition)) or 0
    rows = db.execute(
        select(Dataset, DatasetVersion)
        .join(DatasetVersion)
        .where(
            *condition,
            DatasetVersion.number == 1,
        )
        .order_by(Dataset.created_at.desc(), Dataset.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return DatasetList(items=[view(d, v) for d, v in rows], total=total, page=page, page_size=page_size)


def upload(
    db: Session, storage: LocalFiles, owner: UUID, submission: UUID, title: str, filename: str, content: bytes
) -> DatasetView:
    info = inspect_csv(content, filename)
    checksum = hashlib.sha256(content).hexdigest()
    dataset_id = uuid5(owner, str(submission))
    # Transaction-scoped locking serializes retries without an independent operation state machine.
    lock = int.from_bytes(dataset_id.bytes[:8], "big", signed=True)
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock})
    existing = db.get(Dataset, dataset_id)
    if existing:
        dataset, version = owned(db, owner, dataset_id)
        if (dataset.title, version.filename, version.sha256) != (title, filename, checksum):
            raise HTTPException(
                409, "This submission was already saved with different content. Start a new upload."
            )
        db.commit()
        return view(dataset, version)
    key = f"{owner}/{dataset_id}/{checksum}.csv"
    storage.put(key, content)
    dataset = Dataset(id=dataset_id, owner_id=owner, title=title, submission_id=submission)
    db.add(dataset)
    db.flush()
    version = DatasetVersion(
        dataset_id=dataset_id,
        number=1,
        filename=filename,
        storage_key=key,
        sha256=checksum,
        size_bytes=len(content),
        row_count=info.rows,
        columns=info.columns,
    )
    db.add(version)
    db.commit()
    return view(dataset, version)


def preview(version: DatasetVersion, storage: LocalFiles, page: int, page_size: int) -> Preview:
    source = storage.read(version.storage_key, version.sha256)
    reader = csv.reader(io.StringIO(source.decode("utf-8-sig"), newline=""), strict=True)
    next(reader)
    start = (page - 1) * page_size
    return Preview(
        version_id=version.id,
        columns=version.columns,
        rows=list(islice(reader, start, start + page_size)),
        total_rows=version.row_count,
        page=page,
        page_size=page_size,
    )
