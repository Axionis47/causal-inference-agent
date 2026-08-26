"""Content-addressed frame artifacts: deterministic CSV bytes plus dtype fidelity (PRD-003 §9.6, §19)."""

from __future__ import annotations

import hashlib
import io
from typing import Annotated, Final, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from causal.shared.contracts import Identity, PayloadLocator, Sha256Hex

__all__ = [
    "FRAME_DTYPES", "FRAME_HASH_MISMATCH", "FRAME_SCHEMA_MISMATCH", "UNSUPPORTED_FRAME_DTYPE",
    "FrameArtifactV1", "FrameColumnV1", "FrameError", "FrameObjectStore", "dtype_name",
    "frame_bytes", "frame_columns", "read_frame", "write_frame",
]

FRAME_HASH_MISMATCH: Final = "frame_hash_mismatch"
FRAME_SCHEMA_MISMATCH: Final = "frame_schema_mismatch"
UNSUPPORTED_FRAME_DTYPE: Final = "unsupported_frame_dtype"

# A CSV carries no dtypes, so a frame artifact names them from this closed V1 vocabulary
# and `read_frame` re-applies them; nothing is ever inferred from the stored bytes.
FRAME_DTYPES: Final[dict[str, pl.DataType]] = {
    "Boolean": pl.Boolean(), "Int64": pl.Int64(), "Float64": pl.Float64(),
    "String": pl.String(), "Date": pl.Date(), "Datetime": pl.Datetime("us"),
}


class FrameError(ValueError):
    """A frame cannot be written or reopened. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class FrameObjectStore(Protocol):
    """The two `ObjectStore` methods a frame artifact needs (D-019 locator)."""

    def put_if_absent(self, digest: str, data: bytes) -> str: ...

    def get(self, locator: str) -> bytes: ...


class FrameColumnV1(BaseModel):
    """One stored column: its name and its dtype name from `FRAME_DTYPES`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    column_name: Identity
    dtype: Identity


class FrameArtifactV1(BaseModel):
    """Where one immutable frame lives, what it hashes to, and how to re-type it."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    columns: Annotated[tuple[FrameColumnV1, ...], Field(min_length=1)]
    row_count: Annotated[int, Field(ge=0)]
    content_hash: Sha256Hex
    object_locator: PayloadLocator

    def schema_overrides(self) -> dict[str, pl.DataType]:
        """The stored dtypes, keyed by column name, for a no-inference reread."""
        return {column.column_name: FRAME_DTYPES[column.dtype] for column in self.columns}


def dtype_name(dtype: pl.DataType) -> str:
    """The `FRAME_DTYPES` name of one polars dtype; an out-of-vocabulary dtype fails closed."""
    name = str(dtype).split("(", 1)[0]
    if name not in FRAME_DTYPES:
        raise FrameError(f"dtype outside the V1 frame vocabulary: {dtype}", UNSUPPORTED_FRAME_DTYPE)
    return name


def frame_columns(frame: pl.DataFrame) -> tuple[FrameColumnV1, ...]:
    """The frame's schema as stored column records, in frame column order."""
    return tuple(
        FrameColumnV1(column_name=name, dtype=dtype_name(dtype))
        for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
    )


def frame_bytes(frame: pl.DataFrame) -> bytes:
    """Deterministic CSV bytes: fixed separator, quoting, null token, and date formats."""
    return frame.write_csv(
        separator=",", quote_char='"', line_terminator="\n", null_value="",
        quote_style="necessary", date_format="%Y-%m-%d",
        datetime_format="%Y-%m-%dT%H:%M:%S%.6f", time_format="%H:%M:%S%.6f",
    ).encode("utf-8")


def write_frame(objects: FrameObjectStore, frame: pl.DataFrame) -> FrameArtifactV1:
    """Serialize, hash, and store one frame at `objects/{sha256}`."""
    data = frame_bytes(frame)
    digest = hashlib.sha256(data).hexdigest()
    locator = objects.put_if_absent(digest, data)
    return FrameArtifactV1(
        columns=frame_columns(frame), row_count=frame.height,
        content_hash=digest, object_locator=locator,
    )


def read_frame(objects: FrameObjectStore, artifact: FrameArtifactV1) -> pl.DataFrame:
    """Reopen a stored frame: verify the hash, then re-apply the recorded dtypes."""
    data = objects.get(artifact.object_locator)
    digest = hashlib.sha256(data).hexdigest()
    if digest != artifact.content_hash:
        raise FrameError(
            f"frame object {artifact.object_locator} hashes to {digest}", FRAME_HASH_MISMATCH
        )
    frame = pl.read_csv(
        io.BytesIO(data), has_header=True, infer_schema_length=0,
        schema_overrides=artifact.schema_overrides(), truncate_ragged_lines=False,
    )
    if frame.columns != [column.column_name for column in artifact.columns]:
        raise FrameError(
            f"frame object {artifact.object_locator} does not match its recorded schema",
            FRAME_SCHEMA_MISMATCH,
        )
    return frame
