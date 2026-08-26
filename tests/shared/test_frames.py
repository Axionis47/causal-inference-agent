"""Content-addressed frame artifacts: dtype fidelity and reopen verification (T-016 §1.3)."""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

import polars as pl
import pytest

from causal.shared.frames import (
    FRAME_HASH_MISMATCH,
    UNSUPPORTED_FRAME_DTYPE,
    FrameError,
    dtype_name,
    frame_bytes,
    read_frame,
    write_frame,
)
from tests.conftest import MemoryObjects, requires_docker

FRAME = pl.DataFrame({
    "unit": ["u1", "u2", None], "count": [1, None, 3], "score": [1.5, 2.25, None],
    "flag": [True, None, False], "day": [dt.date(2020, 1, 2), None, dt.date(2021, 3, 4)],
    "seen": [dt.datetime.fromisoformat("2020-01-02T03:04:05.123456"), None, None],
})


@requires_docker
def test_a_stored_frame_reopens_with_its_dtypes_and_values(object_store: Any) -> None:
    artifact = write_frame(object_store, FRAME)
    digest = hashlib.sha256(frame_bytes(FRAME)).hexdigest()
    assert artifact.object_locator == f"objects/{digest}"
    assert artifact.content_hash == digest
    assert artifact.row_count == 3
    reopened = read_frame(object_store, artifact)
    assert reopened.dtypes == FRAME.dtypes
    assert reopened.equals(FRAME)


def test_the_same_values_always_serialize_to_the_same_bytes() -> None:
    assert frame_bytes(FRAME) == frame_bytes(FRAME.clone())
    assert frame_bytes(FRAME) != frame_bytes(FRAME.with_columns(pl.col("count") * 2))


def test_a_tampered_object_fails_the_reopen_hash() -> None:
    objects = MemoryObjects()
    artifact = write_frame(objects, FRAME)
    objects.data[artifact.object_locator] = b"unit,count,score,flag,day,seen\n"
    with pytest.raises(FrameError) as error:
        read_frame(objects, artifact)
    assert error.value.code == FRAME_HASH_MISMATCH


def test_a_dtype_outside_the_v1_vocabulary_fails_closed() -> None:
    with pytest.raises(FrameError) as error:
        dtype_name(pl.Int32())
    assert error.value.code == UNSUPPORTED_FRAME_DTYPE
