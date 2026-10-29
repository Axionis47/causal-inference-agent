"""CSV structure validation only. The original stream is never rewritten."""

import csv
import hashlib
import io
from typing import BinaryIO

from causal_agent.common.addresses import key
from causal_agent.common.config import Ingestion


class InvalidUpload(ValueError):
    pass


def validate(stream: BinaryIO, filename: str, limits: Ingestion) -> tuple[str, int]:
    if not filename.lower().endswith(".csv"):
        raise InvalidUpload("Choose a .csv file.")
    digest, size = hashlib.sha256(), 0
    stream.seek(0)
    while chunk := stream.read(64 * 1024):
        size += len(chunk)
        if size > limits.max_bytes:
            raise InvalidUpload(f"The CSV exceeds the {limits.max_bytes} byte limit.")
        if b"\x00" in chunk:
            raise InvalidUpload("The CSV contains NUL bytes; upload a UTF-8 text CSV.")
        digest.update(chunk)
    stream.seek(0)
    wrapper = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    try:
        csv.field_size_limit(max(csv.field_size_limit(), limits.max_bytes))
        reader = csv.reader(wrapper, strict=True)
        header = next(reader, None)
        if not header or any(not h.strip() for h in header):
            raise InvalidUpload("The CSV needs a nonempty header for every column.")
        normalized = [key(h) for h in header]
        if len(set(header)) != len(header) or len(set(normalized)) != len(header) or not all(normalized):
            raise InvalidUpload("Column headers must be unique, including after normalization (for example 'exam score' and 'exam-score' conflict).")
        if len(header) > limits.max_columns:
            raise InvalidUpload(f"The CSV exceeds {limits.max_columns} columns.")
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != len(header):
                raise InvalidUpload(f"CSV record {count + 1} has {len(row)} fields; expected {len(header)}.")
            if count > limits.max_rows or count * len(header) > limits.max_cells:
                raise InvalidUpload("The CSV exceeds the configured row or cell limit.")
        if not count:
            raise InvalidUpload("The CSV has headers but no data records.")
    except (UnicodeError, csv.Error) as exc:
        raise InvalidUpload(f"Invalid UTF-8 CSV: {exc}") from exc
    finally:
        wrapper.detach()
        stream.seek(0)
    return digest.hexdigest(), size
