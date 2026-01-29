import csv
import io
from dataclasses import dataclass

from backend.config import settings


@dataclass(frozen=True)
class CsvInfo:
    columns: list[str]
    rows: int


def inspect_csv(content: bytes, filename: str) -> CsvInfo:
    limits = settings()
    if not filename.lower().endswith(".csv"):
        raise ValueError("Choose a .csv file.")
    if not content or len(content) > limits.max_upload_bytes:
        raise ValueError("Choose a nonempty CSV of at most 10 MB.")
    if b"\x00" in content:
        raise ValueError("The file must be a UTF-8 CSV without NUL bytes.")
    try:
        reader = csv.reader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader, [])
        if not header or any(not name.strip() for name in header):
            raise ValueError("Every column needs a nonempty header.")
        if len(set(header)) != len(header):
            raise ValueError("Column headers must be unique.")
        if len(header) > limits.max_columns:
            raise ValueError(f"The CSV may contain at most {limits.max_columns} columns.")
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != len(header):
                raise ValueError(f"Record {count + 1} has {len(row)} fields; expected {len(header)}.")
            if count > limits.max_rows:
                raise ValueError(f"The CSV may contain at most {limits.max_rows:,} rows.")
        if not count:
            raise ValueError("The CSV needs at least one data row.")
        return CsvInfo(header, count)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("Unable to read this file as a UTF-8 CSV.") from error
