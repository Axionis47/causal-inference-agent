"""Tests for the gate's bundle normalisation (raw files -> canonical parquet).

normalize_bundle converts every tabular file Kaggle shipped to one parquet under
the job's normalized dir, flags the rest non-tabular, and records a parse error
when a tabular-looking file does not load. Downstream and the row preview read
only the parquet, so this pass is what makes a multi-format bundle previewable.
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from src.storage.job_data import job_normalized_dir, job_raw_dir
from src.storage.normalize import normalize_bundle


@pytest.fixture(autouse=True)
def storage_root(tmp_path, monkeypatch):
    """Point the per-job storage root at a temp dir for every test here."""
    fake = types.SimpleNamespace(local_storage_path=str(tmp_path / "store"))
    monkeypatch.setattr("src.storage.job_data.get_settings", lambda: fake)


def test_normalize_bundle_converts_csv_tsv_json_to_parquet():
    job = "norm-1"
    raw = job_raw_dir(job)
    (raw / "a.csv").write_text("x,y\n1,2\n3,4\n")
    (raw / "b.tsv").write_text("x\ty\n5\t6\n")
    (raw / "c.json").write_text('[{"x": 7, "y": 8}]')

    by = {r.name: r for r in normalize_bundle(job)}

    assert by["a.csv"].tabular is True
    assert by["a.csv"].normalized_name == "a.csv.parquet"
    assert by["a.csv"].columns == ["x", "y"]
    assert by["a.csv"].n_rows == 2
    assert by["b.tsv"].columns == ["x", "y"] and by["b.tsv"].n_rows == 1
    assert by["c.json"].tabular is True and by["c.json"].n_rows == 1

    # The parquet really exists and round-trips the values.
    norm = job_normalized_dir(job)
    assert (norm / "a.csv.parquet").is_file()
    assert pd.read_parquet(norm / "a.csv.parquet")["y"].tolist() == [2, 4]


def test_normalize_reads_excel_to_parquet():
    job = "norm-xl"
    raw = job_raw_dir(job)
    pd.DataFrame({"x": [1, 2]}).to_excel(raw / "sheet.xlsx", index=False)

    r = {x.name: x for x in normalize_bundle(job)}["sheet.xlsx"]
    assert r.tabular is True
    assert r.n_rows == 2
    assert r.columns == ["x"]
    assert (job_normalized_dir(job) / "sheet.xlsx.parquet").is_file()


def test_normalize_flags_non_tabular_file_without_writing_parquet():
    job = "norm-2"
    raw = job_raw_dir(job)
    (raw / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    (r,) = normalize_bundle(job)
    assert r.name == "logo.png"
    assert r.tabular is False
    assert r.normalized_name is None
    assert not (job_normalized_dir(job) / "logo.png.parquet").exists()


def test_normalize_records_error_for_unparseable_tabular_file():
    job = "norm-3"
    raw = job_raw_dir(job)
    (raw / "broken.json").write_text("{not valid json")

    (r,) = normalize_bundle(job)
    assert r.tabular is False
    assert r.normalized_name is None
    assert r.error  # the parser message is kept for the UI


def test_normalize_bundle_drops_stale_parquet_from_a_prior_run():
    job = "norm-4"
    raw = job_raw_dir(job)
    (raw / "a.csv").write_text("x\n1\n")
    normalize_bundle(job)

    # A leftover parquet from a previous, different bundle for the same job.
    norm = job_normalized_dir(job)
    (norm / "old.csv.parquet").write_bytes(b"stale")

    normalize_bundle(job)
    assert not (norm / "old.csv.parquet").exists()
    assert (norm / "a.csv.parquet").is_file()
