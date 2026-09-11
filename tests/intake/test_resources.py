"""Resource processing preserves inventory while isolating individual failures."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from causal.intake.archive import ArchiveSafety
from causal.intake.resources import inventory_archive, process_resource
from tests.intake.test_archive import build_zip


@pytest.mark.parametrize(
    ("name", "data", "status", "has_profile", "has_document"),
    [
        ("table.csv", b"a\n1\n", "parsed", True, False),
        ("table.tsv", b"a\tb\n1\t2\n", "parsed", True, False),
        ("empty.csv", b"", "failed", False, False),
        ("readme.md", b"# Dataset", "parsed", False, True),
        ("broken.txt", b"\xff", "failed", False, False),
        ("metadata.json", b"{}", "parsed", False, True),
        ("metadata.yaml", b"units: dollars", "excluded", False, False),
        ("figure.pdf", b"%PDF", "unreadable", False, False),
        ("model.ipynb", b"{}", "excluded", False, False),
    ],
)
def test_resource_outcomes(
    name: str, data: bytes, status: str, has_profile: bool, has_document: bool,
) -> None:
    archive_bytes = build_zip({name: data})
    safety = ArchiveSafety()
    inventory = inventory_archive(archive_bytes, safety.admit(archive_bytes), safety)
    result = process_resource(inventory.entries[0])
    assert result.status == status
    assert (result.profile is not None) == has_profile
    assert (result.document is not None) == has_document
    assert (result.reason is None) == (status == "parsed")


def test_inventory_preserves_manifest_hashes_and_withheld_members() -> None:
    archive_bytes = build_zip({"table.csv": b"a\n1\n", "script.py": b"print('no')"})
    safety = ArchiveSafety()
    inventory = inventory_archive(archive_bytes, safety.admit(archive_bytes), safety)
    table, script = inventory.entries
    assert table.sha256 == hashlib.sha256(b"a\n1\n").hexdigest()
    assert script.data is None and script.byte_size == len(b"print('no')")
    assert script.sha256 == hashlib.sha256(archive_bytes).hexdigest()
    manifest = inventory.manifest_payload("dataset-id")
    assert manifest == {
        "schema_version": "source-manifest.v1", "dataset_id": "dataset-id",
        "resources": [
            {"logical_name": entry.name, "classification": entry.classification,
             "sha256": entry.sha256, "byte_size": entry.byte_size,
             "media_type": entry.media_type, "reason": entry.reason}
            for entry in inventory.entries],
    }


def test_corrupt_member_preserves_good_table_and_terminal_resource_status() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("good.csv", b"a\n1\n")
        archive.writestr("broken.md", b"original document")
    data = buffer.getvalue().replace(b"original document", b"tampered document")
    safety = ArchiveSafety()
    admission = safety.admit(data)
    assert admission.safe
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        safety.extract_admitted(data, admission)
    inventory = inventory_archive(data, admission, safety)
    results = [process_resource(entry) for entry in inventory.entries]
    assert [result.status for result in results] == ["parsed", "failed"]
    assert results[0].profile is not None and results[0].profile["row_count"] == 1
    assert results[1].resource.name == "broken.md"
    assert results[1].reason == "archive member unreadable: BadZipFile"


def test_unsafe_inventory_records_every_member_without_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = build_zip({"good.csv": b"a\n1\n", "../bad.csv": b"a\n2\n"})
    safety = ArchiveSafety()

    def no_extract(*args: object, **kwargs: object) -> None:
        pytest.fail("a refused archive must never be extracted")

    monkeypatch.setattr(safety, "extract_admitted", no_extract)
    inventory = inventory_archive(data, safety.admit(data), safety)
    assert [entry.name for entry in inventory.entries] == ["good.csv", "../bad.csv"]
    assert [entry.byte_size for entry in inventory.entries] == [4, 4]
    assert all(entry.data is None for entry in inventory.entries)
    assert all(entry.sha256 == hashlib.sha256(data).hexdigest() for entry in inventory.entries)
    assert process_resource(inventory.entries[1]).status == "unsafe"
    assert inventory.entries[1].reason == "path traversal"


def test_corrupt_archive_inventory_is_empty_without_reopening() -> None:
    data = b"this is not a zip archive"
    safety = ArchiveSafety()
    admission = safety.admit(data)
    assert not admission.safe
    assert inventory_archive(data, admission, safety).entries == ()
