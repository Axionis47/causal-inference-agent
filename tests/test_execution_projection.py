"""Post-approval input mapping preserves source values without numerical analysis."""

import csv
import hashlib
import io
from uuid import uuid4

import pytest

from backend.analysis.contracts import InputProjection
from backend.analysis.domain import execution_projection
from backend.context.contracts import empty_content
from backend.context.service import fingerprint


@pytest.fixture(autouse=True)
def isolated():
    # Override the integration suite's database reset for these pure value checks.
    yield


def csv_bytes(rows):
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    return output.getvalue().encode()


def parsed(raw):
    return list(csv.reader(io.StringIO(raw.decode(), newline="")))


def fixture():
    context = empty_content(uuid4(), ["treatment", "outcome", "unit"])
    identities = {column.identity.name: column.identity for column in context.columns.values()}
    manifest = InputProjection(columns=[identities["outcome"], identities["treatment"], identities["unit"]])
    rows = [["1.0", "001.2300", "a,b"], ["0", "", "line\r\nbreak"], ["+0.0", "1e-09", "λ"]]
    return context, identities, manifest, rows


def test_execution_projection_preserves_every_row_and_cell_string_with_bound_hashes():
    context, identities, manifest, rows = fixture()
    source = csv_bytes([["treatment", "outcome", "unit"], *rows])
    original_context, original_manifest = context.model_dump(), manifest.model_dump()
    projected, receipt = execution_projection(context, source, manifest)
    ordered = sorted(identities.values(), key=lambda column: column.column_id)
    assert parsed(projected) == [[column.column_id for column in ordered], *[[row[column.position] for column in ordered] for row in rows]]
    assert receipt["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert receipt["projection_sha256"] == hashlib.sha256(projected).hexdigest()
    assert receipt["manifest_hash"] == fingerprint(original_manifest)
    assert receipt["rows"] == len(rows) and receipt["column_count"] == 3
    assert context.model_dump() == original_context and manifest.model_dump() == original_manifest


@pytest.mark.parametrize("padding", [0, 48, 188])
def test_projection_is_invariant_to_explicit_renaming_reordering_and_inert_padding(padding):
    context, identities, manifest, rows = fixture()
    original, _ = execution_projection(context, csv_bytes([["treatment", "outcome", "unit"], *rows]), manifest)
    source_order = ["unit", "outcome", "treatment"]
    names = {name: "opaque_" + str(index) for index, name in enumerate(source_order)}
    changed = context.model_copy(deep=True)
    for position, name in enumerate(source_order):
        identity = changed.columns[identities[name].column_id].identity
        identity.name, identity.position = names[name], position
    extras = empty_content(uuid4(), ["unused_" + str(index) for index in range(padding)])
    for column in extras.columns.values():
        column.identity.position += len(source_order)
    changed.columns.update(extras.columns)
    header = [names[name] for name in source_order] + [column.identity.name for column in extras.columns.values()]
    transformed_rows = [[row[identities[name].position] for name in source_order] + [str(index * 37 + j) for j in range(padding)] for index, row in enumerate(rows)]
    transformed_manifest = InputProjection(columns=[changed.columns[column.column_id].identity for column in manifest.columns])
    projected, receipt = execution_projection(changed, csv_bytes([header, *transformed_rows]), transformed_manifest)
    assert projected == original
    assert receipt["source_column_count"] == 3 + padding and receipt["column_count"] == 3


@pytest.mark.parametrize("defect", ["header_order", "header_name", "row_width", "manifest_name", "manifest_position", "manifest_duplicate", "registry_position"])
def test_projection_fails_closed_for_source_or_frozen_mapping_mismatches(defect):
    context, _, manifest, rows = fixture()
    header = ["treatment", "outcome", "unit"]
    if defect == "header_order":
        header.reverse()
    elif defect == "header_name":
        header[0] = "renamed_without_rebinding"
    elif defect == "row_width":
        rows[0].append("extra")
    elif defect == "manifest_name":
        manifest = manifest.model_copy(deep=True)
        manifest.columns[0].name = "invented"
    elif defect == "manifest_position":
        manifest = manifest.model_copy(deep=True)
        manifest.columns[0].position = 99
    elif defect == "manifest_duplicate":
        manifest.columns.append(manifest.columns[0])
    else:
        next(iter(context.columns.values())).identity.position = 99
    with pytest.raises(ValueError):
        execution_projection(context, csv_bytes([header, *rows]), manifest)


def test_projection_uses_utf8_bom_as_encoding_metadata_and_retains_blank_cells():
    context, _, manifest, rows = fixture()
    source = csv_bytes([["treatment", "outcome", "unit"], *rows])
    plain, first = execution_projection(context, source, manifest)
    bom, second = execution_projection(context, b"\xef\xbb\xbf" + source, manifest)
    assert plain == bom
    assert first["source_sha256"] != second["source_sha256"]
    assert first["projection_sha256"] == second["projection_sha256"]
