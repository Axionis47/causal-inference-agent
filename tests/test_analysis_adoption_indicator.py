"""Explicit cohort-to-indicator diagnostic bindings, without numerical estimation."""

import csv
import io

import pytest
from test_scientific_v8 import longitudinal_study, relation

from backend.analysis import domain


@pytest.fixture(autouse=True)
def isolated():
    yield


def cohort_study():
    context, ids, frame, raw = longitudinal_study()
    indicator = frame.treatments.treatments[0].variable.model_copy(deep=True)
    cohort_role = next(role for role in frame.roles.assignments if role.variable.id == ids["adoption"])
    frame.treatments.treatments[0].variable = cohort_role.variable.model_copy(deep=True)
    frame.treatments.effects[0].treatment_ids = [ids["adoption"]]
    process = frame.treatments.assignments[0]
    process.treatment_ids = [ids["adoption"]]
    process.history_variables = [ids["t"]]
    assert process.diagnostic_window is not None
    process.diagnostic_window.assignment_variable_id = ids["t"]
    frame.roles.assignments.remove(cohort_role)
    cohort_role.variable = indicator
    cohort_role.roles = ["mediator"]
    cohort_role.timing = "after_treatment"
    cohort_role.operational_use = []
    frame.roles.assignments.append(cohort_role)
    for role in frame.roles.assignments:
        role.treatment_ids = [ids["adoption"]]
    return context, ids, frame, raw


def test_explicit_indicator_for_cohort_factor_preserves_rows_and_global_cutoff():
    context, ids, frame, raw = cohort_study()
    original = frame.model_dump()
    projected, manifest = domain.projection(frame, context, raw, purpose="adoption_structure", scientific_version=15)
    rows = list(csv.DictReader(io.StringIO(projected.decode())))
    assert [row[ids["t"]] for row in rows] == ["0", "1", "0", "0"]
    assert [row[ids["y"]] for row in rows] == ["11", "", "21", ""]
    assert [row[ids["unit"]] for row in rows] == ["unit_0", "unit_0", "unit_1", "unit_1"]
    assert ids["proxy"] not in rows[0]
    assert manifest["access"][ids["t"]]["rule"] == "verified_adoption_assignment_metadata"
    assert manifest["access"][ids["t"]]["window"]["treatment_ids"] == [ids["adoption"]]
    assert frame.model_dump() == original
    changed = raw.replace(b",92,", b",992,").replace(b",94,", b",994,")
    assert domain.projection(frame, context, changed, purpose="adoption_structure", scientific_version=15)[0] == projected


def test_indicator_binding_does_not_grant_unscoped_baseline_or_identifier_access():
    context, ids, frame, raw = cohort_study()
    projected, _ = domain.projection(frame, context, raw, purpose="unit_structure", scientific_version=15)
    assert ids["t"] not in next(csv.reader(io.StringIO(projected.decode())))
    frame.treatments.assignments[0].diagnostic_window = None
    projected, _ = domain.projection(frame, context, raw, scientific_version=15)
    assert ids["t"] not in next(csv.reader(io.StringIO(projected.decode())))


def test_fixed_group_label_without_adoption_times_never_grants_target_history_access():
    context, ids, frame, raw = longitudinal_study()
    window = frame.treatments.assignments[0].diagnostic_window
    assert window is not None
    window.assignment_variable_id = ids["t"]
    window.assignment_encoding = "group"
    window.adoption_variable_id = None
    window.cutoff = 3
    # Constant exposed group labels contain no first-on timing. With no adoption
    # mapping this would otherwise expose known target-period outcomes 92 and 94.
    raw = raw.replace(b"0,11", b"1,11")
    with pytest.raises(ValueError, match="cannot establish adoption timing"):
        domain.projection(frame, context, raw, purpose="adoption_structure", scientific_version=15)


def test_cohort_indicator_projection_is_invariant_to_source_names_and_column_order():
    context, _, frame, raw = cohort_study()
    expected, _ = domain.projection(frame, context, raw, purpose="adoption_structure", scientific_version=15)
    records = list(csv.reader(io.StringIO(raw.decode())))
    ordered = sorted(context.columns.values(), key=lambda column: column.identity.position, reverse=True)
    old_positions = [column.identity.position for column in ordered]
    for position, column in enumerate(ordered):
        column.identity = column.identity.model_copy(update={"name": f"opaque_{position}", "position": position})
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([column.identity.name for column in ordered])
    writer.writerows([[row[index] for index in old_positions] for row in records[1:]])
    actual, _ = domain.projection(frame, context, output.getvalue().encode(), purpose="adoption_structure", scientific_version=15)
    assert actual == expected


@pytest.mark.parametrize("defect", [
    "missing_binding", "unbound_variable", "missing_history", "unrelated_factor",
    "multiple_factors", "group_encoding", "unit_as_indicator", "time_as_indicator",
    "late_cutoff", "switching", "cohort_mismatch", "missing_code", "unknown_code",
])
def test_separate_indicator_requires_explicit_consistent_supported_mapping(defect):
    context, ids, frame, raw = cohort_study()
    process = frame.treatments.assignments[0]
    window = process.diagnostic_window
    assert window is not None
    if defect == "missing_binding":
        window.assignment_variable_id = None
    elif defect == "unbound_variable":
        window.assignment_variable_id = "unbound"
        process.history_variables = ["unbound"]
    elif defect == "missing_history":
        process.history_variables = []
    elif defect == "unrelated_factor":
        window.adoption_variable_id = ids["proxy"]
    elif defect == "multiple_factors":
        process.treatment_ids.append(ids["unit"])
    elif defect == "group_encoding":
        window.assignment_encoding = "group"
    elif defect == "unit_as_indicator":
        window.assignment_variable_id = ids["unit"]
        process.history_variables = [ids["unit"]]
    elif defect == "time_as_indicator":
        window.assignment_variable_id = ids["time"]
        process.history_variables = [ids["time"]]
    elif defect == "late_cutoff":
        window.cutoff = 3
    elif defect == "switching":
        raw += b"0,99,a,3,2,999\n"
    elif defect == "cohort_mismatch":
        raw = raw.replace(b",a,1,2,", b",a,1,3,").replace(b",a,2,2,", b",a,2,3,")
    elif defect == "missing_code":
        raw = raw.replace(b"\n1,92", b"\n,92")
    else:
        raw = raw.replace(b"\n1,92", b"\n2,92")
    with pytest.raises(ValueError):
        domain.projection(frame, context, raw, purpose="adoption_structure", scientific_version=15)


@pytest.mark.parametrize("metadata", ["t", "adoption", "unit", "time"])
@pytest.mark.parametrize("qualification", [{}, {"qualifications": "Only some target-period records"}])
def test_metadata_source_lineage_is_checked_independently_of_scientific_role(metadata, qualification):
    context, ids, frame, raw = cohort_study()
    context.relationships["unsafe"] = relation(ids["y"], ids[metadata], **qualification)
    with pytest.raises(ValueError, match="Outcome measurements or unresolved outcome lineage"):
        domain.projection(frame, context, raw, purpose="adoption_structure", scientific_version=15)


def test_selected_indicator_still_works_when_explicitly_bound():
    context, ids, frame, raw = longitudinal_study()
    window = frame.treatments.assignments[0].diagnostic_window
    assert window is not None
    window.assignment_variable_id = ids["t"]
    projected, manifest = domain.projection(frame, context, raw, purpose="adoption_structure", scientific_version=15)
    assert [row[ids["y"]] for row in csv.DictReader(io.StringIO(projected.decode()))] == ["11", "", "21", ""]
    assert manifest["preperiod_window"]["windows"][0]["assignment_variable_ids"] == [ids["t"]]
