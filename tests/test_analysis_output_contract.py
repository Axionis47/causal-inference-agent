"""Static facts writer regressions. No database, sandbox, or generated-code execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.analysis import code_checks
from backend.analysis.contracts import ProgramVerification
from backend.context.service import fingerprint

NESTED_PROGRAMS = [
    # Reduced from the original-name Hillstrom program: explicit per-effect dicts.
    """import json
results = {}
results["mens_email_vs_no_email"] = {
    "E_confidence_status": 0, "E_region_count": 1,
    "E_region_0_low": float(1), "E_region_0_high": float(3),
    "E_estimate": float(2), "E_n": int(8),
}
results["womens_email_vs_no_email"] = {
    "E_confidence_status": 0, "E_region_count": 1,
    "E_region_0_low": float(2), "E_region_0_high": float(4),
    "E_estimate": float(3), "E_n": int(9),
}
with open("facts.json", "w") as f:
    json.dump(results, f, allow_nan=False)
""",
    # Reduced from the opaque-name program: dynamic effect IDs and opaque library values.
    """import json
import pandas as pd
data = pd.DataFrame()
contrasts = [{"id": "mens_email_vs_no_email"}, {"id": "womens_email_vs_no_email"}]
results = {}
for contrast in contrasts:
    effect_id = contrast["id"]
    results[effect_id] = {
        "E_estimate": data.mean(), "E_ci_low": data.min(),
        "E_ci_high": data.max(), "E_n": len(data),
    }
with open("facts.json", "w") as f:
    json.dump(results, f, allow_nan=False)
""",
]


@pytest.mark.parametrize("program", NESTED_PROGRAMS, ids=["original-name", "opaque-name"])
def test_saved_failure_shapes_are_rejected_before_approval(program):
    with pytest.raises(code_checks.ProgramCheckError, match=r"facts.json flat numeric mapping.*arg-type"):
        code_checks.verify(program)


@pytest.mark.parametrize(
    "value",
    ['{"nested": {"estimate": 2}}', '{"label": "not numeric"}', '{"missing": None}', "[1, 2]", "{1: 2.0}"],
)
def test_literal_known_facts_payloads_have_a_numeric_mapping_contract(value):
    with pytest.raises(code_checks.ProgramCheckError, match="arg-type"):
        code_checks.verify(f'import json\njson.dump({value}, open("facts.json", "w"))\n')


def test_literal_boolean_is_rejected_despite_python_numeric_subtyping():
    with pytest.raises(code_checks.ProgramCheckError, match="boolean values are not numeric facts"):
        code_checks.verify('import json\njson.dump({"primary_estimate": True}, open("facts.json", "w"))\n')


@pytest.mark.parametrize(
    "writer",
    [
        'json.dump(results, open("facts.json", "w"))',
        'with open("facts.json", "w") as facts:\n    json.dump(results, facts, allow_nan=False)',
        'with open(file="facts.json", mode="w") as facts:\n    json.dump(obj=results, fp=facts)',
    ],
)
def test_flat_integer_float_maps_pass_for_direct_and_with_writers(writer):
    program = 'import json\nresults = {"primary_estimate": 2.0, "primary_n": 8}\n' + writer + "\n"
    verified = code_checks.verify(program)
    assert verified["facts_writers_checked"] == 1
    assert verified["passed"] and not verified["target_data_access"]
    assert verified["source_hash"] == fingerprint(program)


def test_incrementally_built_flat_effect_map_passes():
    program = """import json
results = {}
for effect_id in ["primary", "secondary"]:
    results[effect_id + "_estimate"] = float(2)
    results[effect_id + "_n"] = int(8)
with open("facts.json", "w") as stream:
    json.dump(results, stream)
"""
    assert code_checks.verify(program)["facts_writers_checked"] == 1


@pytest.mark.parametrize(
    "imports, writer",
    [
        ("import json as serialization", "serialization.dump"),
        ("from json import dump as write_json", "write_json"),
    ],
)
def test_import_aliases_keep_the_known_destination_contract(imports, writer):
    program = f'{imports}\nwith open("facts.json", "w") as handle:\n    {writer}({{"nested": {{"estimate": 2}}}}, handle)\n'
    with pytest.raises(code_checks.ProgramCheckError, match="arg-type"):
        code_checks.verify(program)


def test_nested_other_json_artifacts_and_unrelated_type_errors_are_not_facts_errors():
    program = """import json
def takes_integer(value: int) -> int:
    return value
takes_integer("unrelated package argument")
with open("metadata.json", "w") as handle:
    json.dump({"assumptions": {"conditional": True}}, handle)
json.dump({"primary_estimate": 2.0, "primary_n": 8}, open("facts.json", "w"))
"""
    verified = code_checks.verify(program)
    assert verified["facts_writers_checked"] == verified["other_json_writers"] == 1


def test_reassigned_stream_is_not_misclassified_as_the_facts_destination():
    program = """import json
with open("facts.json", "w") as handle:
    handle = open("metadata.json", "w")
    json.dump({"nested": {"value": "metadata"}}, handle)
"""
    verified = code_checks.verify(program)
    assert verified["facts_writers_checked"] == 0
    assert verified["json_writers_unknown"] == 1


def test_unknown_paths_and_any_values_report_the_limits_of_the_static_check():
    program = """import json
def untyped_value():
    return {"nested": {"value": 1}}
destination = "facts.json"
json.dump(untyped_value(), open(destination, "w"))
json.dump(untyped_value(), open("facts.json", "w"))
"""
    verified = code_checks.verify(program)
    assert verified["json_writers_unknown"] == verified["facts_writers_checked"] == 1
    assert "Unknown paths" in verified["limitations"] and "Any-typed values" in verified["limitations"]


def test_checking_copy_and_stub_do_not_replace_or_execute_original_source(monkeypatch):
    program = """from __future__ import annotations
import json
raise RuntimeError("Generated source must never execute during verification")
json.dump({"primary_estimate": 2.0}, open("facts.json", "w"))
"""
    captured = {}
    original_run = code_checks.subprocess.run

    def inspect_copy(*args, **kwargs):
        root = Path(kwargs["cwd"])
        captured["original"] = (root / "program.py").read_text()
        captured["checking"] = (root / "facts_check.py").read_text()
        captured["stub"] = (root / "_causal_output_contract.pyi").read_text()
        return original_run(*args, **kwargs)

    monkeypatch.setattr(code_checks.subprocess, "run", inspect_copy)
    verified = code_checks.verify(program)
    assert captured["original"] == program
    assert "_causal_numeric_facts_" not in captured["original"]
    assert "_causal_numeric_facts_" in captured["checking"] and "Mapping[str, float]" in captured["stub"]
    assert verified["source_hash"] == fingerprint(program)


def test_comments_cannot_suppress_the_owned_output_check():
    program = """# mypy: disable-error-code="arg-type"
import json
json.dump({"nested": {"estimate": 2}}, open("facts.json", "w"))  # type: ignore[arg-type]
"""
    with pytest.raises(code_checks.ProgramCheckError, match="arg-type"):
        code_checks.verify(program)


def test_existing_control_flow_checks_still_apply_to_unmodified_checking_source():
    program = """import json
def interval(variance):
    if variance > 0:
        critical = 2.0
    return critical  # type: ignore[possibly-undefined]
json.dump({"primary_estimate": 2.0}, open("facts.json", "w"))
"""
    with pytest.raises(code_checks.ProgramCheckError, match="possibly-undefined"):
        code_checks.verify(program)


def test_output_contract_uses_exact_effect_names_and_declared_result_shapes():
    manifest = code_checks.output_contract(
        {
            "effects": [
                {"effect_id": "mens_email", "result_kind": "estimate"},
                {"effect_id": "point", "result_kind": "estimate_without_inference"},
                {"effect_id": "zero", "result_kind": "structural_zero"},
                {"effect_id": "weak", "result_kind": "confidence_set"},
            ]
        }
    )
    ordinary, point, zero, confidence = manifest["effects"]
    assert ordinary["required_fields"] == [
        "mens_email_estimate",
        "mens_email_ci_low",
        "mens_email_ci_high",
        "mens_email_n",
    ]
    assert point["required_fields"] == ["point_estimate", "point_n"]
    assert zero["required_fields"] == ["zero_estimate"] and zero["fixed_values"] == {"zero_estimate": 0}
    assert confidence["required_fields"] == ["weak_confidence_status", "weak_region_count", "weak_n"]
    assert confidence["region_fields"]["low_pattern"] == "weak_region_{i}_low"
    assert confidence["conditional_fields"]["when_confidence_status_is_3"] == [
        "weak_searched_low",
        "weak_searched_high",
    ]
    assert manifest["nested_effect_objects"] is False
    with pytest.raises(ValueError, match="Unsupported frozen result kind"):
        code_checks.output_contract({"effects": [{"effect_id": "primary", "result_kind": "invented"}]})


def test_verification_scope_metadata_remains_readable_for_historical_records():
    current = code_checks.verify('import json\njson.dump({"primary_estimate": 2}, open("facts.json", "w"))\n')
    assert ProgramVerification.model_validate(current).facts_writers_checked == 1
    legacy = {
        key: value
        for key, value in current.items()
        if key not in {"facts_writers_checked", "json_writers_unknown", "other_json_writers"}
    }
    assert ProgramVerification.model_validate(legacy).facts_writers_checked is None


@pytest.mark.parametrize(
    "kind, values",
    [
        ("estimate", {"estimate": 2.0, "ci_low": 1.0, "ci_high": 3.0, "n": 8}),
        ("estimate_without_inference", {"estimate": 2.0, "n": 8}),
        ("structural_zero", {"estimate": 0}),
        (
            "confidence_set",
            {"confidence_status": 0, "region_count": 1, "n": 8, "region_0_low": 1.0, "region_0_high": 3.0},
        ),
        (
            "confidence_set",
            {"confidence_status": 3, "region_count": 0, "n": 8, "searched_low": -10.0, "searched_high": 10.0},
        ),
    ],
)
def test_manifest_required_fields_match_the_runtime_result_validator(kind, values):
    from backend.analysis.domain import execution_error

    plan = {"effects": [{"effect_id": "primary", "result_kind": kind}]}
    required = set(code_checks.output_contract(plan)["effects"][0]["required_fields"])
    facts = {f"primary_{name}": value for name, value in values.items()}
    assert required <= facts.keys()
    effects = SimpleNamespace(effects=[SimpleNamespace(id="primary")])
    frozen = SimpleNamespace(effects=[SimpleNamespace(effect_id="primary", result_kind=kind)])
    assert execution_error(facts, effects, frozen) == ""
