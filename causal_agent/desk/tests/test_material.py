"""The after-run material: with a memory and a pack in the design dir, the outcome's and the treatment's scale are lines with
numbers, so "is 5.7 points a lot" can be answered against the outcome's range."""

from __future__ import annotations

from copy import deepcopy

from causal_agent.common.contracts import RunRecord
from causal_agent.desk import material as M
from causal_agent.desk.handoff import forced
from causal_agent.memory import store
from causal_agent.memory.records import Field, Memory


def _run(design_dir) -> RunRecord:
    return RunRecord(index=1, dataset="students3", question="q", family="adjustment", specialist="dowhy", status="done", design_dir=str(design_dir))


def test_the_outcome_and_the_treatment_scale_come_from_the_pack_with_their_numbers(tmp_path):
    m = store.migrate("students3", write=False)
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", ["lunch"], memory=m)
    h.column("math score").facts.bounds = ["0", "100"]
    (tmp_path / "handoff.json").write_text(h.model_dump_json(indent=2))
    mat = M.render(_run(tmp_path), m)
    assert mat.by_address["col:math_score.profile.bounds"] == "0 to 100"
    assert mat.numbers["col:math_score.profile.bounds.low"] == 0.0 and mat.numbers["col:math_score.profile.bounds.high"] == 100.0
    line = mat.by_address["col:math_score.profile.numeric"]
    assert line.startswith("min 0, p50 66, max 100, mean 66.089")
    assert mat.numbers["col:math_score.profile.numeric.max"] == 100.0 and abs(mat.numbers["col:math_score.profile.numeric.mean"] - 66.089) < 1e-6
    assert "col:math_score.profile.numeric" in mat.addresses and "col:math_score.profile.numeric.p50" in mat.addresses
    assert not [a for a in mat.addresses if a.startswith("col:test_preparation_course.profile.")]  # a level column has no scale


def test_no_scale_without_a_pack_but_saved_pack_supplies_it_without_live_memory(tmp_path):
    m = store.migrate("students3", write=False)
    assert not [a for a in M.render(_run(tmp_path), m).addresses if ".profile." in a]  # no pack in the design dir
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", ["lunch"], memory=m)
    (tmp_path / "handoff.json").write_text(h.model_dump_json(indent=2))
    assert "col:math_score.profile.numeric" in M.render(_run(tmp_path)).addresses


def test_chat_compacts_frozen_row_membership_without_losing_results_or_semantic_arrays():
    rows = list(range(100, 120))
    result = {
        "designs": {
            "d1": {
                "rows": rows,
                "row_ids": rows,
                "support": {"source_row_ids": rows},
                "weights_policy": {"input_rows": {"included": rows}, "weights": list(range(20))},
                "target": {"horizons": list(range(20)), "included_units": rows},
            }
        }
    }
    original = deepcopy(result)
    run = RunRecord(index=1, dataset="example", question="q", status="done", specialist_result=result)
    material = M.render(run)
    assert run.specialist_result == original
    for field in ("rows", "row_ids", "support.source_row_ids", "weights_policy.input_rows.included", "target.included_units"):
        address = f"design:d1.{field}"
        assert material.numbers[f"{address}.count"] == 20
        assert material.by_address[f"{address}.fingerprint"] == "b78917cf2e63290ab7d1b577ea177cd8d33d7aab374e9b0c66ef71504cb4ba57"
        assert not any(f"{address}.{index}" in material.addresses for index in range(20))
    assert material.by_address["design:d1.target.included_units.count"] == "20 frozen units"
    for field in ("weights_policy.weights", "target.horizons"):
        assert all(material.numbers[f"design:d1.{field}.{index}"] == index for index in range(20))


def test_hypothetical_and_frozen_context_remain_distinct_from_later_durable_memory():
    address = "col:x.feeds_treatment"
    memory = Memory(name="example", fields={address: Field(value=True, status="confirmed", source="user:turn:1")})
    result = {
        "context_id": "hypothetical-context",
        "context_facts": {address: {"value": False, "status": "confirmed", "source": "user:turn:2"}},
    }
    run = RunRecord(index=2, dataset="example", question="q", status="done", what_if={address: "False"}, specialist_result=result)
    saved_result, saved_memory = deepcopy(run.specialist_result), memory.model_dump()
    material = M.render(run, memory)
    assert material.by_address[f"run.what_if.{address}"] == "False"
    assert material.by_address[f"context.fields.{address}.value"] == "False"
    assert "True" in material.by_address[address]
    assert material.has("run.what_if") and material.has("run.context_policy")
    policy = material.by_address["run.context_policy"]
    assert "frozen admitted context" in policy and "later dataset context" in policy
    assert "isolated context copy" in policy and "never wrote durable memory" in policy
    assert run.specialist_result == saved_result and memory.model_dump() == saved_memory


def test_multi_contrast_brief_retains_failed_lead_estimates_without_promoting_a_rival():
    result = {
        "lead_design_id": "lead",
        "designs": {
            "lead": {
                "endpoint_id": "total.backdoor",
                "contrasts": [
                    {"treated": "low", "control": "none", "reason": "frozen comparison", "cites": []},
                    {"treated": "high", "control": "none", "reason": "frozen comparison", "cites": []},
                    {"treated": "unresolved", "control": "none", "reason": "frozen comparison", "cites": []},
                ],
            },
            "rival": {"endpoint_id": "total.randomized"},
        },
        "estimates": [
            {"design_id": "lead", "contrast": "low_vs_none", "method": "linear_regression", "value": 2.0, "ci_low": 1.0, "ci_high": 3.0},
            {"design_id": "lead", "contrast": "high_vs_none", "method": "linear_regression", "error": "rank deficient"},
            {"design_id": "lead", "contrast": "unresolved_vs_none", "method": "linear_regression", "value": None},
            {"design_id": "rival", "contrast": "high_vs_none", "method": "linear_regression", "value": 99.0},
        ],
    }
    run = RunRecord(index=1, dataset="example", question="q", status="partial", specialist_result=result)
    material = M.render(run)
    text = M.brief(run, None, material)
    assert "Lead contrast low_vs_none: Effect 2" in text and "[estimate:lead.low_vs_none.value]" in text
    assert "Lead contrast high_vs_none: failed: rank deficient [estimate:lead.high_vs_none.error]" in text
    assert "Lead contrast unresolved_vs_none: no point estimate was recorded. [estimate:lead.unresolved_vs_none]" in text
    assert material.has("estimate:lead.unresolved_vs_none")
    assert "Lead design: total.backdoor" in text and "99" not in text and "estimate:rival" not in text
    assert run.specialist_result["lead_design_id"] == "lead"


def test_multi_contrast_brief_shows_each_missing_primary_record_from_the_frozen_design():
    run = RunRecord(
        index=1,
        dataset="example",
        question="q",
        status="execution_failed",
        specialist_result={
            "lead_design_id": "lead",
            "designs": {
                "lead": {
                    "endpoint_id": "total.backdoor",
                    "contrasts": [
                        {"treated": "low", "control": "none", "reason": "frozen comparison", "cites": []},
                        {"treated": "high", "control": "none", "reason": "frozen comparison", "cites": []},
                    ],
                }
            },
            "estimates": [],
        },
    )
    material = M.render(run)
    text = M.brief(run, None, material)
    for index, contrast in enumerate(("low_vs_none", "high_vs_none")):
        address = f"design:lead.contrasts.{index}"
        assert f"Lead contrast {contrast}: no primary estimate record was produced. [{address}]" in text
        assert material.has(address)


def test_large_signed_weights_are_summarised_without_mutating_frozen_weights():
    weights = {str(i): i % 3 - 1 for i in range(300)}
    result = {"designs": {"lead": {"target": {"weights": weights, "horizons": list(range(20))}}}}
    original = deepcopy(result)
    run = RunRecord(index=1, dataset="example", question="q", status="done", specialist_result=result)
    material = M.render(run)
    address = "design:lead.target.weights"
    assert run.specialist_result == original
    for field, number in {"count": 300, "sum": 0, "absolute_sum": 200, "negative_count": 100, "zero_count": 100, "min": -1, "max": 1}.items():
        assert material.numbers[f"{address}.{field}"] == number
    assert material.by_address[f"{address}.fingerprint"] == "ae254cdd30a1fe78acdebcdaca82a1870c1ee2d8adf467f65d328a52676fc9af"
    assert not any(f"{address}.{index}" in material.addresses for index in range(300))
    assert all(material.numbers[f"design:lead.target.horizons.{index}"] == index for index in range(20))


def test_requested_groups_remain_primary_when_overall_reference_succeeds():
    result = {
        "lead_design_id": "lead",
        "designs": {
            "lead": {
                "endpoint_id": "adjustment.linear_conditional",
                "target": {"primary_summaries": [{"contrast": "one_vs_zero", "modifier": "x", "level": level} for level in ("low", "high", "missing")]},
            }
        },
        "estimates": [
            {"design_id": "lead", "contrast": "one_vs_zero", "method": "linear_regression", "value": 99.0, "secondary": True},
            {"design_id": "lead", "contrast": "one_vs_zero", "method": "linear_regression", "modifier": "x", "level": "low", "value": 2.0},
            {"design_id": "lead", "contrast": "one_vs_zero", "method": "linear_regression", "modifier": "x", "level": "high", "error": "no treated support"},
        ],
    }
    run = RunRecord(index=1, dataset="example", question="q", status="partial", specialist_result=result)
    material = M.render(run)
    text = M.brief(run, None, material)
    assert "Lead contrast one_vs_zero, x=low: Effect 2" in text
    assert "Lead contrast one_vs_zero, x=high: failed: no treated support" in text
    assert "Lead contrast one_vs_zero, x=missing: no primary estimate record was produced." in text
    assert material.has("design:lead.target.primary_summaries.2")
    assert "99" not in text and result["lead_design_id"] == "lead"
