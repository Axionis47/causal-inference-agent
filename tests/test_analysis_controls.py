"""Deterministic regressions for the failures found in public-data evaluation."""

import csv
import io
from uuid import uuid4

import pytest
from test_analysis import claim, decision, variable

from backend.analysis import code_checks, domain, lineage
from backend.analysis.contracts import RoleAssignment, Roles, Target
from backend.context.contracts import ContextContent, Relationship, empty_content


@pytest.mark.parametrize("comment", ["", "# type: ignore[possibly-undefined]"])
def test_static_gate_rejects_one_zero_variance_branch_even_with_suppression(comment):
    code = f'''# mypy: ignore-errors
def interval(s1, s2):
    if s1 == 0 and s2 == 0:
        critical = 0
    elif s1 == 0:
        df = 4
    elif s2 == 0:
        df = 3
    else:
        critical = 2
    return critical * (s1 + s2) {comment}
'''
    with pytest.raises(code_checks.ProgramCheckError, match="possibly-undefined"):
        code_checks.verify(code)


def test_static_gate_accepts_explicit_failure_and_defined_branches_without_executing():
    code = '''raise RuntimeError("A checker must not execute this module")
def interval(s1, s2, quantile):
    if s1 == 0 and s2 == 0:
        raise ValueError("Undefined inference")
    if s1 == 0:
        df = 4
    elif s2 == 0:
        df = 3
    else:
        df = 5
    critical = quantile(df)
    return critical * (s1 + s2)
'''
    result = code_checks.verify(code)
    assert result["passed"] and result["target_data_access"] is False


@pytest.mark.parametrize("names", [["outcome", "visit", "conversion", "baseline"], ["c8", "a2", "d9", "b3"]])
def test_lineage_is_derived_from_evidence_not_model_proxy_votes_or_names(names):
    context = empty_content(uuid4(), names)
    columns = list(context.columns.values())
    ids = [c.identity.column_id for c in columns]
    target = Target(**decision(), outcome=variable(columns[0].identity.model_dump()))
    roles = Roles(**decision(), assignments=[
        RoleAssignment(variable=variable(c.identity.model_dump()), roles=["unknown"],
            timing="before_treatment" if index == 3 else "after_treatment",
            timing_evidence=claim(), account=claim(), outcome_proxy=bool(index % 2), reason="Fixture")
        for index, c in enumerate(columns) if index
    ])
    first = lineage.classify(roles, context, target)
    assert [r.outcome_proxy for r in first.assignments] == [None, None, False]
    # Explicit, supplied directed lineage supports a positive classification, including transitive paths.
    for index, (source, dest) in enumerate([(ids[0], ids[1]), (ids[1], ids[2])]):
        context.relationships[f"r{index}"] = Relationship(
            value="Derived measurement", kind="derivation", status="reviewed",
            source={"kind": "review", "id": uuid4(), "author_id": uuid4()},
            participants=[{"kind": "column", "id": source, "role": "from"}, {"kind": "column", "id": dest, "role": "to"}],
        )
    second = lineage.classify(roles, context, target)
    assert [r.outcome_proxy for r in second.assignments] == [True, True, False]
    assert set(second.assignments[1].proxy_evidence.references) == {"relationships.r0", "relationships.r1"}
    # Reordering and opposite model votes do not change the accepted classification.
    reordered = ContextContent.model_validate(context.model_dump())
    reordered.columns = dict(reversed(list(reordered.columns.items())))
    for role in roles.assignments:
        role.outcome_proxy = not role.outcome_proxy
    assert lineage.classify(roles, reordered, target) == second
    context.relationships["r0"].status = "unresolved"
    assert lineage.classify(roles, context, target).assignments[1].outcome_proxy is None


def test_unverified_lineage_cannot_enter_diagnostics():
    from test_analysis import scripted

    from backend.analysis.contracts import CausalAccount, Frame, Treatments

    context = empty_content(uuid4(), ["group", "score", "ambiguous"])
    columns = [c.identity.model_dump() for c in context.columns.values()]
    target = scripted(Target, "", {"column_identities": columns}, "test")
    treatments = scripted(Treatments, "", {"column_identities": columns}, "test")
    roles = Roles(**decision(), assignments=[RoleAssignment(
        variable=variable(columns[2]), roles=["confounder"], timing="before_treatment",
        timing_evidence=claim(), account=claim(), outcome_proxy=None, reason="Unresolved lineage",
    )])
    frame = Frame(target=target, treatments=treatments, roles=roles,
        causal=CausalAccount(**decision(), identification="Fixture"), graph_hash="fixed")
    raw, manifest = domain.projection(frame, context, b"group,score,ambiguous\n1,999,888\n")
    rows = list(csv.reader(io.StringIO(raw.decode())))
    assert rows[1:] == [["1"]]
    assert rows[0] == treatments.treatments[0].variable.column_ids
    assert len(manifest["columns"]) == 1
