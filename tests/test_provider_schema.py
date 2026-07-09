"""Compact provider decoding preserves the authoritative local scientific gates."""

from uuid import UUID

import pytest
from historical_analysis.work import ColumnScreen, RelationshipWork

from backend.analysis import contracts, domain, provider
from backend.context.contracts import ContextContent, DatasetStatement, SourceReference


@pytest.mark.parametrize("schema", [
    ColumnScreen, contracts.Target, contracts.Treatments, contracts.Roles, RelationshipWork,
    contracts.CausalAccount, contracts.DiagnosticAction, contracts.Plan, contracts.Program,
    contracts.CodeReview, contracts.Conclusion,
])
def test_supplied_reference_catalogue_does_not_multiply_provider_claim_schemas(schema):
    common = {"column_ids": [f"c{i:012x}" for i in range(8)], "_assumption_namespace": ["probe_"]}
    small = provider.provider_schema(schema, {**common, "_supplied_references": ["question"]})
    large = provider.provider_schema(schema, {
        **common, "_supplied_references": ["question", *[f"e{i:012x}" for i in range(200)]],
    })
    assert small == large


@pytest.mark.parametrize(("basis", "refs", "expected"), [
    ("supplied", ["question"], True),
    ("supplied", ["dataset.population.direct"], True),
    ("supplied", [], False),
    ("supplied", ["dataset.population.interpretation"], False),
    ("supplied", ["dataset.population.unresolved"], False),
    ("supplied", ["dataset.population.withdrawn"], False),
    ("supplied", ["decision:target"], False),
    ("supplied", ["assumption:working"], False),
    ("supplied", ["invented.reference"], False),
    ("assumption", ["question"], False),
    ("assumption", ["assumption:working"], True),
    ("unknown", [], True),
])
def test_local_claim_authority_is_enforced_independently_of_provider_schema(basis, refs, expected):
    source = SourceReference(kind="message", id=UUID(int=1), author_id=UUID(int=2))
    context = ContextContent()
    context.dataset.population = {
        "direct": DatasetStatement(value="Direct source", source=source, status="reviewed"),
        "interpretation": DatasetStatement(value="Inference", source=source, basis="interpretation"),
        "unresolved": DatasetStatement(value="Unresolved source", source=source, status="unresolved"),
        "withdrawn": DatasetStatement(value="Withdrawn source", source=source, status="withdrawn"),
    }
    decision = contracts.Target(
        status="needs_input", summary="Synthetic authority check", changes="None", permits="None",
        unresolved=["No scientific decision is requested."],
        measurement=contracts.Claim(statement="Synthetic claim", basis=basis, references=refs),
        assumptions=[contracts.Assumption(id="working", statement="Working premise", rationale="Test", affects=["measurement"])],
    )
    gate = next(item for item in domain.checks(decision, context, {"target": {}})
                if item.rule == "evidence_authority_and_dependencies")
    assert gate.passed is expected, gate.detail
