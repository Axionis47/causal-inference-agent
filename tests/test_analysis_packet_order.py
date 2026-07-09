"""Object-map persistence order must not alter immutable semantic packets."""

from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import pytest
from historical_analysis import work
from historical_analysis.runtime import Runtime

from backend.analysis import packets, provider, references
from backend.analysis.contracts import Treatments
from backend.context.contracts import SourceReference, Statement, empty_content
from backend.context.service import fingerprint


@pytest.fixture(autouse=True)
def isolated():
    yield


def fixture():
    context = empty_content(UUID(int=77), ["assignment", "outcome", "historical"])
    ids = {column.identity.name: cid for cid, column in context.columns.items()}
    context.columns = {cid: context.columns[cid] for cid in sorted(context.columns, reverse=True)}
    # Deliberately retain meaningful nonalphabetical effect and contrast order.
    effects = [{"id": identity, "priority": priority, "treatment_ids": [ids["assignment"]],
                "comparison": {"kind": "static", "static": {"terms": [
                    {"weight": -1, "value_ids": ["control"]}, {"weight": 1, "value_ids": ["active"]},
                ]}}} for identity, priority in [("z_primary", "primary"), ("a_secondary", "secondary")]]
    prior = {"target": {"outcome": {"column_ids": [ids["outcome"]]}, "requested_effects": deepcopy(effects)},
             "treatments": {"treatments": [{"variable": {"id": ids["assignment"], "column_ids": [ids["assignment"]]}}],
                            "effects": effects}}
    runtime = Runtime.__new__(Runtime)
    runtime.context, runtime.version = context, SimpleNamespace(id=UUID(int=77))
    runtime.run = SimpleNamespace(question="Estimate the fixed ordered contrasts.", context_revision_id=UUID(int=88),
                                  inputs={"prior_exposure": "No target data inspected."})
    runtime.h = SimpleNamespace(saved={"decision:" + key: value for key, value in prior.items()})
    return context, prior, runtime


def make_packet(kind, context, prior, runtime):
    if kind == "shared":
        return runtime.scoped_packet(context, set(context.columns))
    return work.packet(context, set(context.columns), prior, runtime.run.question)


def provider_request(packet):
    offered, aliases, choices = references.selections({key: value for key, value in packet.items() if not key.startswith("_")})
    choices.update(_explicit_fields=["treatments"], _json_schema=["treatments"])
    return {"payload": offered, "aliases": aliases, "schema": provider.provider_schema(Treatments, choices),
            "schema_format": "response_json_schema", "output_limit": 8192,
            "thinking_config": {"thinking_budget": 2048, "include_thoughts": False}}


@pytest.mark.parametrize("kind", ["shared", "worker"])
def test_column_map_reordering_preserves_packet_hash_provider_request_and_source_positions(kind):
    context, prior, runtime = fixture()
    before = make_packet(kind, context, prior, runtime)
    reordered = context.model_copy(deep=True)
    reordered.columns = dict(reversed(list(reordered.columns.items())))
    after = make_packet(kind, reordered, prior, runtime)

    assert before == after
    assert fingerprint(before) == fingerprint(after)
    assert provider_request(before) == provider_request(after)
    assert fingerprint(provider_request(before)) == fingerprint(provider_request(after))
    assert [item["column_id"] for item in before["column_identities"]] == sorted(context.columns)
    assert {item["column_id"]: item["position"] for item in before["column_identities"]} == {
        cid: column.identity.position for cid, column in context.columns.items()
    }
    assert before["decisions"] == prior
    assert [effect["id"] for effect in before["decisions"]["treatments"]["effects"]] == ["z_primary", "a_secondary"]
    assert [term["weight"] for term in before["decisions"]["treatments"]["effects"][0]["comparison"]["static"]["terms"]] == [-1, 1]


@pytest.mark.parametrize("kind", ["shared", "worker"])
def test_canonical_identity_order_does_not_hide_real_source_position_or_contrast_changes(kind):
    context, prior, runtime = fixture()
    before = make_packet(kind, context, prior, runtime)
    changed = context.model_copy(deep=True)
    next(iter(changed.columns.values())).identity.position = 99
    assert fingerprint(make_packet(kind, changed, prior, runtime)) != fingerprint(before)
    prior["treatments"]["effects"][0]["comparison"]["static"]["terms"].reverse()
    assert fingerprint(make_packet(kind, context, prior, runtime)) != fingerprint(before)


@pytest.mark.parametrize("kind", ["shared", "worker"])
@pytest.mark.parametrize("width", [12, 60, 200])
def test_semantic_request_omits_physical_positions_and_unrelated_dataset_width(kind, width):
    context, prior, runtime = fixture()
    selected = set(context.columns)
    column = next(iter(context.columns.values()))
    column.definition["meaning"] = Statement(
        value="This is the recorded assignment.", qualifications="Receipt is not established.",
        source=SourceReference(kind="message", id=UUID(int=89), author_id=UUID(int=90)),
    )

    def scoped_packet(full):
        runtime.context = full
        if kind == "shared":
            return runtime.scoped_packet(packets.scope(full, selected), selected)
        return work.packet(full, selected, prior, runtime.run.question)

    original = scoped_packet(context)
    wider = context.model_copy(deep=True)
    for index, item in enumerate(wider.columns.values()):
        item.identity.position = width - 1 - index
    wider.columns.update(empty_content(UUID(int=91), [f"padding_{index}" for index in range(width - 3)]).columns)
    expanded = scoped_packet(wider)
    saved = deepcopy(expanded)
    request = provider_request(expanded)

    assert request == provider_request(original)
    assert expanded == saved, "Projection must not rewrite the immutable audit packet."
    assert expanded["evidence_scope"]["total_columns"] == width
    assert {item["position"] for item in expanded["column_identities"]} == {width - 1, width - 2, width - 3}
    assert "total_columns" not in request["payload"]["evidence_scope"]
    assert all("position" not in item for item in request["payload"]["column_identities"])
    decoded = references.translate(request["payload"], request["aliases"])
    assert decoded["decisions"] == prior
    assert decoded["evidence_records"][0]["statement"]["qualifications"] == "Receipt is not established."
    assert {item["name"] for item in decoded["column_identities"]} == {"assignment", "outcome", "historical"}
