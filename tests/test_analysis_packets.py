"""Provider constraints and scoped packets preserve their scientific dependencies."""

from backend.analysis import packets, provider
from backend.analysis.contracts import Target


def test_target_schema_limits_each_claim_to_one_compatible_source():
    schema = provider.provider_schema(Target, {
        "_authority_policy": ["claim-authority-19"], "_source_fields": ["target"],
        "_source:measurement": ["method"], "_source:unit": ["observations"],
        "_source:requested_effect": ["question"],
    })
    def object_shape(node):
        return next((item for item in node.get("anyOf", []) if item.get("type") == "object"), node)
    for field, refs in (("measurement", ["method"]), ("unit", ["observations"])):
        claim = object_shape(schema["properties"][field])["properties"]
        assert claim["references"]["maxItems"] == 1
        assert claim["references"]["items"]["enum"] == refs
        assert claim["rule"]["enum"] == ["source", "unknown"]
    timing = object_shape(schema["properties"]["timing"])["properties"]
    assert timing["references"]["maxItems"] == 0
    assert timing["rule"]["enum"] == ["unknown"]
    effect = schema["properties"]["requested_effects"]["items"]["properties"]["question"]["properties"]
    assert effect["references"]["items"]["enum"] == ["question"]


def test_provider_claims_keep_compact_shape_and_local_authority_validation():
    schema = provider.provider_schema(Target, {
        "_supplied_references": ["question", "e0"],
        "_assumption_namespace": ["target_"],
    })
    claims = schema["properties"]["measurement"]["anyOf"]
    assert {item["type"] for item in claims} == {"object", "null"}
    claim = next(item for item in claims if item["type"] == "object")
    assert claim["properties"]["basis"]["enum"] == ["supplied", "deduced", "assumption", "unknown"]
    assert claim["properties"]["references"]["items"] == {"type": "string"}
    assumption = schema["properties"]["assumptions"]["items"]
    assert assumption["properties"]["id"]["pattern"].startswith("^target_")


def test_empty_retrieval_catalogue_is_closed_at_provider_boundary():
    from historical_analysis.work import ColumnScreen
    schema = provider.provider_schema(ColumnScreen, {"requested_keys": []})
    assert schema["properties"]["requested_keys"]["maxItems"] == 0


def test_role_scope_is_required_and_selects_only_accepted_effects_and_treatments():
    from backend.analysis import references
    from backend.analysis.contracts import Roles
    payload = {"column_identities": [{"column_id": "source-treatment"}], "decisions": {
        "treatments": {"effects": [{"id": "primary"}], "treatments": [{"variable": {"id": "source-treatment"}}]}}}
    _, _, choices = references.selections(payload)
    schema = provider.provider_schema(Roles, choices)
    role = schema["properties"]["assignments"]["items"]
    assert {"effect_ids", "treatment_ids", "time_scope"} <= set(role["required"])
    assert role["properties"]["effect_ids"]["items"]["enum"] == ["primary"]
    assert role["properties"]["treatment_ids"]["items"]["enum"] == [references.handle("c", "source-treatment")]
    assert role["properties"]["effect_ids"]["minItems"] == 1
    assert not {"outcome_proxy", "proxy_evidence"} & role["properties"].keys()


def test_operational_keys_survive_reduction_but_inert_identifiers_do_not():
    roles = [{"variable": {"column_ids": [identity]}, "roles": ["identifier"], "operational_use": uses}
             for identity, uses in [("unit", ["unit_key"]), ("inert", [])]]
    prior = {"roles": {"assignments": roles, "assumptions": []}}
    assert packets.active(prior) == {"unit"}
    assert packets.decisions(prior)["roles"]["assignments"] == roles[:1]


def test_cross_treatment_dispositions_remain_with_the_selected_variable():
    roles = [{"variable": {"column_ids": [identity]}, "roles": ["irrelevant"]}
             for identity in ("treatment", "unrelated")]
    prior = {"treatments": {"treatments": [{"variable": {"column_ids": ["treatment"]}}]},
             "roles": {"assignments": roles, "assumptions": []}}
    assert packets.decisions(prior)["roles"]["assignments"] == roles[:1]


def test_accepted_claim_retrieves_exact_remote_source_without_unrelated_context():
    from types import SimpleNamespace
    from uuid import uuid4

    from historical_analysis.runtime import Runtime

    from backend.context.contracts import SourceReference, Statement, empty_content

    context = empty_content(uuid4(), ["target", "remote", "irrelevant"])
    ids = {item.identity.name: cid for cid, item in context.columns.items()}
    source = SourceReference(kind="message", id=uuid4(), author_id=uuid4())
    for name in ("remote", "irrelevant"):
        context.columns[ids[name]].definition["meaning"] = Statement(value=name, source=source)
    required = f"columns.{ids['remote']}.definition.meaning"
    saved = {"decision:target": {"outcome": {"column_ids": [ids["target"]]},
                                "measurement": {"references": [required]}}}
    events = []

    def remember(key, value, **kwargs):
        saved[key] = value

    runtime = Runtime.__new__(Runtime)
    runtime.context = context
    runtime.h = SimpleNamespace(saved=saved, remember=remember, event=lambda *args: events.append(args))
    assert runtime.inspect("causal")
    scoped = runtime.semantic_context("causal")
    assert set(scoped.columns) == {ids["target"], ids["remote"]}
    assert saved["dependency_evidence:causal"] == [required]
    assert events[0][-1] == {"keys": [required], "reason": "accepted_dependency", "model_calls": 0}
    assert runtime.inspect("causal") and len(events) == 1


def test_only_offered_assumption_ledger_ids_can_be_normalized():
    offered = {"target": {"assumptions": [{"id": "accepted"}]},
               "correction": {"summary": "Rejected assumption:invented"}}
    assert packets.assumption_ids(offered) == {"accepted"}

