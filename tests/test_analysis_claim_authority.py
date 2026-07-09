"""Frozen-source authority is distinct from model interpretation and causal reliance."""

import json
from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest
from test_scientific_v8 import study

from backend.analysis import authority, domain, lineage, packets, provider, references, registration
from backend.analysis.contracts import CausalAccount, Claim, Plan, Roles, Target, Treatments
from backend.context.contracts import DatasetStatement, Relationship, SourceReference, Statement


@pytest.fixture(autouse=True)
def isolated():
    yield


SOURCE = SourceReference(kind="message", id=UUID(int=101), author_id=UUID(int=102))


def supplied(text):
    return Statement(value=text, basis="source_statement", status="reviewed", source=SOURCE)


@pytest.mark.parametrize("field", ["measurement", "observations"])
def test_target_unit_uses_the_source_category_exposed_to_the_model(field):
    context, _, target, _, _ = study()
    for category in ("measurement", "observations"):
        getattr(context.dataset, category)["unit"] = DatasetStatement(
            value="Each row represents one participant.", source=SOURCE,
            basis="source_statement", status="reviewed",
        )
    ref = f"dataset.{field}.unit"
    decoded = target.model_copy(update={"unit": Claim.model_validate(
        authority.proposal_decoder()(proposal("A participant", refs=[ref]))
    )})
    admitted, _ = authority.admit(decoded, context, {}, "Question", "target")
    record = next(item for item in packets.evidence_records(context) if item["key"] == ref)
    expected = field == "observations"
    assert ("unit" in record["source_predicates"]) is expected
    assert (admitted.unit.admission.status == "admitted") is expected
    if expected:
        assert admitted.unit.statement == "Each row represents one participant."


def fixture():
    context, ids, target, treatments, roles = study()
    for name, cid in ids.items():
        context.columns[cid].definition[f"def_{name}"] = supplied(f"Exact documented measurement {name}.")
    role = roles.assignments[0]
    role.variable.meaning.references = [f"columns.{ids['c']}.definition.def_c"]
    return context, ids, {"target": target.model_dump(), "treatments": treatments.model_dump()}, roles


def proposal(text, rule="source", refs=(), obligation=""):
    return {
        "statement": text,
        "rule": rule,
        "references": list(refs),
        "prerequisites": [],
        "required_obligation": obligation,
    }


def role_proposal(roles, account, timing=None):
    value = roles.model_dump(mode="json")
    role = value["assignments"][0]
    role["variable"]["meaning"] = proposal(
        "Model paraphrase must not replace source", refs=role["variable"]["meaning"]["references"]
    )
    role["account"] = account
    role["timing_evidence"] = timing or proposal("Unknown timing", "unknown")
    return Roles.model_validate(authority.proposal_decoder()(value))


def relation(context, identity, left, right, kind="causal_influence", **kwargs):
    context.relationships[identity] = Relationship(
        value="Documented typed relationship",
        kind=kind,
        participants=[
            {"kind": "column", "id": left, "role": "from"},
            {"kind": "column", "id": right, "role": "to"},
        ],
        basis="source_statement",
        status="reviewed",
        source=SOURCE,
        directness="direct" if kind == "causal_influence" else "unspecified",
        **kwargs,
    )
    return "relationships." + identity


def test_actual_wide17_purchase_modifier_loses_supplied_authority_without_changing_prior_science():
    historical = json.loads(Path("tests/fixtures/authority_purchase_modifier.json").read_text())
    context, ids, prior, roles = fixture()
    source = Statement.model_validate(historical["source"])
    ref = f"columns.{ids['c']}.definition.def_c"
    context.columns[ids["c"]].definition["def_c"] = source
    roles.assignments[0].roles = historical["proposed_roles"]
    raw = proposal(historical["original_account"]["statement"], refs=[ref])
    prior["causal"] = {"edges": [[ids["t"], ids["y"]]], "adjustment": []}
    prior["plan"] = {"estimator": "Frozen difference in means", "adjustment": []}
    before = deepcopy(prior)
    result, _ = authority.admit(role_proposal(roles, raw), context, prior, "Fixed requested effect", "roles")
    role = result.assignments[0]
    assert role.proposed_roles == ["effect_modifier"] and role.roles == ["unknown"]
    assert role.coverage_status == "assessed" and role.operational_use == []
    assert role.account.basis == "unknown" and role.account.admission.status == "candidate"
    entry = next(item for item in result.assumptions if item.id == role.account.admission.ledger_id)
    assert (
        entry.status == "candidate"
        and entry.original_proposal["statement"] == historical["original_account"]["statement"]
    )
    assert role.account.admission.sources[0].statement == source.value
    assert prior == before
    authority.validate(result, prior, phase="roles")


def test_source_cardinality_failure_reaches_bounded_correction_without_replaying_claim():
    context, ids, prior, roles = fixture()
    ref = f"columns.{ids['c']}.definition.def_c"
    context.columns[ids['c']].units["unit"] = supplied("points")
    value = role_proposal(roles, proposal("Unestablished scientific role", "unknown"))
    raw = proposal("Combined interpretation " + "x" * 5000, refs=[ref, f"columns.{ids['c']}.units.unit"])
    value.assignments[0].variable.meaning = Claim.model_validate(authority.proposal_decoder()(raw))
    result, _ = authority.admit(value, context, prior, "Fixed question", "roles")
    claim = result.assignments[0].variable.meaning
    assert claim.admission.status == "candidate"
    assert "exactly one reference" in claim.admission.reason
    feedback = references.correction(result.model_dump(mode="json"), [{"rule": "scientific_authority", "passed": False, "detail": "Missing source authority"}], "saved-attempt")
    entry = next(item for item in feedback["claim_feedback"] if item["path"].endswith("variable.meaning"))
    assert "selected 2" in entry["detail"]
    assert "rejected_fields" not in feedback
    assert "Combined interpretation" not in json.dumps(feedback)
    assert len(feedback["claim_feedback"]) <= 12


@pytest.mark.parametrize("name", ["randomized_group", "private_z91"])
@pytest.mark.parametrize("reference", ["question", "definition", "process"])
def test_assignment_mechanism_requires_compatible_frozen_process_source_independent_of_names(name, reference):
    context, ids, prior, _ = fixture()
    context.columns[ids["t"]].identity.name = name
    process = DatasetStatement(
        value="The study administrator allocated these categories by a documented lottery.",
        source=SOURCE,
        basis="source_statement",
        status="reviewed",
        column_ids=[ids["t"]],
    )
    context.dataset.study_process["allocation"] = process
    ref = {
        "question": "question",
        "definition": f"columns.{ids['t']}.definition.def_t",
        "process": "dataset.study_process.allocation",
    }[reference]
    value = Treatments.model_validate(prior["treatments"])
    proposed = authority.proposal_decoder()(
        {"claim": proposal("I recognize this dataset; assignment was randomized.", refs=[ref])}
    )["claim"]
    value.assignments[0].mechanism = Claim.model_validate(proposed)
    result, _ = authority.admit(
        value, context, {"target": prior["target"]}, "Estimate the randomized group effect", "treatments"
    )
    mechanism = result.assignments[0].mechanism
    if reference == "process":
        assert mechanism.basis == "supplied" and mechanism.statement == process.value
    else:
        assert mechanism.basis == "unknown" and mechanism.admission.status == "candidate"


@pytest.mark.parametrize("invent_arrow", [False, True])
def test_relationship_record_preserves_association_without_causal_arrow_or_irrelevance(invent_arrow):
    from historical_analysis.work import RelationshipWork

    context, ids, prior, _ = fixture()
    context.relationships["shared"] = Relationship(
        value="These columns share the same bookkeeping record.",
        kind="association",
        participants=[
            {"kind": "column", "id": ids["c"], "role": "participant"},
            {"kind": "column", "id": ids["t"], "role": "participant"},
        ],
        basis="source_statement",
        status="reviewed",
        source=SOURCE,
    )
    value = RelationshipWork.model_validate(
        authority.proposal_decoder()(
            {
                "status": "ready",
                "summary": "Coverage",
                "changes": "One disposition",
                "permits": "Graph review",
                "relationships": [
                    {
                        "reference": "relationships.shared",
                        "effect_ids": ["primary"],
                        "disposition": "applicable" if invent_arrow else "not_applicable",
                        "scope": "The frozen primary effect",
                        "source_id": ids["c"] if invent_arrow else "",
                        "target_id": ids["t"] if invent_arrow else "",
                        "rationale": proposal(
                            "These variables are scientifically irrelevant",
                            "relationship_record",
                            ["relationships.shared"],
                        ),
                    }
                ],
            }
        )
    )
    result, _ = authority.admit(value, context, prior, "Question", "relationship")
    rationale = result.relationships[0].rationale
    if invent_arrow:
        assert rationale.admission.status == "candidate"
    else:
        assert (
            rationale.admission.status == "admitted"
            and rationale.statement == context.relationships["shared"].value
        )
        assert result.relationships[0].source_id == result.relationships[0].target_id == ""
        authority.validate(result, prior, phase="relationship")


def test_single_column_relationship_does_not_authorize_a_derived_multi_column_role():
    context, ids, prior, roles = fixture()
    zt = relation(context, "zt", ids["c"], ids["t"])
    zy = relation(context, "zy", ids["c"], ids["y"])
    roles.assignments[0].variable.id = "derived_measurement"
    roles.assignments[0].variable.column_ids = [ids["c"], ids["y"]]
    roles.assignments[0].variable.derivation = "A declared combination, not a source-equivalent measurement."
    result, _ = authority.admit(
        role_proposal(roles, proposal("Common cause", "common_cause", [zt, zy])),
        context,
        prior,
        "Question",
        "roles",
    )
    assert result.assignments[0].account.admission.status == "candidate"
    assert result.assignments[0].roles == ["unknown"]
    assert result.assignments[0].variable.meaning.admission.status == "candidate"


@pytest.mark.parametrize("binding", ["exact", "missing", "component", "superset"])
def test_group_measurement_definition_requires_exact_explicit_dataset_source_binding(binding):
    context, ids, prior, roles = fixture()
    group = [ids["c"], ids["y"]]
    bounds = {"exact": group, "missing": [], "component": group[:1], "superset": [*group, ids["t"]]}
    source = DatasetStatement(
        value="These two columns jointly encode one declared measurement using the specified combination.",
        column_ids=bounds[binding],
        source=SOURCE,
        basis="source_statement",
        status="reviewed",
    )
    context.dataset.measurement["joint_definition"] = source
    variable = roles.assignments[0].variable
    variable.id, variable.column_ids, variable.derivation = (
        "joint_measurement",
        group,
        "The specified combination.",
    )
    variable.meaning.references = ["dataset.measurement.joint_definition"]
    result, _ = authority.admit(
        role_proposal(roles, proposal("Optional role", "unknown")), context, prior, "Question", "roles"
    )
    meaning = result.assignments[0].variable.meaning
    assert (meaning.admission.status == "admitted") is (binding == "exact")
    if binding == "exact":
        assert meaning.statement == source.value and meaning.basis == "supplied"
    assert result.assignments[0].roles == ["unknown"]  # Measurement meaning never substitutes a causal role.


@pytest.mark.parametrize("name", ["randomized_treatment", "income", "opaque_q79"])
def test_familiar_or_private_names_and_question_text_do_not_supply_role_premises(name):
    context, ids, prior, roles = fixture()
    context.columns[ids["c"]].identity.name = name
    roles.assignments[0].roles = ["effect_modifier"]
    result, _ = authority.admit(
        role_proposal(roles, proposal("This must modify the effect", refs=["question"])),
        context,
        prior,
        "Estimate an effect, assuming nothing about modification",
        "roles",
    )
    assert result.assignments[0].account.admission.status == "candidate"
    assert result.assignments[0].roles == ["unknown"]


@pytest.mark.parametrize(
    "defect", [None, "proposed", "interpretation", "qualified", "unknown_ref", "conflict"]
)
def test_exact_source_restoration_is_scope_and_status_sensitive(defect):
    context, ids, prior, roles = fixture()
    source = context.columns[ids["c"]].definition["def_c"]
    source.value = "Exact frozen content, not the model paraphrase."
    if defect == "proposed":
        source.status = "proposed"
    if defect == "interpretation":
        source.basis = "interpretation"
    if defect == "qualified":
        source.qualifications = "Conditional working premise only."
    if defect == "conflict":
        source.conflicts_with = [f"columns.{ids['y']}.definition.def_y"]
    if defect == "unknown_ref":
        roles.assignments[0].variable.meaning.references = ["invented.source"]
    value = role_proposal(roles, proposal("Optional role", "unknown"))
    result, _ = authority.admit(value, context, prior, "Question", "roles")
    meaning = result.assignments[0].variable.meaning
    if defect is None:
        assert meaning.statement == source.value and meaning.basis == "supplied"
        assert meaning.admission.original_proposal["statement"] != meaning.statement
    else:
        assert meaning.basis == "unknown" and meaning.admission.status != "admitted"
        if defect == "qualified":
            assert meaning.admission.sources[0].qualifications == source.qualifications


def test_typed_common_cause_and_temporal_rules_bind_exact_endpoints_and_scope():
    context, ids, prior, roles = fixture()
    zt = relation(context, "zt", ids["c"], ids["t"])
    zy = relation(context, "zy", ids["c"], ids["y"])
    before = relation(context, "before", ids["c"], ids["t"], "temporal_order")
    value = role_proposal(
        roles,
        proposal("Model embellishment: very large effects", "common_cause", [zt, zy]),
        proposal("Before assignment", "temporal_order", [before]),
    )
    result, _ = authority.admit(
        value,
        context,
        prior,
        "Question",
        "roles",
        scope={"event": {"kind": "point", "treatment_ids": [ids["t"]]}},
    )
    role = result.assignments[0]
    assert role.roles == ["confounder"] and role.account.basis == "deduced"
    assert "very large" not in role.account.statement
    assert role.account.admission.scope.treatment_ids == [ids["t"]]
    assert role.account.admission.scope.assignment_event_key
    assert role.timing == "before_treatment" and role.timing_evidence.basis == "deduced"
    authority.validate(result, prior, phase="roles")
    # A different directed endpoint cannot be repaired from a label or similar meaning.
    context.relationships["zt"].participants[1].id = ids["y"]
    rejected, _ = authority.admit(value, context, prior, "Question", "roles")
    assert rejected.assignments[0].roles == ["unknown"]


def test_question_and_unrelated_column_cannot_authorize_target_measurement():
    context, ids, prior, _ = fixture()
    target = Target.model_validate(prior["target"])
    raw = target.model_dump(mode="json")
    # Decode this isolated claim using the same raw boundary as a full target.
    for ref in ("question", f"columns.{ids['c']}.definition.def_c"):
        raw["measurement"] = authority.proposal_decoder()(
            {"claim": proposal("Outcome measurement", refs=[ref])}
        )["claim"]
        result, _ = authority.admit(
            Target.model_validate(raw), context, {}, "A question is not a study fact", "target"
        )
        assert result.measurement.admission.status == "candidate"
        assert result.measurement.admission.scope.variable_ids == [ids["y"]]


def causal(prior, ids, *, obligation="consistency", rule="working_assumption"):
    return CausalAccount.model_validate(
        authority.proposal_decoder()(
            {
                "status": "ready",
                "summary": "Conditional specification",
                "changes": "Declared premises",
                "permits": "Conditional interpretation",
                "identification": "Conditional",
                "nodes": [],
                "edges": [],
                "identifications": [
                    {
                        "effect_id": "primary",
                        "strategy": "backdoor",
                        "adjustment": [],
                        "evidence": proposal(
                            "The backdoor estimand is identified under the declared graph",
                            rule,
                            obligation="backdoor",
                        ),
                        "obligations": [
                            {
                                "name": obligation,
                                "claim": proposal(
                                    "Explicit conditional premise", rule, obligation=obligation
                                ),
                            }
                        ],
                    }
                ],
            }
        )
    )


@pytest.mark.parametrize(
    "phase,frozen,obligation,active",
    [
        ("causal", False, "consistency", True),
        ("causal", False, "comparable_outcome_measurement", True),
        ("causal", True, "comparable_outcome_measurement", False),
        ("roles", False, "consistency", False),
        ("causal", True, "consistency", False),
        ("causal", False, "effect_modification", False),
    ],
)
def test_only_necessary_scoped_causal_obligations_can_rely_on_assumptions(phase, frozen, obligation, active):
    context, ids, prior, _ = fixture()
    result, _ = authority.admit(
        causal(prior, ids, obligation=obligation), context, prior, "Question", phase, frozen=frozen
    )
    claim = result.identifications[0].obligations[0].claim
    entry = next(item for item in result.assumptions if item.id == claim.admission.ledger_id)
    assert (entry.status == "relied_upon") is active
    assert entry.scope.effect_ids == ["primary"] and entry.required_obligation == obligation
    if active:
        authority.validate(result, prior, phase="causal")
    else:
        with pytest.raises(ValueError):
            authority.validate(result, prior, phase="causal", frozen=frozen)


def test_explicit_conflict_cannot_be_laundered_through_working_assumption():
    context, ids, prior, _ = fixture()
    ref = f"columns.{ids['c']}.definition.def_c"
    context.columns[ids["y"]].definition["def_y"].conflicts_with = [ref]
    raw = causal(prior, ids).model_dump(mode="json")
    raw["identifications"][0]["obligations"][0]["claim"]["admission"]["original_proposal"]["references"] = [
        ref
    ]
    result, _ = authority.admit(CausalAccount.model_validate(raw), context, prior, "Question", "causal")
    claim = result.identifications[0].obligations[0].claim
    assert claim.admission.status == "rejected" and claim.basis == "unknown"
    assert (
        next(item for item in result.assumptions if item.id == claim.admission.ledger_id).status == "rejected"
    )


def test_provider_cannot_author_basis_admission_or_ledger_status_and_schema_stays_flat():
    for forbidden in ({"basis": "supplied"}, {"admission": {}}, {"status": "relied_upon"}):
        with pytest.raises(ValueError):
            authority.proposal_decoder()({"claim": {**proposal("Text"), **forbidden}})
    schema = provider.provider_schema(
        Treatments,
        {
            "_authority_policy": [authority.POLICY],
            "_claim_refs": ["question"],
            "_json_schema": ["treatments"],
        },
    )
    claim = schema["$defs"]["Claim"]
    assert set(claim["properties"]) == {
        "statement",
        "rule",
        "references",
        "prerequisites",
        "required_obligation",
    }
    assert not {"ClaimAdmission", "ClaimScope", "SourceExcerpt"} & schema["$defs"].keys()
    assert "anyOf" not in claim and "basis" not in claim["properties"]


def test_historical_claim_serialization_and_registration_remain_unchanged():
    old = {
        "statement": "Historical source claim",
        "basis": "supplied",
        "references": ["question"],
        "prerequisites": [],
    }
    assert Claim.model_validate(old).model_dump(mode="json") == old
    context, _, prior, roles = fixture()
    assert registration.normalize(roles, context, prior, "roles_", scientific_version=19) == (roles, [])
    assert (
        domain.references({"admission": {"original_proposal": {"references": ["invented.source"]}}}) == set()
    )


def test_reference_hydration_preserves_accepted_binding_and_nested_raw_proposal_exactly():
    from historical_analysis.work import role_reference_decoder

    context, _, prior, roles = fixture()
    accepted, _ = authority.admit(
        role_proposal(roles, proposal("Optional role", "unknown")), context, prior, "Question", "roles"
    )
    variable = accepted.assignments[0].variable.model_dump(mode="json")
    raw = roles.model_dump(mode="json")
    raw["assignments"][0].update(
        variable={"id": variable["id"]},
        account=proposal("No role established", "unknown"),
        timing_evidence=proposal("Unknown timing", "unknown"),
    )
    decoded = authority.proposal_decoder(base_decoder=role_reference_decoder([variable]))(raw)
    assert decoded["assignments"][0]["variable"] == variable
    assert "basis" not in variable["meaning"]["admission"]["original_proposal"]


@pytest.mark.parametrize("path_kind", ["none", "known", "qualified", "proposed", "baseline"])
def test_harness_lineage_admission_proves_access_disposition_without_claiming_unknown_absence(path_kind):
    context, ids, prior, roles = fixture()
    timing = None
    if path_kind == "baseline":
        zt = relation(context, "zt", ids["c"], ids["t"])
        zy = relation(context, "zy", ids["c"], ids["y"])
        before = relation(context, "before", ids["c"], ids["t"], "temporal_order")
        account = proposal("Common cause", "common_cause", [zt, zy])
        timing = proposal("Before assignment", "temporal_order", [before])
    else:
        account = proposal("No role established", "unknown")
        if path_kind != "none":
            relation(context, "derivation", ids["y"], ids["c"], "derivation")
            if path_kind == "qualified":
                context.relationships["derivation"].qualifications = "Only within the follow-up window."
            if path_kind == "proposed":
                context.relationships["derivation"].status = "proposed"
    value, _ = authority.admit(role_proposal(roles, account, timing), context, prior, "Question", "roles")
    classified = lineage.classify(
        value, context, Target.model_validate(prior["target"]), scientific_version=19
    )
    role = classified.assignments[0]
    assert role.outcome_proxy is (
        True if path_kind == "known" else False if path_kind == "baseline" else None
    )
    assert role.proxy_evidence.admission.predicate == "diagnostic_lineage"
    assert role.proxy_evidence.admission.scope.variable_ids == [ids["c"]]
    assert role.proxy_evidence.basis == "deduced"
    authority.validate(classified, prior, phase="roles")
    if path_kind in {"none", "qualified", "proposed"}:
        assert "withheld" in role.proxy_evidence.statement
    if path_kind == "qualified":
        assert (
            role.proxy_evidence.admission.sources[0].qualifications
            == context.relationships["derivation"].qualifications
        )


def test_plan_cannot_promote_or_cite_a_new_inference_premise_after_causal_freeze():
    context, ids, prior, _ = fixture()
    accepted, _ = authority.admit(causal(prior, ids), context, prior, "Question", "causal")
    prior["causal"] = accepted.model_dump(mode="json")
    plan = Plan(
        status="ready",
        summary="Fixed plan",
        changes="None",
        permits="Execution",
        effects=[],
        population="Fixed",
        missingness="Fixed",
        exclusions="Fixed",
        multiplicity="Fixed",
        reporting="Fixed",
        limitations=["Conditional on the frozen causal premises."],
        assumptions=[
            {
                "id": "new_inference",
                "statement": "A new premise",
                "rationale": "Invented after causal freeze",
                "affects": ["primary"],
            }
        ],
        citations=["assumption:new_inference"],
    )
    admitted, _ = authority.admit(plan, context, prior, "Question", "plan", frozen=True)
    assert admitted.assumptions[0].status == "candidate"
    gates = domain.checks(admitted, context, prior, scientific_version=19)
    assert any(gate.rule == "evidence_authority_and_dependencies" and not gate.passed for gate in gates)
