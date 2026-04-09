"""Deterministic admission of frozen sources and explicitly scoped conditional premises.

Source existence is not source entailment. Only exact compatible source content or
the finite typed rules below can become active authority. Other interpretations
remain inspectable candidates, never silently converted into causal facts.
"""

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, cast

from pydantic import BaseModel

from backend.analysis.contracts import Assumption, ClaimAdmission, ClaimScope, SourceExcerpt
from backend.context.contracts import ContextContent, Relationship, statement_entries
from backend.context.service import fingerprint

RULES = ("source", "relationship_record", "temporal_order", "causal_edge", "common_cause", "mediator_chain", "collider", "accepted_binding", "working_assumption", "unknown")
POLICY = "claim-authority-19"
PENDING = "Awaiting deterministic claim admission."
SOURCE_FIELDS = {
    "columns": {
        "measurement_definition": {"definition", "concept", "construction", "derivation", "codes"},
        "measurement": {"definition", "units", "method", "measurement_source"},
        "unit": {"entity"}, "timing": {"collection_time", "reference_period"},
        "requested_intervention": {"codes", "definition", "determined_by"},
        "mechanism": {"determined_by"},
        "property": {"definition", "codes", "units", "entity", "collection_time", "reference_period", "method", "measurement_source"},
    },
    "dataset": {
        "measurement_definition": {"measurement"}, "population": {"population"},
        "unit": {"observations", "organization"}, "measurement": {"measurement"},
        "mechanism": {"study_process"}, "timing": {"measurement", "study_process"},
        "requested_intervention": {"study_process"},
        "property": {"study_process"},
    },
}


def source_predicates(ref: str, record: Any = None) -> list[str]:
    """Expose the same field eligibility used by admission, without granting authority."""
    if ref == "question" or ref.startswith("dataset.question."):
        return ["requested_effect", "requested_intervention"]
    if isinstance(record, Relationship) and ref.startswith("relationships."):
        return ["mechanism", "requested_intervention"] if record.kind == "assignment_rule" else ["timing"] if record.kind == "temporal_order" else []
    parts = ref.split(".")
    if (parts[0], len(parts)) not in {("columns", 4), ("dataset", 3)}:
        return []
    field = parts[-2]
    return sorted(predicate for predicate, fields in SOURCE_FIELDS[parts[0]].items() if field in fields)


OBLIGATIONS = {
    "backdoor": {"conditional_exchangeability", "no_unmeasured_confounding", "consistency", "positivity", "no_interference", "comparable_outcome_measurement"},
    "source_assignment": {"assignment_validity", "consistency", "positivity", "no_interference", "comparable_outcome_measurement"},
    "did": {"parallel_trends", "no_anticipation", "comparison_support", "consistency", "no_interference", "comparable_outcome_measurement"},
    "discontinuity": {"cutoff_rule", "continuity", "local_randomization", "exclusion", "monotonicity", "relevance", "consistency", "comparable_outcome_measurement"},
    "structural_zero": {"graph_completeness", "consistency"},
}


def proposal_decoder(*, base_decoder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                     allow_empty_defaults: bool = False) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Decode the small provider shape; source adjudication happens after alias reversal."""
    def decode(proposal: dict[str, Any]) -> dict[str, Any]:
        def forbidden(obj: Any) -> None:
            if isinstance(obj, list):
                for item in obj:
                    forbidden(item)
            elif isinstance(obj, dict):
                if any(key in obj for key in ("admission", "proposed_roles", "coverage_status", "admission_reason")):
                    raise ValueError("Claim admissions and effective role coverage are harness-owned.")
                if "basis" in obj or ("status" in obj and obj["status"] in {"candidate", "relied_upon", "rejected"}):
                    raise ValueError("Models cannot author claim authority or assumption-ledger status.")
                for item in obj.values():
                    forbidden(item)
        forbidden(proposal)
        output = base_decoder(proposal) if base_decoder else deepcopy(proposal)

        def walk(obj: Any, path: str = "proposal") -> Any:
            if isinstance(obj, list):
                return [walk(item, f"{path}[{index}]") for index, item in enumerate(obj)]
            if not isinstance(obj, dict):
                return obj
            is_edge = {"source", "target", "explanation"} <= obj.keys()
            is_claim = "statement" in obj and "references" in obj
            if (is_edge or is_claim) and "basis" not in obj:
                if allow_empty_defaults:
                    obj.setdefault("prerequisites", [])
                    if obj.get("rule") != "working_assumption":
                        obj.setdefault("required_obligation", "")
                allowed = ({"source", "target", "explanation"} if is_edge else {"statement"}) | {"rule", "references", "prerequisites", "required_obligation"}
                if set(obj) != allowed or obj["rule"] not in RULES:
                    raise ValueError(f"{path}: a claim proposal requires exactly its statement, rule, references, prerequisites and required_obligation; edges also name source and target. Missing fields: {sorted(allowed - obj.keys())}; unexpected fields: {sorted(obj.keys() - allowed)}. rule must be one of {RULES}.")
                original = deepcopy(obj)
                value = {key: deepcopy(item) for key, item in obj.items() if key not in {"rule", "required_obligation"}}
                value.update(basis="unknown", admission=ClaimAdmission(
                    status="candidate", authority="none", rule=obj["rule"], predicate="pending",
                    scope=ClaimScope(), original_proposal=original, reason=PENDING,
                ).model_dump(mode="json"))
                return value
            # These objects can enter only through the trusted reference decoder.
            # Their saved proposal is audit evidence, not a fresh provider claim.
            return {key: deepcopy(item) if key in {"admission", "original_proposal"} else walk(item, path + "." + key)
                    for key, item in obj.items()}
        return cast(dict[str, Any], walk(output))
    return decode


def latest_ledger(prior: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Accepted decision order owns lifecycle; raw rejected proposals are not inputs."""
    result: dict[str, dict[str, Any]] = {}
    for decision in prior.values():
        if isinstance(decision, dict):
            for entry in decision.get("assumptions", []):
                result[entry["id"]] = deepcopy(entry)
    return result


def established(claim: Any) -> bool:
    admission = getattr(claim, "admission", None)
    return bool(admission and admission.status == "admitted" and (
        admission.authority == "source" and claim.basis == "supplied"
        or admission.authority == "deduction" and claim.basis == "deduced"
    ))


def validate(value: BaseModel, prior: dict[str, Any], *, phase: str, frozen: bool = False) -> None:
    """Active authority is an admission, never a reference's mere existence."""
    raw = value.model_dump(mode="json")
    accepted = latest_ledger(prior)
    ledger = {**accepted, **{item["id"]: item for item in raw.get("assumptions", [])}}
    relied: set[str] = set()

    def walk(obj: Any, path: str = "", inactive: bool = False) -> None:
        if isinstance(obj, list):
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]", inactive)
        elif isinstance(obj, dict):
            if "roles" in obj and "proposed_roles" in obj:
                inactive = set(obj["roles"]) <= {"unknown"} and not obj.get("operational_use")
            claim = {"statement", "basis", "references"} <= obj.keys() or {"source", "target", "explanation", "basis"} <= obj.keys()
            if claim:
                admission = obj.get("admission")
                if not admission or admission["predicate"] == "pending":
                    raise ValueError(f"{path}: current claims require finalized harness-owned admission.")
                if admission["status"] != "admitted":
                    if inactive or re.search(r"\.properties\[\d+\]$", path) or raw.get("status") != "ready":
                        return
                    raise ValueError(f"{path}: an inactive or rejected claim cannot support a ready scientific decision.")
                expected = {"source": "supplied", "deduction": "deduced", "assumption": "assumption"}.get(admission["authority"])
                if expected is None or obj["basis"] != expected:
                    raise ValueError(f"{path}: claim authority must match its admitted rule.")
                if admission["authority"] == "assumption":
                    identity = admission.get("ledger_id")
                    entry = ledger.get(identity)
                    if not entry or entry.get("status") != "relied_upon" or not entry.get("required_obligation") or entry.get("scope") != admission["scope"]:
                        raise ValueError(f"{path}: an assumption needs a relied-upon ledger entry for this exact required obligation and scope.")
                    if frozen and accepted.get(identity) != entry:
                        raise ValueError("A new assumption cannot be promoted after causal freeze.")
                    relied.add(identity)
                return
            for key, item in obj.items():
                if key not in {"admission", "original_proposal", "assumptions"}:
                    walk(item, f"{path}.{key}" if path else key, inactive)
    walk(raw)
    for entry in raw.get("assumptions", []):
        if entry.get("status") is None:
            raise ValueError("Current assumption ledger entries require explicit lifecycle status.")
        if entry["status"] == "relied_upon" and entry["id"] not in relied and accepted.get(entry["id"]) != entry:
            raise ValueError("An unreferenced model premise cannot become relied-upon authority.")
        if entry["status"] == "relied_upon" and phase != "causal" and accepted.get(entry["id"]) != entry:
            raise ValueError("Only the causal specification may adopt a new necessary working assumption.")


def _directed(record: Relationship) -> tuple[str, str] | None:
    starts = [item.id for item in record.participants if item.role == "from"]
    ends = [item.id for item in record.participants if item.role == "to"]
    if len(starts) != 1 or len(ends) != 1 or len(record.participants) != 2:
        return None
    return starts[0], ends[0]


def source_selectable(record: Any) -> bool:
    return record.basis == "source_statement" and record.status == "reviewed" and not any(
        getattr(record, field, "") for field in ("qualifications", "conditions")
    )


def _scope(prior: dict[str, Any], owner: dict[str, Any], scope: dict[str, Any] | None) -> ClaimScope:
    treatments = prior.get("treatments", {})
    owned = scope or {}
    variable = owner.get("variable") or owner.get("outcome") or {}
    return ClaimScope(
        variable_ids=list(variable.get("column_ids", [])) or ([owner["variable_id"]] if owner.get("variable_id") else list(owner.get("treatment_ids", []))),
        effect_ids=list(owner.get("effect_ids", [item["id"] for item in treatments.get("effects", [])])),
        treatment_ids=list(owner.get("treatment_ids", [item["variable"]["id"] for item in treatments.get("treatments", [])])),
        assignment_event_key=owner.get("assignment_event_key") or (fingerprint(owned["event"]) if "event" in owned else None),
    )


def admit[T: BaseModel](
    value: T, context: ContextContent, prior: dict[str, Any], question: str, phase: str,
    *, frozen: bool = False, scope: dict[str, Any] | None = None,
) -> tuple[T, list[str]]:
    """Restore compatible sources, evaluate finite rules, and retain inactive candidates."""
    output = deepcopy(value.model_dump(mode="json"))
    catalog = dict(statement_entries(context))
    prior_ledger = latest_ledger(prior)
    ledger: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    target = prior.get("target", output if phase == "target" else {})
    outcome_id = (target.get("outcome") or {}).get("id")
    treatments = prior.get("treatments", {})
    hosts: dict[str, tuple[str, dict[str, Any], ClaimScope]] = {}
    for index, role in enumerate(output.get("assignments", []) if phase == "roles" else []):
        role_scope = _scope(prior, role, scope)
        hosts[f"assignments[{index}].account"] = ("role", role, role_scope)
        hosts[f"assignments[{index}].timing_evidence"] = ("role_timing", role, role_scope)
    for index, edge in enumerate(output.get("edges", [])):
        edge_scope = _scope(prior, {}, scope)
        edge_scope.variable_ids = [edge["source"], edge["target"]]
        hosts[f"edges[{index}]"] = ("causal_edge", edge, edge_scope)
    for i, identification in enumerate(output.get("identifications", [])):
        effect: dict[str, Any] = next((item for item in treatments.get("effects", []) if item["id"] == identification["effect_id"]), {})
        effect_scope = ClaimScope(effect_ids=[identification["effect_id"]], treatment_ids=effect.get("treatment_ids", []))
        hosts[f"identifications[{i}].evidence"] = ("identification", identification, effect_scope)
        for j, obligation in enumerate(identification.get("obligations", [])):
            hosts[f"identifications[{i}].obligations[{j}].claim"] = ("obligation:" + obligation["name"], identification, effect_scope)
    for i, relationship in enumerate(output.get("relationships", [])):
        selected_effects = set(relationship.get("effect_ids", []))
        relationship_scope = ClaimScope(effect_ids=sorted(selected_effects), treatment_ids=sorted({
            identity for effect in treatments.get("effects", []) if effect["id"] in selected_effects for identity in effect["treatment_ids"]
        }))
        hosts[f"relationships[{i}].rationale"] = ("relationship_record", relationship, relationship_scope)

    def classify(path: str, owner: dict[str, Any]) -> tuple[str, dict[str, Any], ClaimScope]:
        if path in hosts:
            return hosts[path]
        claim_scope = _scope(prior, owner, scope)
        if owner.get("column_ids"):
            claim_scope.variable_ids = list(owner["column_ids"])
        if re.fullmatch(r"requested_effects\[\d+\]\.question", path) or re.fullmatch(r"effects\[\d+\]\.contrast", path):
            return "requested_effect", owner, claim_scope
        if path.endswith(".comparison.evidence") or path.endswith(".change"):
            return "requested_intervention", owner, claim_scope
        if path.endswith(".meaning"):
            return "measurement_definition", owner, claim_scope
        if re.search(r"\.properties\[\d+\]$", path):
            return "property", owner, claim_scope
        return path.rsplit(".", 1)[-1], owner, claim_scope

    def source_compatible(predicate: str, ref: str, claim_scope: ClaimScope) -> bool:
        if predicate not in source_predicates(ref, catalog.get(ref)):
            return False
        if ref == "question" or ref.startswith("dataset.question."):
            return True
        record = catalog[ref]
        if isinstance(record, Relationship):
            participants = {item.id for item in record.participants if item.kind == "column"}
            return bool(claim_scope.variable_ids) and set(claim_scope.variable_ids) <= participants
        parts = ref.split(".")
        if parts[0] == "columns" and len(parts) == 4:
            return claim_scope.variable_ids == [parts[1]]
        bound = set(getattr(catalog[ref], "column_ids", []))
        if predicate == "measurement_definition" or len(claim_scope.variable_ids) > 1 and predicate in {"measurement", "property"}:
            return bool(bound) and bound == set(claim_scope.variable_ids)
        return not bound or bool(claim_scope.variable_ids) and set(claim_scope.variable_ids) <= bound

    def assess(obj: dict[str, Any], path: str, owner: dict[str, Any]) -> None:
        saved = obj.get("admission")
        if saved and saved.get("reason") != PENDING:
            return  # Only the reference decoder can hydrate these accepted objects.
        original = deepcopy(saved["original_proposal"] if saved else obj)
        rule = original.get("rule", "unknown")
        refs = list(original.get("references", []))
        predicate, host, claim_scope = classify(path, owner)
        referenced_premises = [prior_ledger[ref.removeprefix("assumption:")] for ref in refs if ref.startswith("assumption:") and ref.removeprefix("assumption:") in prior_ledger]
        evidence_refs = list(dict.fromkeys([*refs, *[ref for entry in referenced_premises for ref in entry.get("depends_on", [])]]))
        excerpts = [SourceExcerpt(reference=ref, statement=question, basis="request", status="frozen") if ref == "question" else SourceExcerpt(
            reference=ref, statement=catalog[ref].value, qualifications=catalog[ref].qualifications,
            conditions=getattr(catalog[ref], "conditions", ""), timing=getattr(catalog[ref], "timing", ""),
            basis=catalog[ref].basis, status=catalog[ref].status,
        ) for ref in evidence_refs if ref == "question" or ref in catalog]
        selected = [catalog[ref] for ref in evidence_refs if ref in catalog]
        supplied = bool(refs) and all(ref == "question" or ref in catalog and source_selectable(catalog[ref]) for ref in refs)
        unknown = set(evidence_refs) - (set(catalog) | {"question"} | {"assumption:" + key for key in prior_ledger})
        mismatched_premise = any(
            entry.get("status") == "rejected" or entry.get("scope") != claim_scope.model_dump(mode="json")
            or entry["statement"] != original.get("statement", original.get("explanation"))
            or not set(entry.get("depends_on", [])) <= set(catalog) | {"question"}
            for entry in referenced_premises
        )
        admission = ClaimAdmission(status="candidate", authority="none", rule=rule, predicate=predicate,
            source_refs=refs, sources=excerpts, scope=claim_scope, original_proposal=original,
            reason="The selected sources do not establish this typed claim under an implemented deduction rule.")
        positive = {_directed(record) for record in selected if isinstance(record, Relationship) and record.kind == "causal_influence" and source_selectable(record)}
        negative = {_directed(record) for record in catalog.values() if isinstance(record, Relationship) and record.kind == "no_causal_influence" and source_selectable(record)}
        contradiction = bool((positive - {None}) & (negative - {None})) or any(
            getattr(record, "conflicts_with", None) and record.status != "withdrawn"
            and (ref in evidence_refs or bool(set(record.conflicts_with or []) & set(evidence_refs)))
            for ref, record in catalog.items()
        )
        if unknown or contradiction or mismatched_premise or any(record.status in {"withdrawn", "unresolved"} for record in selected):
            admission.status = "rejected"
            admission.reason = "Unknown/withdrawn/unresolved source references or contradictory typed causal records cannot supply authority or be laundered into assumptions."
        elif rule == "source" and supplied and len(refs) == 1 and all(source_compatible(predicate, ref, claim_scope) for ref in refs):
            admission.status, admission.authority = "admitted", "source"
            admission.reason = "Exact compatible frozen source content restored; no model interpretation is promoted."
            obj["explanation" if "explanation" in obj else "statement"] = "\n\n".join(item.statement for item in excerpts)
        elif rule == "source":
            if len(refs) != 1:
                admission.reason = (
                    f"source requires exactly one reference; this proposal selected {len(refs)}. "
                    "Select one compatible reviewed statement for this field. Its exact content will be restored. "
                    "Do not combine definitions, units, timing or request instructions into one supplied claim. "
                    "Keep unsupported interpretation unknown; additional citations do not supply a deduction."
                )
            elif not supplied:
                admission.reason = "The selected source is not an unconditional reviewed source statement. Preserve its uncertainty; it cannot establish this field as supplied."
            else:
                admission.reason = f"The selected source field or variable scope cannot establish predicate {predicate}. Select a compatible frozen statement or retain this interpretation as unknown."
        elif rule == "relationship_record" and predicate == "relationship_record" and refs == [host["reference"]]:
            record = catalog.get(refs[0])
            effect_ids = {effect["id"] for effect in treatments.get("effects", [])}
            if isinstance(record, Relationship) and record.basis == "source_statement" and record.status == "reviewed" and claim_scope.effect_ids and set(claim_scope.effect_ids) <= effect_ids and host.get("scope", "").strip():
                pair = _directed(record)
                endpoints = (host.get("source_id", ""), host.get("target_id", ""))
                directed = pair is not None and endpoints == pair and host["disposition"] == "applicable"
                non_directed = record.kind in {"association", "membership", "assignment_rule"} and endpoints == ("", "") and host["disposition"] in {"applicable", "not_applicable"}
                preserved = all(not text or text in host["scope"] for text in (record.conditions, record.qualifications, record.timing))
                if (directed or non_directed) and preserved:
                    admission.status, admission.authority = "admitted", "source"
                    admission.scope.variable_ids = sorted(participant.id for participant in record.participants)
                    admission.reason = "Exact reviewed relationship retained with its endpoints, conditions, qualifications and timing. Record retention alone grants no new causal arrow."
                    obj["statement"] = record.value
                elif not preserved:
                    admission.reason = "The relationship scope must retain the exact source conditions, qualifications and timing. Copy these fields into scope without removing their conditions."
        elif rule in {"causal_edge", "common_cause", "mediator_chain", "collider", "temporal_order"} and supplied:
            variable = (host.get("variable") or {}).get("id")
            ts = set(claim_scope.treatment_ids)
            needed: set[tuple[str, str]] = set()
            eligible = False
            if predicate == "causal_edge" and rule == "causal_edge":
                needed, eligible = {(host["source"], host["target"])}, True
                positive = {_directed(record) for record in selected if isinstance(record, Relationship) and record.kind == "causal_influence" and record.directness == "direct" and source_selectable(record)}
            elif predicate == "role" and variable and outcome_id and ts:
                roles = set(host["roles"])
                if rule == "common_cause" and roles <= {"confounder"}:
                    needed, eligible = {(variable, t) for t in ts} | {(variable, outcome_id)}, True
                elif rule == "mediator_chain" and roles <= {"mediator"}:
                    needed, eligible = {(t, variable) for t in ts} | {(variable, outcome_id)}, True
                elif rule == "collider" and roles <= {"collider"}:
                    needed, eligible = {(t, variable) for t in ts} | {(outcome_id, variable)}, True
            elif predicate == "role_timing" and rule == "temporal_order" and variable and ts:
                temporal = {_directed(record) for record in selected if isinstance(record, Relationship) and record.kind == "temporal_order" and source_selectable(record)}
                needed = {(variable, t) for t in ts} if host["timing"] == "before_treatment" else {(t, variable) for t in ts} if host["timing"] == "after_treatment" else set()
                eligible, positive = bool(needed), temporal
            if eligible and needed <= positive:
                admission.status, admission.authority = "admitted", "deduction"
                admission.reason = "Finite typed relationship deduction with exact endpoints and frozen effect/event scope."
                obj["explanation" if "explanation" in obj else "statement"] = f"{rule}: " + "; ".join(f"{start} precedes {end}" if rule == "temporal_order" else f"{start} causally influences {end}" for start, end in sorted(needed))
        elif rule == "accepted_binding" and predicate == "role":
            variable = (host.get("variable") or {}).get("id")
            permitted: set[str] = set()
            for process in treatments.get("assignments", []):
                window = process.get("diagnostic_window") or {}
                if window.get("unit_variable_id") == variable:
                    permitted.add("identifier")
                if window.get("time_variable_id") == variable:
                    permitted.add("time")
            for effect in treatments.get("effects", []):
                comparison = (effect.get("comparison") or {}).get("discontinuity") or {}
                if effect["id"] in claim_scope.effect_ids and comparison.get("running_variable_id") == variable:
                    permitted.add("running_variable")
            if set(host["roles"]) <= permitted and permitted:
                admission.status, admission.authority = "admitted", "deduction"
                admission.reason = "Operational role follows the exact previously accepted typed variable binding; no causal effect or assignment mechanism is inferred."
                obj["statement"] = f"Accepted operational binding for {variable}: {', '.join(sorted(host['roles']))}."
        elif rule == "working_assumption":
            obligation = original.get("required_obligation", "")
            required = predicate.removeprefix("obligation:") if predicate.startswith("obligation:") else "graph_structure" if predicate == "causal_edge" else host.get("strategy") if predicate == "identification" else None
            necessary = predicate == "causal_edge" or predicate == "identification" and host.get("strategy") in OBLIGATIONS or predicate.startswith("obligation:") and obligation in OBLIGATIONS.get(host.get("strategy", ""), set())
            if phase == "causal" and not frozen and necessary and obligation and obligation == required and claim_scope.effect_ids:
                admission.status, admission.authority = "admitted", "assumption"
                admission.reason = "Explicit conditional premise relied upon only for this required causal obligation and exact scope before causal freeze."
            elif predicate.startswith("obligation:"):
                admission.reason = f"Use a supported named conditional obligation for {host.get('strategy')}: {', '.join(sorted(OBLIGATIONS.get(host.get('strategy', ''), set())))}. required_obligation must equal the selected name. Optional unresolved matters belong in limitations, not required identification obligations."
        basis = {"source": "supplied", "deduction": "deduced", "assumption": "assumption", "none": "unknown"}[admission.authority]
        obj["basis"] = basis
        obj["references"] = [ref for ref in evidence_refs if ref in catalog or ref == "question"]
        if admission.authority == "assumption" or admission.status != "admitted":
            identity = "authority_" + fingerprint({"path": path, "scope": claim_scope.model_dump(mode="json"), "proposal": original})[:32]
            status: Literal["candidate", "relied_upon", "rejected"] = "relied_upon" if admission.authority == "assumption" else "rejected" if admission.status == "rejected" else "candidate"
            entry = Assumption(id=identity, statement=str(original.get("statement", original.get("explanation", "Unadmitted proposal"))),
                rationale=admission.reason, depends_on=obj["references"], affects=[path], status=status,
                required_obligation=original.get("required_obligation") or None, scope=claim_scope,
                original_proposal=original, admission_reason=admission.reason)
            if status == "relied_upon" and len(referenced_premises) == 1:
                inherited = deepcopy(referenced_premises[0])
                identity = inherited["id"]
                inherited.update(status="relied_upon", required_obligation=original["required_obligation"], admission_reason=admission.reason)
                entry = Assumption.model_validate(inherited)
            ledger[identity] = entry.model_dump(mode="json")
            admission.ledger_id = identity
            if status == "relied_upon":
                obj["references"].append("assumption:" + identity)
        obj["admission"] = admission.model_dump(mode="json")
        notes.append(f"{path}: {admission.status} ({admission.authority}). {admission.reason}")

    def walk(obj: Any, path: str = "", owner: dict[str, Any] | None = None) -> None:
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}[{i}]", owner)
        elif isinstance(obj, dict):
            if ("statement" in obj and "basis" in obj and "references" in obj) or {"source", "target", "explanation", "basis"} <= obj.keys():
                assess(obj, path, owner or {})
                return
            for key, item in obj.items():
                if key not in {"assumptions", "admission", "original_proposal"}:
                    walk(item, f"{path}.{key}" if path else key, obj)
    walk(output)
    for role in output.get("assignments", []) if phase == "roles" else []:
        role["proposed_roles"] = role.get("proposed_roles") or list(role["roles"])
        role["coverage_status"] = "assessed"
        if role["account"].get("admission", {}).get("status") != "admitted":
            role["roles"], role["operational_use"] = ["unknown"], []
        if role["timing_evidence"].get("admission", {}).get("status") != "admitted":
            role["timing"] = "unknown"
    if "assumptions" in output:
        # Model-authored ledger entries never acquire active status by appearing in a list.
        for item in output["assumptions"]:
            if item["id"] not in prior_ledger:
                item.update(status="candidate", admission_reason="Unrelied model proposal; no required scoped causal obligation has adopted it.")
        output["assumptions"].extend(ledger.values())
    return type(value).model_validate(output), notes
