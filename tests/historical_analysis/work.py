"""Test-only historical worker and receipt fixtures. Not an application path."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any
from uuid import UUID

from langsmith import get_current_run_tree
from pydantic import Field

from backend import model_access
from backend.analysis import domain, lineage, packets, provider, references, registration
from backend.analysis.contracts import (
    Assumption,
    Decision,
    GateCheck,
    RelationshipDisposition,
    Roles,
    Target,
    Variable,
)
from backend.context.contracts import ContextContent
from backend.context.domain import project
from backend.validation import Contract
from historical_analysis import tasks


class ColumnScreen(Contract):
    target_candidate: bool
    treatment_candidate: bool
    context_candidate: bool = Field(default=False, description="Required to bind units, assignment, cohort, time or receipt meaning, not merely potentially useful as a later covariate.")
    summary: str = Field(min_length=1, max_length=1500)
    citations: list[str] = Field(default_factory=list, max_length=12)
    requested_keys: list[str] = Field(default_factory=list, max_length=6)
    widening_reason: str = Field(default="", max_length=1000)
    assumptions: list[Assumption] = Field(default_factory=list, max_length=20)
    variable: Variable | None = Field(default=None, description="One source-qualified, directly observed binding for the owned column. Its id and sole column_id are the exact column identity; derivation is empty. Meaning is supplied or explicitly unknown, never guessed from a name. Required for acceptance, nullable only while requesting evidence. Later event roles reuse this binding; accepted Target/Treatment definitions remain authoritative for selected variables.")


class RelationshipWork(Decision):
    relationships: list[RelationshipDisposition] = Field(default_factory=list, max_length=1)


def validate_screen(value: ColumnScreen, context: ContextContent, columns: list[str]) -> None:
    variable = value.variable
    if len(columns) != 1 or variable is None or variable.id != columns[0] or variable.column_ids != columns or variable.derivation:
        raise ValueError("An accepted screen must bind exactly its one observed column using that column identity and an empty derivation.")
    domain.validate_variable(variable, context)
    meaning = variable.meaning
    if meaning.basis not in {"supplied", "unknown"} or meaning.prerequisites:
        raise ValueError("Screening establishes a supplied or explicitly unknown measurement meaning, not an assumed definition or a later decision's authority.")
    if meaning.basis == "supplied" and (not meaning.references or not set(meaning.references) <= domain.supplied(context)):
        raise ValueError("A supplied screen meaning requires direct offered source evidence; preserve unknown meaning when it is not established.")


def accepted_screen_mapping(records: list[dict[str, Any]], expected_columns: set[str]) -> dict[str, dict[str, Any]]:
    """Derive the one column registry from accepted receipts, never first-role wins."""
    screens = [record for record in records if record["kind"] == "screen"]
    roots = [record for record in screens if not record["scope"].get("parent")]
    if len(roots) != len(expected_columns) or {record["key"] for record in roots} != {"screen:" + cid for cid in expected_columns}:
        raise ValueError("Accepted screen bindings must cover the exact pinned column inventory once.")
    bindings: dict[str, dict[str, Any]] = {}
    for root in roots:
        columns = root["column_ids"]
        if len(columns) != 1 or root["key"] != "screen:" + columns[0]:
            raise ValueError("Screen binding ownership must match its canonical column obligation.")
        children = [record for record in screens if record["scope"].get("parent") == root["id"]]
        if len(children) > 1 or (children and root["status"] != "expanded"):
            raise ValueError("A column binding can have only its one bounded widening replacement.")
        final = children[0] if root["status"] == "expanded" and children else root
        if final["status"] != "accepted" or final["column_ids"] != columns or not final["attempts"]:
            raise ValueError("Every column binding needs its accepted owned screen receipt before role work.")
        receipt = final["attempts"][-1]
        if receipt["status"] != "accepted" or not receipt.get("checks") or any(not check.get("passed") for check in receipt["checks"]):
            raise ValueError("A provider proposal or expanded parent cannot authorize a column binding.")
        variable = ColumnScreen.model_validate(receipt["output"]).variable
        if variable is None or variable.id != columns[0] or variable.column_ids != columns or variable.derivation:
            raise ValueError("The accepted screen receipt does not preserve its exact observed column binding.")
        bindings[columns[0]] = {"variable": variable.model_dump(mode="json"), "receipt_id": receipt["id"]}
    if any(record["scope"].get("parent") not in {root["id"] for root in roots} for record in screens if record["scope"].get("parent")):
        raise ValueError("An orphan screen widening task cannot contribute a column binding.")
    return bindings


def validate_role_variables(value: Roles, established: list[dict[str, Any]]) -> None:
    registry: dict[str, Variable] = {}
    for record in established:
        variable = Variable.model_validate(record)
        if variable.id in registry:
            raise ValueError("A role packet must offer one authoritative definition per variable identity.")
        registry[variable.id] = variable
    for assignment in value.assignments:
        expected = registry.get(assignment.variable.id)
        if expected is None or assignment.variable != expected:
            raise ValueError("Reuse the exact established Variable, including its label, columns, meaning, evidence and derivation. Event-specific interpretation belongs in the role account; it cannot redefine the measurement.")


def role_reference_decoder(established: list[dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Restore exact offered bindings before decoding their scoped handles."""
    registry: dict[str, dict[str, Any]] = {}
    for record in established:
        variable = Variable.model_validate(record)
        if variable.id in registry:
            raise ValueError("A role reference registry must have unique accepted variable identities.")
        registry[variable.id] = variable.model_dump(mode="json")
    if not registry:
        raise ValueError("A role request cannot fall back to authoring variables without its accepted binding registry.")

    def decode(proposal: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(proposal)
        if not isinstance(result.get("assignments", []), list):
            raise ValueError("Role reference output must contain an assignments list.")
        for assignment in result.get("assignments", []):
            pointer = assignment.get("variable") if isinstance(assignment, dict) else None
            if not isinstance(pointer, dict) or set(pointer) != {"id"} or not isinstance(pointer["id"], str) or pointer["id"] not in registry:
                raise ValueError("Each role variable must be exactly {id: offered_identity}, without copied or changed variable fields.")
            assignment["variable"] = deepcopy(registry[pointer["id"]])
        return result

    return decode


def catalogue(context: ContextContent, selected: set[str]) -> list[str]:
    keys = {f"columns.{cid}" for cid in selected}
    for rid, relation in context.relationships.items():
        if relation.status != "withdrawn" and any(p.id in selected for p in relation.participants):
            keys.add(f"relationships.{rid}")
            keys.update(f"columns.{p.id}" if p.kind == "column" else f"concepts.{p.id}" for p in relation.participants)
    return sorted(keys)


def packet(context: ContextContent, columns: set[str], prior: dict[str, Any], question: str, *, extra: list[str] | None = None) -> dict[str, Any]:
    selected = columns | packets.selected(prior)
    scoped = packets.scope(context, selected)
    if extra:
        # Key limits bound individual read requests, not the assembled evidence store.
        for key in extra:
            addition, _, _ = project(context, [key])
            scoped = merge_context(scoped, addition)
    packets.bounded(scoped)
    included = set(scoped.columns)
    offered_keys = {item["key"] for item in packets.evidence_records(scoped)}
    available = []
    for key in catalogue(context, included):
        candidate = project(context, [key])[0]
        if any(item["key"] not in offered_keys for item in packets.evidence_records(candidate)):
            available.append(key)
    catalogue_records = []
    for key in available:
        kind, identity = key.split(".", 1)
        record: dict[str, Any] = {"key": key, "kind": kind, "authority": "index_only; retrieve before citing"}
        if kind == "relationships":
            relation = context.relationships[identity]
            record.update({"relationship_kind": relation.kind, "participants": [p.model_dump() for p in relation.participants]})
        elif kind == "columns":
            record["name"] = context.columns[identity].identity.name
        catalogue_records.append(record)
    return {
        "question": question,
        "task_columns": sorted(columns),
        "column_identities": [scoped.columns[cid].identity.model_dump() for cid in sorted(scoped.columns)],
        "evidence_records": packets.evidence_records(scoped),
        "decisions": packets.decisions(prior),
        "decision_refs": ["decision:" + key for key in prior],
        "available_context": available,
        "context_catalogue": catalogue_records,
        "evidence_scope": packets.exposure(scoped, len(context.columns), columns),
        "_context": scoped.model_dump(mode="json"),
        "_prior": prior,
    }


def merge_context(left: ContextContent, right: ContextContent) -> ContextContent:
    def merge(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            return {key: merge(a[key], b[key]) if key in a and key in b else a[key] if key in a else b[key] for key in a.keys() | b.keys()}
        return b
    return ContextContent.model_validate(merge(left.model_dump(mode="json"), right.model_dump(mode="json")))


def run(task_id: str) -> dict[str, list[str]]:
    """No live Runtime, ORM instance, session or unbounded conversation enters a worker."""
    task = tasks.view(UUID(task_id))
    schema: type[Any] = ColumnScreen if task["kind"] == "screen" else RelationshipWork if task["kind"] == "relationship" else Roles
    for _ in range(2):
        source = task["packet"]
        payload = {key: value for key, value in source.items() if not key.startswith("_")}
        old = tasks.view(UUID(task_id))["attempts"]
        if old:
            last = old[-1]
            payload["correction"] = references.correction(last["output"], last["checks"], last["id"])
        offered, aliases, choices = references.selections(payload)
        prompt = str(source["_prompt"])
        decoder = None
        if schema is Roles and "role_scope" in source:
            owned = offered["role_scope"]["effect_treatments"]
            choices["effect_ids"] = sorted(owned)
            choices["treatment_ids"] = sorted({identity for identities in owned.values() for identity in identities})
            decoder = role_reference_decoder(offered.get("established_variables", []))
            choices["_variable_refs"] = [item["id"] for item in offered["established_variables"]]
            prompt += (
                "\nThis task owns role_scope.column_id at exactly the supplied assignment event. "
                "Explicitly cover every effect/treatment pair in role_scope.effect_treatments, including irrelevant dispositions. "
                "A common assignment may name several effects only when its treatment scope applies to each; otherwise split assignments. "
                "Do not assess other columns or events. Timing and causal roles remain semantic claims, not inferred from task order. "
                "For an established_variables entry, reuse that exact Variable unchanged; its meaning and binding are already authoritative. "
                "Return variable as exactly {id: one offered established variable identity}; the harness restores the accepted fields. Do not copy labels, meanings, citations, columns or derivation into the response pointer. "
                "Every owned column already has a binding. Keep event-specific interpretation in the role account; do not re-author the measurement definition, label or evidence. Unknown binding meaning stays unknown."
            )
        current_policy = bool(source.get("_authority_policy"))
        if current_policy:
            from backend.analysis import authority

            choices["_authority_policy"] = [authority.POLICY]
            choices["_claim_refs"] = ["question", *[item["key"] for item in offered.get("evidence_records", [])],
                                      *["assumption:" + identity for identity in packets.assumption_ids(offered.get("decisions", {}))]]
            decoder = authority.proposal_decoder(base_decoder=decoder)
            offered = packets.model_view(offered)
        thinking = provider.thinking_config(str(source["_model"]), references.output_limit(task["kind"]))
        prepared = provider.prepare(schema, prompt, offered, str(source["_model"]),
            max_output_tokens=references.output_limit(task["kind"]), choices=choices,
            thinking_budget=thinking["thinking_budget"] if thinking else None, reference_decoder=decoder)
        characters = len(prompt) + packets.size(offered) + packets.size(prepared.schema)
        tree = get_current_run_tree()
        metrics = {"input_characters": characters, "output_limit": references.output_limit(task["kind"]), "aliases": aliases,
                   "thinking_config": thinking, "thinking_limit_kind": "soft" if thinking else "provider_default",
                   "trace_id": str(tree.id) if tree else None, "trace_url": tree.get_url() if tree and tree.session_name else None}
        if decoder is not None:
            metrics["reference_decoding"] = "accepted_variable_id"
        if characters > packets.MODEL_INPUT_LIMIT:
            raise packets.PacketOverflow("The complete task request exceeds its bounded context; no evidence was truncated and no model call was made.")
        request = {"system": prompt, "payload": offered, "schema": prepared.schema, "output_limit": metrics["output_limit"], "thinking_config": thinking, "aliases": aliases}
        if decoder is not None:
            request["reference_decoding"] = "accepted_variable_id"
        claim = tasks.claim(UUID(task_id), request=request, metrics=metrics)
        if claim is None:
            if tasks.view(UUID(task_id))["status"] == "running":
                raise RuntimeError("A previous worker lease is still active. Retry after its saved lease expires.")
            break
        output: dict[str, Any] = {}
        raw: str | None = None
        try:
            proposal = model_access.structured(prepared)
            raw, response_metrics = model_access.response_artifacts(proposal)
            metrics.update(response_metrics)
            output = references.translate(proposal.model_dump(mode="json"), aliases)
            requests = output.get("requested_keys", [])
            if requests:
                if not set(requests) <= set(source["available_context"]) or not output.get("widening_reason"):
                    raise ValueError("Widening requires exact available keys and the decision-specific reason.")
                if task["scope"].get("parent"):
                    raise ValueError("This column's additional widening allowance is already spent.")
                receipt = tasks.finish(claim, output, [GateCheck(rule="bounded_evidence_request", kind="deterministic", passed=True, detail="Additional evidence requested within this task's six-read allowance.").model_dump()], status="expanded", metrics=metrics, raw_output=raw)
                return {"completed": [receipt["id"]]}
            context = ContextContent.model_validate(source["_context"])
            prior = packets.decisions(source["_prior"])
            value = schema.model_validate(output)
            registration_checks = []
            if isinstance(value, Decision):
                value = schema.model_validate(domain.resolve(value, {}, known_assumptions=packets.assumption_ids(prior)).model_dump())
                value, notes = registration.normalize(value, context, prior, str(source["assumption_namespace"]), scientific_version=19 if current_policy else 15)
                registration_checks = [registration.gate(notes)]
            if isinstance(value, Roles):
                value = domain.normalize_role_scope(value, prior, source.get("role_scope"))
            if current_policy:
                if isinstance(value, Decision) and value.status == "needs_input":
                    value = value.model_copy(update={"status": "unsupported"})
                value, _ = authority.admit(value, context, prior, str(source["question"]),
                                           "roles" if task["kind"] == "role" else str(task["kind"]), scope=source.get("role_scope"))
            if isinstance(value, ColumnScreen):
                decision = Decision(status="ready", summary=value.summary, citations=sorted(set(value.citations) | domain.references(value.variable.model_dump() if value.variable else {})), changes="This column was screened.", permits="Resolve candidates after complete inventory coverage.")
                # An explicit unknown measurement completes inventory screening
                # without manufacturing a source citation. Essential meanings
                # still need supplied evidence at Target/Treatments.
                checks = [check for check in domain.checks(decision, context, prior) if check.rule != "decision_is_actionable"]
                try:
                    validate_screen(value, context, task["column_ids"])
                    checks.append(GateCheck(rule="owned_column_binding", kind="deterministic", passed=True, detail="This source-qualified or unknown observed binding is owned by this column's accepted screen receipt."))
                except ValueError as error:
                    checks.append(GateCheck(rule="owned_column_binding", kind="deterministic", passed=False, detail=str(error)))
            elif isinstance(value, Roles):
                value = lineage.classify(value, context, Target.model_validate(prior["target"]), scientific_version=19 if current_policy else 15)
                checks = domain.checks(value, context, prior, role_columns=set(task["column_ids"]) | packets.selected(prior), scientific_version=19 if current_policy else 15)
                if "role_scope" in source:
                    try:
                        validate_role_variables(value, source.get("established_variables", []))
                        checks.append(GateCheck(rule="immutable_variable_binding", kind="deterministic", passed=True, detail="Event roles reuse their exact accepted column binding; selected Target/Treatment definitions retain authority."))
                    except ValueError as error:
                        checks.append(GateCheck(rule="immutable_variable_binding", kind="deterministic", passed=False, detail=str(error)))
                    try:
                        domain.validate_role_scope(value, source["role_scope"])
                        checks.append(GateCheck(rule="role_effect_event_coverage", kind="deterministic", passed=True, detail="Every owned effect and assignment-event scope has an explicit disposition."))
                    except ValueError as error:
                        checks.append(GateCheck(rule="role_effect_event_coverage", kind="deterministic", passed=False, detail=str(error)))
            else:
                checks = domain.checks(value, context, prior, scientific_version=19 if current_policy else 15)
                expected = task["scope"]["relationship"]
                checks.append(GateCheck(rule="relationship_identity", kind="deterministic", passed=len(value.relationships) == 1 and value.relationships[0].reference == expected, detail="A relationship decision must account for its exact source record."))
                try:
                    domain.validate_relationships(value.relationships, context, {item["id"] for item in prior["treatments"]["effects"]})
                    checks.append(GateCheck(rule="relationship_structure", kind="deterministic", passed=True, detail="Relationship endpoints, effect scopes and qualifications are valid without inferring causal arrows."))
                except ValueError as error:
                    checks.append(GateCheck(rule="relationship_structure", kind="deterministic", passed=False, detail=str(error)))
            checks = [*registration_checks, *checks]
            output = value.model_dump(mode="json")
            status = "accepted" if all(g.passed for g in checks) else "blocked" if isinstance(value, Decision) and value.status != "ready" else "rejected"
            receipt = tasks.finish(claim, output, [g.model_dump() for g in checks], status=status, metrics=metrics, raw_output=raw)
            if status != "rejected":
                return {"completed": [receipt["id"]]}
        except tasks.TaskConflict:
            raise
        except (ValueError, RuntimeError) as error:
            error_raw, error_metrics = model_access.response_artifacts(error)
            if error_raw is not None:
                raw = error_raw
            metrics.update(error_metrics)
            checks = [GateCheck(rule="typed_submission", kind="deterministic", passed=False, detail=str(error)[:4000])]
            status = "blocked" if isinstance(error, packets.PacketOverflow) else "rejected"
            tasks.finish(claim, {"error": str(error)[:4000], **output}, [g.model_dump() for g in checks], status=status, raw_output=raw, metrics=metrics)
            if status == "blocked":
                break
    return {"completed": []}
