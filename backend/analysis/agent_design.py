"""Analysis design tools with source-bound roles and restricted diagnostics."""

from typing import Any, Literal
from uuid import uuid5

from pydantic import Field

from backend.analysis import authority, domain, lineage, packets
from backend.analysis.agent_context import (
    NoArguments,
    Publish,
    ReadContext,
    Workspace,
    decision,
    require_checks,
)
from backend.analysis.agent_review import require_review
from backend.analysis.agent_runtime import Tool
from backend.analysis.contracts import (
    CausalAccount,
    CausalEdge,
    CausalNode,
    Claim,
    DiagnosticAction,
    Frame,
    IdentificationClaim,
    InputProjection,
    Plan,
    Program,
    RelationshipDisposition,
    RoleAssignment,
    Roles,
    Snapshot,
    Target,
    Treatments,
    Variable,
)
from backend.context.service import fingerprint
from backend.execution import runner as sandbox
from backend.storage import files
from backend.validation import Contract


class RoleInput(Contract):
    scope_key: str
    assignments: list[RoleAssignment] = Field(min_length=1, max_length=20)


class RolesInput(Contract):
    scopes: list[RoleInput] = Field(min_length=1, max_length=30)


class StructureInput(Contract):
    nodes: list[CausalNode] = Field(max_length=100)
    edges: list[CausalEdge] = Field(max_length=300)
    identification: str = Field(min_length=1, max_length=5000)
    threats: list[str] = Field(default_factory=list, max_length=30)


class IdentificationInput(Contract):
    identification: IdentificationClaim


class RelationshipChoice(Contract):
    reference: str = Field(description="One exact relationships.<id> key from the frozen context.")
    effect_ids: list[str] = Field(min_length=1, max_length=20)
    disposition: Literal["applicable", "not_applicable"]
    scope: str = Field(min_length=1, max_length=2000, description="Explain applicability to the requested effects. The backend retains all original conditions, qualifications and timing alongside this interpretation.")
    source_id: str = Field(default="", description="Only needed to disambiguate a directed source with multiple from participants. Otherwise the backend selects its unique exact endpoint.")
    target_id: str = Field(default="", description="Only needed to disambiguate multiple to participants. Assignment rules, associations and membership have no directed graph endpoints.")


class RelationshipInput(Contract):
    relationships: list[RelationshipChoice] = Field(min_length=1, max_length=30)


class MethodInput(Contract):
    plan: Plan


class InspectDesign(Contract):
    column_ids: list[str] = Field(default_factory=list, max_length=30, description="Additional columns whose event-specific role scope keys are needed. Empty returns up to ten available scopes. Accepted outcome and treatment bindings already have their roles and must not be submitted here.")


class Design(Workspace):
    def study(self) -> dict[str, Any]:
        handoff = self.store.accepted("study")
        if handoff is None:
            raise ValueError("Design requires the accepted study handoff.")
        return {name: handoff["records"][name] for name in ("target", "treatments")}

    def scopes(self) -> list[dict[str, Any]]:
        study = self.study()
        return domain.role_scopes(self.context, Target.model_validate(study["target"]), Treatments.model_validate(study["treatments"]))

    def roles(self) -> Roles:
        study, records = self.study(), self.records()
        established = domain.variables(Target.model_validate(study["target"]), Treatments.model_validate(study["treatments"]))
        assignments: list[RoleAssignment] = []
        ledger: dict[str, Any] = {}
        for scope in self.scopes():
            saved = records.get(scope["key"])
            if saved is None:
                cid = scope["column_id"]
                unknown = Claim(statement="Not selected for use. Scientific role remains unresolved.", basis="unknown")
                # A complete inventory is deterministic. Unselected columns are not
                # presumed irrelevant, and do not require individual model calls.
                pending = Roles.model_validate(decision("Unselected column retained as unresolved.", citations=["decision:treatments"],
                    assignments=[RoleAssignment(variable=established.get(cid) or Variable(id=cid, label=self.context.columns[cid].identity.name,
                        column_ids=[cid], meaning=unknown), roles=["unknown"], timing="unknown", account=unknown,
                        timing_evidence=unknown, reason="No scientific or operational use has been admitted.",
                        effect_ids=list(scope["effect_treatments"]),
                        treatment_ids=sorted({tid for ids in scope["effect_treatments"].values() for tid in ids}),
                        time_scope="Unresolved; withheld from diagnostic and target computation.").model_dump(mode="json")]))
                pending = domain.normalize_role_scope(pending, study, scope)
                pending, _ = authority.admit(pending, self.context, study, self.run.question, "roles", scope=scope)
                pending = lineage.classify(pending, self.context, Target.model_validate(study["target"]), scientific_version=19)
                saved = pending.model_dump(mode="json")
            role = Roles.model_validate(saved)
            assignments.extend(role.assignments)
            ledger.update({item.id: item.model_dump(mode="json") for item in role.assumptions})
        return Roles.model_validate(decision("Role inventory retains unresolved columns without granting access or scientific irrelevance.",
            citations=["decision:treatments"], assignments=[item.model_dump(mode="json") for item in assignments], assumptions=list(ledger.values())))

    def editable(self) -> None:
        if "frame" in self.records():
            raise ValueError("The causal account is frozen for diagnostic access. It cannot change after observations.")

    def inspect(self, args: InspectDesign, key: str) -> dict[str, Any]:
        if not set(args.column_ids) <= self.context.columns.keys():
            raise ValueError("Select exact existing column IDs from the catalogue.")
        scopes = self.scopes()
        selected = [scope for scope in scopes if scope["column_id"] in args.column_ids] if args.column_ids else scopes[:10]
        ids = {scope["column_id"] for scope in selected}
        roles = self.roles().model_dump(mode="json")
        roles["assignments"] = [item for item in roles["assignments"] if set(item["variable"]["column_ids"]) & ids]
        return {"role_scopes": selected, "role_scope_count": len(scopes),
                "conditional_obligations": {strategy: sorted(names) for strategy, names in authority.OBLIGATIONS.items()},
                "obligation_policy": "Select the necessary conditional premises for the chosen strategy. This catalogue is not a requirement to submit every premise. Preserve other optional uncertainties in limitations. A measurement-comparability premise remains an assumption, not an observed fact.",
                "role_policy": "Outcome and treatment bindings are already accepted. These scopes concern additional columns only. Submit a scope only if a relevant scientific or operational use is needed. Otherwise proceed directly to the causal structure and identification; unselected columns remain withheld.",
                "column_inventory_size": len(self.context.columns), "roles": packets.model_view(roles),
                "source_relationships": [{"reference": "relationships." + identity, "kind": item.kind,
                    "participants": [part.model_dump(mode="json") for part in item.participants],
                    "conditions": item.conditions, "qualifications": item.qualifications, "timing": item.timing}
                    for identity, item in self.context.relationships.items() if item.status == "reviewed"],
                "records": self.visible_records()}

    def set_roles(self, args: RolesInput, key: str) -> dict[str, Any]:
        self.editable()
        study = self.study()
        scopes = {item["key"]: item for item in self.scopes()}
        results = []
        for item in args.scopes:
            scope = scopes.get(item.scope_key)
            if scope is None:
                raise ValueError("Select an exact role scope from inspect_design.role_scopes[].key. Outcome and treatment bindings are already accepted, so never submit them as additional roles. Unused columns need no submission.")
            value = Roles.model_validate(decision("Bound roles for the selected assignment event.", assignments=[entry.model_dump(mode="json") for entry in item.assignments]))
            value = domain.normalize_role_scope(value, study, scope)
            value, _ = authority.admit(value, self.context, study, self.run.question, "roles", scope=scope)
            value = lineage.classify(value, self.context, Target.model_validate(study["target"]), scientific_version=19)
            self.check_exposure(value.model_dump(mode="json"))
            domain.validate_role_scope(value, scope)
            require_checks(domain.checks(value, self.context, study, role_columns=packets.selected(study) | {scope["column_id"]}, scientific_version=19))
            results.append(self.save(item.scope_key, value.model_dump(mode="json")))
        return {"updates": results, "next_action": "Unselected or unknown columns remain withheld and do not require further submissions. Continue to structure and identification unless the exact estimator needs this column."}

    def admit_causal(self, value: CausalAccount) -> CausalAccount:
        prior = {**self.study(), "roles": self.roles().model_dump(mode="json")}
        self.check_exposure(value.model_dump(mode="json"))
        accepted, notes = authority.admit(value, self.context, prior, self.run.question, "causal")
        try:
            authority.validate(accepted, prior, phase="causal")
        except ValueError as error:
            raise ValueError(str(error) + "\n" + "\n".join(notes)) from error
        return accepted

    def set_structure(self, args: StructureInput, key: str) -> dict[str, Any]:
        self.editable()
        accepted = self.admit_causal(CausalAccount.model_validate(decision("Proposed conditional graph structure.", **args.model_dump(mode="json"))))
        return self.save("structure", accepted.model_dump(mode="json"))

    def set_identification(self, args: IdentificationInput, key: str) -> dict[str, Any]:
        self.editable()
        if args.identification.effect_id not in {item["id"] for item in self.study()["treatments"]["effects"]}:
            raise ValueError("Identification must select an accepted requested effect.")
        accepted = self.admit_causal(CausalAccount.model_validate(decision("Proposed effect-specific identification.",
            identification="One effect's identification account.", identifications=[args.identification.model_dump(mode="json")])))
        return self.save("identification:" + args.identification.effect_id, accepted.model_dump(mode="json"))

    def set_relationships(self, args: RelationshipInput, key: str) -> dict[str, Any]:
        self.editable()
        selections = []
        for item in args.relationships:
            source = self.context.relationships.get(item.reference.removeprefix("relationships.")) if item.reference.startswith("relationships.") else None
            if source is None or source.status != "reviewed":
                raise ValueError("Select one exact reviewed source_relationships reference from inspect_design.")
            # Reading the selected existing record exposes it to the agent and
            # preserves provenance without asking the model to transcribe facts.
            self.read(ReadContext(keys=[item.reference]), key)
            directed = source.kind in {"causal_influence", "no_causal_influence", "temporal_order"}
            endpoints = {}
            for name, role in (("source_id", "from"), ("target_id", "to")):
                choices = [part.id for part in source.participants if part.role == role]
                selected = getattr(item, name)
                if not directed:
                    if selected:
                        raise ValueError("Assignment rules, associations and membership retain participants without directed graph endpoints.")
                    endpoints[name] = ""
                elif selected in choices:
                    endpoints[name] = selected
                elif not selected and len(choices) == 1:
                    endpoints[name] = choices[0]
                else:
                    raise ValueError(f"Choose {name} from the exact {role} participants: {choices}.")
            selections.append({"reference": item.reference, "effect_ids": item.effect_ids,
                "disposition": item.disposition, **endpoints,
                "scope": "\n".join([item.scope, *[value for value in (source.conditions, source.qualifications, source.timing) if value]]),
                "rationale": {"statement": source.value, "rule": "relationship_record", "references": [item.reference],
                              "prerequisites": [], "required_obligation": ""}})
        accepted = self.admit_causal(CausalAccount.model_validate(authority.proposal_decoder()(decision("Source relationships retained with their exact scope.",
            identification="Source relationship review.", relationships=selections))))
        domain.validate_relationships(accepted.relationships, self.context, {item["id"] for item in self.study()["treatments"]["effects"]})
        return {"updates": [self.save(item.reference, item.model_dump(mode="json")) for item in accepted.relationships]}

    def causal(self) -> CausalAccount:
        records = self.records()
        if "structure" not in records:
            raise ValueError("Set the graph structure before freezing the causal account.")
        value = CausalAccount.model_validate(records["structure"])
        ledger = {item.id: item for item in value.assumptions}
        value.identifications = []
        for name, saved in sorted(records.items()):
            if name.startswith("identification:"):
                part = CausalAccount.model_validate(saved)
                value.identifications.extend(part.identifications)
                ledger.update({item.id: item for item in part.assumptions})
        value.relationships = [RelationshipDisposition.model_validate(saved) for name, saved in sorted(records.items()) if name.startswith("relationships.")]
        value.assumptions = list(ledger.values())
        value.citations = sorted(domain.references(value.model_dump(mode="json")) - {"decision:causal"})
        study = self.study()
        return domain.canonical_graph(value, Target.model_validate(study["target"]), Treatments.model_validate(study["treatments"]), self.roles())

    def freeze_causal(self, args: NoArguments, key: str) -> dict[str, Any]:
        if "frame" in self.records():
            return {"frame": packets.model_view(self.records()["frame"])}
        prior = self.study()
        roles = self.roles()
        self.validate(roles, prior)
        prior["roles"] = roles.model_dump(mode="json")
        causal = self.causal()
        self.validate(causal, prior)
        frame = Frame(target=Target.model_validate(prior["target"]), treatments=Treatments.model_validate(prior["treatments"]),
                      roles=roles, causal=causal, graph_hash=domain.graph_hash(causal))
        reviewed = require_review(self, {"frame": packets.science(frame.model_dump(mode="json")), "column_inventory_size": len(self.context.columns)},
            "Check the complete causal account against frozen sources and requested effects. Verify role relevance, exact comparison, assignment, qualified relationships, identification and necessary conditional assumptions. Unknown unselected columns are not automatically blockers. Reject unsupported claims of identification or missing essential premises.")
        raw, manifest = domain.projection(frame, self.context, files().read(self.version.storage_key, self.version.sha256), scientific_version=19)
        del raw
        self.save("causal_review", reviewed)
        self.save("diagnostic_manifest", manifest)
        return self.save("frame", frame.model_dump(mode="json"))

    def diagnose(self, args: DiagnosticAction, key: str) -> dict[str, Any]:
        records = self.records()
        if "frame" not in records:
            raise ValueError("Freeze the validated causal account before requesting diagnostic data.")
        if args.action != "run_code":
            return self.save("diagnostic_conclusion", args.model_dump(mode="json"))
        count = sum(name.startswith("diagnostic:") for name in records)
        if count >= 4:
            raise ValueError("Four diagnostic programs have already run. Decide from the retained evidence.")
        domain.validate_program(Program(code=args.code, summary=args.question))
        raw, manifest = domain.projection(Frame.model_validate(records["frame"]), self.context,
            files().read(self.version.storage_key, self.version.sha256), scientific_version=19)
        runtime = sandbox.ready()
        result = sandbox.run(uuid5(self.run.id, "diagnostic:" + key), args.code, raw, {}, runtime["runtime_hash"])
        reference = f"diagnostic:{count}"
        record = {"reference": reference, **args.model_dump(mode="json"), "manifest": manifest, "result": result}
        # Existing artifact readers use the execution receipt, never model prose.
        self.run.receipts = {**self.run.receipts, reference: record}
        self.save(reference, record)
        return {"reference": reference, "result": {name: value for name, value in result.items() if name != "files"}, "manifest": packets.diagnostic_view(manifest)}

    def set_method(self, args: MethodInput, key: str) -> dict[str, Any]:
        records = self.records()
        if "frame" not in records:
            raise ValueError("Freeze the causal account before the executable method.")
        frame = records["frame"]
        value = args.plan
        if value.assumptions:
            raise ValueError("Method selection cannot add or rewrite assumptions. Cite the frozen causal ledger.")
        # Carry the immutable ledger without making the model transcribe it.
        value.assumptions = CausalAccount.model_validate(frame["causal"]).assumptions
        computed = {name for name in records if name.startswith("diagnostic:")}
        self.check_exposure(value.model_dump(mode="json"))
        require_checks(domain.checks(value, self.context, frame, computed, scientific_version=19))
        return self.save("plan", value.model_dump(mode="json"))

    def publish(self, args: Publish, key: str) -> dict[str, Any]:
        self.store.submission(self.task, key)
        records = self.records()
        if not {"frame", "plan"} <= records.keys():
            raise ValueError("Freeze the causal account and validate the executable method first.")
        prior = records["frame"]
        snapshot = Snapshot(version_id=self.version.id, context_revision_id=self.run.context_revision_id,
            source_sha256=self.version.sha256, context_hash=fingerprint(self.context.model_dump(mode="json")),
            question=self.run.question, frame=Frame.model_validate(prior), plan=Plan.model_validate(records["plan"]),
            diagnostic_refs=sorted(name for name in records if name.startswith("diagnostic:")),
            prior_exposure=str(self.run.inputs["prior_exposure"]),
            input_projection=InputProjection(columns=[self.context.columns[cid].identity for cid in sorted(packets.active(prior))]))
        return {"handoff": self.store.publish(self.task, {"snapshot": snapshot.model_dump(mode="json"),
            "diagnostics": [records[name] for name in snapshot.diagnostic_refs], "review": records["causal_review"]}, args.summary, args.unresolved)}

    def tools(self) -> list[Tool]:
        return [*self.common_tools(),
            Tool("inspect_design", "Inspect the saved design and inventory count. Select only columns you need to assess to retrieve their exact role scope keys. No model call per column is required. Unselected unknown columns remain inventoried and withheld.", InspectDesign, self.inspect),
            Tool("set_roles", "Bind only relevant scientific or operational roles in batches. Select exact role scope keys and retain effect, treatment and event meaning. Other columns remain unresolved and unused.", RolesInput, self.set_roles, claims=True),
            Tool("set_structure", "Propose graph nodes and edges with traceable source rules or necessary conditional graph assumptions. Preserve accepted variable identities. This does not execute diagnostics.", StructureInput, self.set_structure, claims=True),
            Tool("set_identification", "Set one exact effect's identification argument, adjustment and named conditional obligations. Working assumptions are allowed here before freeze and remain explicit.", IdentificationInput, self.set_identification, claims=True),
            Tool("set_relationships", "Select existing reviewed source relationships and explain their applicability. The backend reads each exact frozen record and retains its source content, endpoints, conditions, qualifications and timing without model transcription. This retains evidence; it does not prove a causal arrow. Use inspect_design.source_relationships for exact references.", RelationshipInput, self.set_relationships),
            Tool("freeze_causal_account", "Validate all bindings, roles, relationships and identification, obtain independent review, then freeze the causal account and expose the restricted diagnostic manifest. After this, scientific premises cannot change.", NoArguments, self.freeze_causal),
            Tool("run_diagnostic", "Run a bounded deterministic Python diagnostic against the physical restricted projection, or record a finish conclusion. At most four programs. Read os.environ['DATA_CSV']; write facts.json in the working directory. Target outcomes and unknown columns remain withheld. Use the manifest's canonical column IDs. Observations cannot change frozen causal claims.", DiagnosticAction, self.diagnose),
            Tool("set_method", "Specify the exact estimator, inference, population, missingness, exclusions and reporting contract for every accepted effect. Preserve identifying adjustment. Leave assumptions empty; the backend carries the frozen ledger.", MethodInput, self.set_method),
            Tool("publish_handoff", "Seal the frozen analysis specification and diagnostic evidence for computation. At most three submissions. No target-effect execution occurs here.", Publish, self.publish),
        ]
