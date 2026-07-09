"""Historical packet/coordinator fixtures, without an executable analysis graph."""


from typing import Annotated, Any, TypedDict

from backend import numerical as evidence
from backend.analysis import domain, packets, prompts
from backend.analysis.contracts import (
    Conclusion,
    Roles,
    Target,
    Treatments,
)
from backend.context.contracts import ContextContent, statement_entries
from backend.context.domain import project
from backend.context.service import fingerprint
from backend.responses import Reply
from historical_analysis import tasks, work


class State(TypedDict):
    run_id: str
    work_kind: str
    completed: Annotated[list[str], lambda a, b: sorted(set(a) | set(b))]

def bind_requested_numbers(value: Conclusion, question: str) -> Conclusion:
    """Resolve literals from an explicitly cited, immutable request, never inferred results.

    Arm labels and requested confidence levels need not be transcribed a second time.
    Computed numbers still require the model's exact numerical source selection.
    """
    value = value.model_copy(deep=True)
    if "question" in value.citations:
        bound = evidence.bind_literals(Reply.model_validate(value.report.model_dump()), {"question": question})
        value.report = type(value.report).model_validate({"text": bound.text, "numbers": [n.model_dump() for n in bound.numbers]})
    return value


class Runtime:
    def prior(self) -> dict[str, Any]:
        return {
            name: self.h.saved["decision:" + name]
            for name in ("target", "treatments", "roles", "causal")
            if "decision:" + name in self.h.saved
        }

    def frozen_identity(self) -> dict[str, Any]:
        frozen = self.run.inputs.get("frozen_context")
        if not isinstance(frozen, dict):
            return {}
        return {name: frozen[name] for name in (
            "dataset_version_id", "source_sha256", "context_revision_id", "context_hash",
            "readiness_fingerprint", "policy_version",
        )}

    def semantic_context(self, name: str | None = None) -> ContextContent:
        candidates = [name] if name else ["causal", "roles", "treatments", "target"]
        for candidate in candidates:
            value = self.h.saved.get("semantic_context:" + str(candidate))
            if value is not None:
                return ContextContent.model_validate(value)
        return self.context

    def packet(self, name: str | None = None) -> dict[str, Any]:
        context = self.semantic_context(name)
        return {**self.scoped_packet(context, set(context.columns)), "_retrieved": self.h.saved.get("dependency_evidence:" + str(name), [])}

    def scoped_packet(self, context: ContextContent, columns: set[str]) -> dict[str, Any]:
        packets.bounded(context)
        return {
            "question": self.run.question,
            "frozen_inputs": self.frozen_identity(),
            "dataset_version": str(self.version.id),
            "context_revision": str(self.run.context_revision_id),
            "column_identities": [context.columns[cid].identity.model_dump(mode="json") for cid in sorted(context.columns)],
            "evidence_records": packets.evidence_records(context),
            "decisions": packets.decisions(self.prior()),
            "decision_refs": ["decision:" + key for key in self.prior()],
            "prior_exposure": self.run.inputs["prior_exposure"],
            "evidence_scope": packets.exposure(context, len(self.context.columns), columns),
        }

    def inspect(self, name: str) -> bool:
        cache = "semantic_context:" + name
        if cache not in self.h.saved:
            columns = set(self.h.saved.get("binding_candidates", [])) if name in {"target", "treatments"} else packets.active(self.prior())
            content = packets.scope(self.context, columns)
            required = domain.references(packets.decisions(self.prior()))
            if name == "target":
                required.update(self.h.saved.get("binding_evidence", []))
            if name in {"target", "treatments"} and self.context.analysis_brief:
                from backend.context.contracts import BriefClaim
                brief = self.context.analysis_brief
                claims = [brief.comparison, brief.population, brief.unit, brief.timing, *brief.assumptions]
                if brief.target:
                    claims.append(brief.target.meaning)
                claims.extend(claim for treatment in brief.treatments for claim in (treatment.meaning, treatment.change))
                required.update(key for claim in claims if isinstance(claim, BriefClaim) for key in claim.source_keys)
            if name == "causal":
                for record in self.h.saved.get("relationship_decisions", []):
                    if any(item["reference"].removeprefix("relationships.") in content.relationships for item in record.get("relationships", [])):
                        required.update(domain.references(record))
            indexed = {key for key, statement in statement_entries(self.context) if statement.status != "withdrawn"}
            offered = {key for key, _ in statement_entries(content)}
            retrieved = sorted((required & indexed) - offered)
            for key in retrieved:
                content = work.merge_context(content, project(self.context, [key])[0])
            packets.bounded(content)
            self.h.remember("dependency_evidence:" + name, retrieved, commit=False)
            self.h.remember(cache, content.model_dump(mode="json"), commit=not retrieved)
            if retrieved:
                self.h.event("dependency_evidence:" + name, name, "evidence_retrieved", "Retrieved exact evidence cited by accepted prerequisites.", {"keys": retrieved, "reason": "accepted_dependency", "model_calls": 0})
        return True

    def prepare_work(self, kind: str) -> dict[str, Any]:
        checkpoint = {"screen": "target", "role": "roles", "relationship": "causal"}[kind]
        prerequisites = {"screen": ["bind"], "role": ["target", "treatments"], "relationship": ["roles"]}[kind]
        dependencies = [receipt.id for name in prerequisites if (receipt := self.h.accepted(name)) is not None]
        if len(dependencies) != len(prerequisites):
            raise ValueError("Scientific tasks require accepted prerequisite receipts.")
        self.run.stage = checkpoint
        self.db.commit()
        prior = {} if kind == "screen" else self.prior()
        if kind == "role":
            prior = {key: value for key, value in prior.items() if key in {"target", "treatments"}}
        role_scopes = {item["key"]: item for item in domain.role_scopes(self.context, Target.model_validate(prior["target"]), Treatments.model_validate(prior["treatments"]))} if kind == "role" else {}
        screen_bindings = work.accepted_screen_mapping(tasks.list_tasks(self.run.id, include_packets=False), set(self.context.columns)) if kind == "role" else {}
        selected_variables = domain.variables(Target.model_validate(prior["target"]), Treatments.model_validate(prior["treatments"])) if kind == "role" else {}
        if kind == "relationship":
            scopes = [(f"relationships.{rid}", {p.id for p in relation.participants if p.kind == "column"})
                      for rid, relation in self.context.relationships.items()
                      if relation.status != "withdrawn"]
        elif kind == "role":
            scopes = [(key, {scope["column_id"]}) for key, scope in role_scopes.items()]
        else:
            columns = set(self.context.columns)
            scopes = [(cid, {cid}) for cid in sorted(columns)]
        for identity, columns in scopes:
            task_dependencies = list(dependencies)
            key = identity if kind == "role" else f"{kind}:{identity}"
            scope: dict[str, Any] = {"scientific_scope": fingerprint(prior), **({"relationship": identity} if kind == "relationship" else {})}
            if kind == "role":
                scope["role_scope"] = {field: role_scopes[key][field] for field in ("column_id", "event", "effect_treatments")}
            payload = work.packet(self.context, columns, prior, self.run.question, extra=[identity] if kind == "relationship" else None)
            payload.update({"_model": self.run.inputs["model"], "_prompt": self.run.inputs["bindings" if kind == "screen" else "roles"] if kind != "relationship" else prompts.RELATIONSHIP,
                            "dataset_version": str(self.version.id), "context_revision": str(self.run.context_revision_id), "assumption_namespace": f"a_{fingerprint(key)[:10]}_",
                            "frozen_inputs": self.frozen_identity(), "_authority_policy": self.run.inputs["contract"]})
            if kind == "relationship":
                payload["relationship_key"] = identity
            if kind == "role":
                payload["role_scope"] = scope["role_scope"]
                selected = [item.model_dump(mode="json") for item in selected_variables.values() if columns & set(item.column_ids)]
                binding = screen_bindings[role_scopes[key]["column_id"]]
                payload["established_variables"] = selected or [binding["variable"]]
                payload["variable_binding_source"] = "selected_decision" if selected else "column_screen"
                task_dependencies.append(binding["receipt_id"])
            tasks.ensure(self.run.id, key=key, checkpoint=checkpoint, kind=kind, column_ids=sorted(columns), scope=scope,
                         reason=f"Resolve {kind} evidence for this scientific obligation.", dependency_receipt_ids=task_dependencies,
                         packet=payload, budget_kind="relationship" if kind == "relationship" else "column", obligation_keys=role_scopes[key]["obligation_keys"] if kind == "role" else [key])
        return {"work_kind": kind}

    def complete_work(self, state: State) -> dict[str, Any]:
        kind = state["work_kind"]
        records = [task for task in tasks.list_tasks(self.run.id, include_packets=False) if task["kind"] == kind]
        role_scopes = {item["key"]: item for item in domain.role_scopes(self.context, Target.model_validate(self.prior()["target"]), Treatments.model_validate(self.prior()["treatments"]))} if kind == "role" else {}
        expected = set(role_scopes) if kind == "role" else {f"relationship:relationships.{rid}" for rid, relation in self.context.relationships.items() if relation.status != "withdrawn"} if kind == "relationship" else {f"screen:{cid}" for cid in self.context.columns}
        roots = [task for task in records if not task["scope"].get("parent")]
        if {task["key"] for task in roots} != expected:
            raise ValueError("Scientific task coverage differs from the complete pinned inventory; unexamined work cannot be accepted.")
        accepted = []
        for task in roots:
            children = [child for child in records if child["scope"].get("parent") == task["id"]]
            if len(children) > 1 or (children and task["status"] != "expanded"):
                raise ValueError("A scientific obligation must have at most one bounded widening dependency.")
            final = children[0] if task["status"] == "expanded" and children else task
            if final["status"] != "accepted":
                self.run.status, self.run.error = "blocked", f"Scientific obligation {task['key']} is {final['status']}. Inspect its evidence and attempt allowance."
                self.db.commit()
                return {}
            if kind == "role":
                expected_final = ["widen:" + cid for cid in task["column_ids"]] if final is not task else task["obligation_keys"]
                if task["obligation_keys"] != role_scopes[task["key"]]["obligation_keys"] or final["obligation_keys"] != expected_final or final["scope"].get("role_scope") != task["scope"].get("role_scope"):
                    raise ValueError("Role task obligations must preserve every canonical effect/event scope.")
                domain.validate_role_scope(Roles.model_validate(final["attempts"][-1]["output"]), role_scopes[task["key"]])
            accepted.append(final["attempts"][-1])
        if kind == "screen":
            selected = [cid for task in records if task["status"] == "accepted" and (task["attempts"][-1]["output"]["target_candidate"] or task["attempts"][-1]["output"]["treatment_candidate"] or task["attempts"][-1]["output"].get("context_candidate", False)) for cid in task["column_ids"]]
            if self.context.analysis_brief:
                # Screening covers the inventory; it cannot reselect the applied question.
                brief = self.context.analysis_brief
                selected = list({cid for treatment in brief.treatments for cid in treatment.column_ids}
                                | set(brief.target.column_ids if brief.target else []))
            self.h.remember("binding_candidates", sorted(set(selected)))
            self.h.remember("binding_receipts", [r["id"] for r in accepted])
            self.h.remember("binding_evidence", sorted({ref for r in accepted if set(r["output"].get("variable", {}).get("column_ids", [])) & set(selected) for ref in domain.references(r["output"])}))
        elif kind == "role":
            values = [Roles.model_validate(receipt["output"]) for receipt in accepted]
            ledger: dict[str, Any] = {}
            for value in values:
                for assumption in value.assumptions:
                    if assumption.id in ledger and ledger[assumption.id] != assumption:
                        raise ValueError("Conflicting assumption definitions cannot be merged.")
                    ledger[assumption.id] = assumption
            combined = Roles(status="ready", citations=["decision:target", "decision:treatments"], summary="Every owned column/effect/assignment-event scope has an explicit scientific disposition.", changes="Canonical role obligations are accepted and reconciled.", permits="Establish and validate the causal account.", assignments=[a for value in values for a in value.assignments], assumptions=list(ledger.values()), unresolved=list(dict.fromkeys(u for value in values for u in value.unresolved)))
            checks = domain.checks(combined, self.context, self.prior(), scientific_version=19)
            if all(c.passed for c in checks):
                self.h.remember("decision:roles", combined.model_dump(mode="json"), commit=False)
            self.h.finish("roles", 0, {"_prerequisite_ids": [r["id"] for r in accepted], "decision_refs": [r["id"] for r in accepted], "coverage": {"selected": sorted(packets.selected(self.prior())), "assessed": sorted({scope["column_id"] for scope in role_scopes.values()}), "total": len(self.context.columns), "role_tasks": len(role_scopes), "effect_event_obligations": sum(len(scope["obligation_keys"]) for scope in role_scopes.values())}}, combined.model_dump(mode="json"), checks, key="roles_coverage", terminal=None if all(c.passed for c in checks) else "blocked")
        else:
            self.h.remember("relationship_decisions", [r["output"] for r in accepted])
            self.h.remember("relationship_receipts", [r["id"] for r in accepted])
        return {}

