"""Test-only historical claim and review harness for compatibility regressions."""
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, TypeVar, cast

from langsmith import get_current_run_tree
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend import model_access
from backend.analysis import domain, effect_references, packets, provider, references
from backend.analysis.checkpoints import BY_ID, budgets, receipts
from backend.analysis.contracts import (
    CheckpointReceipt,
    CodeReview,
    Decision,
    GateCheck,
    Target,
    Treatments,
)
from backend.context.service import fingerprint
from backend.models import AnalysisRun, now
from historical_analysis import tasks

T = TypeVar("T", bound=BaseModel)

class BudgetExhausted(ValueError):
    pass

class Harness:
    def __init__(self, db: Session, run: AnalysisRun, shared_owner: tasks.SharedOwner | None = None):
        self.db, self.run = db, run
        self.shared_owner = shared_owner
        self.saved: dict[str, Any] = deepcopy(run.receipts)

    def remember(self, key: str, value: Any, *, commit: bool = True) -> None:
        if key in self.saved and self.saved[key] != value:
            raise ValueError("A saved decision or execution receipt cannot be replaced.")
        self.saved[key] = deepcopy(value)
        self.run.receipts = deepcopy(self.saved)
        self.run.updated_at = now()
        if commit:
            self.db.commit()

    def event(
        self,
        key: str,
        stage: str,
        kind: str,
        summary: str,
        detail: dict[str, Any] | None = None,
        *,
        commit: bool = True,
    ) -> None:
        from backend.analysis.service import emit

        emit(self.db, self.run, key, stage, kind, summary, detail)
        if commit:
            self.db.commit()

    def accepted(self, checkpoint: str) -> CheckpointReceipt | None:
        return next(
            (
                r
                for r in reversed(receipts(self.run))
                if r.checkpoint == checkpoint and r.status == "accepted" and r.complete
            ),
            None,
        )

    def start(
        self, checkpoint: str, attempt: int, packet: dict[str, Any], *, key: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        identity = key or f"{checkpoint}:{attempt}"
        saved = self.saved.get("input:" + identity)
        if saved:
            if saved["hash"] != fingerprint(packet):
                raise ValueError("A resumed task differs from its original evidence packet.")
            return dict(saved)
        prerequisites = []
        for name in BY_ID[checkpoint].prerequisites:
            receipt = self.accepted(name)
            if receipt:
                prerequisites.append(receipt.id)
            elif checkpoint == "report":
                failed = next((r for r in reversed(receipts(self.run)) if r.checkpoint == name), None)
                if failed:
                    prerequisites.append(failed.id)
            else:
                raise ValueError(f"Checkpoint {checkpoint} requires accepted {name}.")
        task_dependencies = set(packet.get("_prerequisite_ids", []))
        tasks.require_dependencies(self.db, self.run, task_dependencies)
        prerequisites.extend(sorted(task_dependencies - set(prerequisites)))
        value = {
            "hash": fingerprint(packet),
            "checkpoint": checkpoint,
            "packet": packet,
            "started_at": now().isoformat(),
            "prerequisites": prerequisites,
        }
        self.run.stage = checkpoint
        self.remember("input:" + identity, value, commit=False)
        self.event(
            "start:" + identity,
            checkpoint,
            "task_started",
            BY_ID[checkpoint].question,
            {"input_hash": value["hash"], "prerequisites": prerequisites},
            commit=commit,
        )
        return value

    def finish(
        self,
        checkpoint: str,
        attempt: int,
        packet: dict[str, Any],
        output: dict[str, Any],
        checks: list[GateCheck],
        *,
        key: str | None = None,
        terminal: str | None = None,
        complete: bool = True,
        commit: bool = True,
    ) -> CheckpointReceipt:
        identity = key or f"{checkpoint}:{attempt}"
        record_key = "checkpoint:" + identity
        if record_key in self.saved:
            return CheckpointReceipt.model_validate(self.saved[record_key])
        started = self.start(checkpoint, attempt, packet, key=identity, commit=commit)
        passed = all(check.passed for check in checks)
        status = terminal or ("accepted" if passed else "rejected")
        tree = get_current_run_tree()
        offered = [item["key"] for item in packet.get("evidence_records", [])]
        receipt = CheckpointReceipt.model_validate(
            {
                "id": fingerprint(
                    {
                        "run": str(self.run.id),
                        "key": identity,
                        "output": output,
                        "checks": [item.model_dump() for item in checks],
                    }
                ),
                "checkpoint": checkpoint,
                "attempt": attempt,
                "status": status,
                "complete": complete,
                "input_hash": started["hash"],
                "output_hash": fingerprint(output),
                "input_refs": ["question", *offered, *packet.get("decision_refs", [])],
                "prerequisite_ids": started["prerequisites"],
                "offered": offered,
                "retrieved": list(
                    dict.fromkeys(
                        [*packet.get("_retrieved", []), *[
                            key for name, value in self.saved.items() if name.startswith("read:")
                            for key in value.get("included_keys", [])
                        ]]
                    )
                ),
                "cited": sorted(domain.references(output)),
                "checks": [item.model_dump() for item in checks],
                "output": output,
                "assumptions": output.get("assumptions", []),
                "started_at": started["started_at"],
                "completed_at": now(),
                "budgets": budgets(self.run),
                "evidence_scope": packet.get("evidence_scope"),
                "trace_id": str(tree.id) if tree else None,
                "trace_url": tree.get_url() if tree and tree.client else None,
            }
        )
        if status == "accepted" and not passed:
            raise ValueError("A rejected gate cannot produce an accepted checkpoint.")
        self.remember(record_key, receipt.model_dump(mode="json"), commit=False)
        if status in {"blocked", "exhausted"}:
            self.run.status = "blocked"
        kind = (
            "gate_accepted"
            if status == "accepted"
            else "gate_blocked"
            if status in {"blocked", "exhausted"}
            else "submission_rejected"
        )
        from backend.analysis.authority_view import summary

        self.event(
            record_key,
            checkpoint,
            kind,
            str(output.get("summary", output.get("error", status))),
            {"receipt_id": receipt.id, "status": status, "checks": [item.model_dump() for item in checks],
             **({"claim_authority": summary(output)} if self.run.inputs.get("frozen_context") else {})},
            commit=commit,
        )
        return receipt

    def ask(self, key: str, schema: type[T], prompt: str, payload: dict[str, Any]) -> T:
        cache = "model:" + key
        if cache in self.saved:
            return schema.model_validate(self.saved[cache])
        if "model_error:" + key in self.saved:
            raise ValueError(str(self.saved["model_error:" + key]))
        model_payload = {key: value for key, value in payload.items() if not key.startswith("_")}
        frozen = self.run.inputs.get("frozen_context")
        if isinstance(frozen, dict):
            model_payload["frozen_inputs"] = {name: frozen[name] for name in (
                "dataset_version_id", "source_sha256", "context_revision_id", "context_hash",
                "readiness_fingerprint", "policy_version",
            )}
        semantic = bool(model_payload.get("column_identities"))
        offered, mapping, choices = references.selections(model_payload) if semantic else (model_payload, {}, {})
        decoder = None
        if schema is Treatments:
            choices["_explicit_fields"] = ["treatments"]
            choices["_json_schema"] = ["treatments"]
            choices["_compact_fields"] = ["treatments"]
            accepted_target = offered.get("decisions", {}).get("target", {})
            decoder = effect_references.decoder(accepted_target, offered.get("decision_refs", []))
            choices["_effect_refs"] = [item["id"] for item in accepted_target["requested_effects"]]
        if isinstance(frozen, dict):
            from backend.analysis import authority

            choices["_authority_policy"] = [authority.POLICY]
            if schema is Target:
                choices["_source_fields"] = ["target"]
                for predicate in ("measurement_definition", "measurement", "population", "unit", "timing", "requested_effect"):
                    choices["_source:" + predicate] = [
                        item["key"] for item in offered.get("evidence_records", [])
                        if predicate in item.get("source_predicates", []) and item.get("source_selectable")
                    ]
                choices["_source:requested_effect"].append("question")
            snapshot = offered.get("snapshot", {})
            offered_prior = offered.get("decisions", {**snapshot.get("frame", offered.get("frame", {})),
                                                       "plan": snapshot.get("plan", {})})
            choices["_claim_refs"] = ["question", *[item["key"] for item in offered.get("evidence_records", [])],
                                      *["assumption:" + identity for identity in packets.assumption_ids(offered_prior)]]
            decoder = authority.proposal_decoder(base_decoder=decoder)
            offered = packets.model_view(offered)
        thinking = provider.thinking_config(str(self.run.inputs["model"]), references.output_limit(key))
        request = provider.prepare(schema, prompt, offered, str(self.run.inputs["model"]),
            max_output_tokens=references.output_limit(key), choices=choices,
            thinking_budget=thinking["thinking_budget"] if thinking else None, reference_decoder=decoder)
        encoded_schema = request.schema
        schema_format = request.schema_format
        characters = len(prompt) + packets.size(offered) + packets.size(encoded_schema)
        if characters > packets.MODEL_INPUT_LIMIT:
            raise packets.PacketOverflow(
                f"The complete model input needs {characters} characters, above its {packets.MODEL_INPUT_LIMIT} limit. "
                "No model call was made and no evidence was truncated. Narrow the task or review its context."
            )
        parts = key.split(":")
        obligation = ":".join(parts[:2]) if parts[0] in {"program", "review", "diagnostic"} else parts[0]
        tree = get_current_run_tree()
        trace = {"trace_id": str(tree.id) if tree else None, "trace_url": tree.get_url() if tree and tree.session_name else None}
        if decoder is not None:
            trace["reference_decoding"] = "accepted_target_effect_id"
        try:
            reservation = tasks.reserve_shared(self.db, self.run, key, fingerprint(offered), {"input_characters": characters, "aliases": mapping, "thinking_config": thinking, "thinking_limit_kind": "soft" if thinking else "provider_default", "provider_schema_format": schema_format, **trace, "request": {"system": prompt, "payload": offered, "schema": encoded_schema, "schema_format": schema_format, "aliases": mapping, "output_limit": references.output_limit(key), "thinking_config": thinking, **({"reference_decoding": "accepted_target_effect_id"} if decoder is not None else {})}},
                obligation_key=obligation, obligation_limit=2,
                recover=self.shared_owner is not None, owner=self.shared_owner)
        except tasks.AllowanceExhausted as error:
            # Preserve an uncertain predecessor and its event even when no
            # allowance remains for a replacement provider call.
            self.db.commit()
            raise BudgetExhausted(str(error)) from error
        number = self.run.calls
        self.event(
            f"model_start:{number}",
            self.run.stage,
            "model_started",
            f"Resolving {key} from its pinned evidence packet.",
            {
                "call": number,
                "provider_attempt_key": "shared:" + reservation,
                "input_hash": fingerprint(payload),
                "input_characters": characters,
                "input_limit": packets.MODEL_INPUT_LIMIT,
                "evidence_scope": payload.get("evidence_scope"),
            },
        )
        raw_proposal: str | None = None
        response_metrics: dict[str, Any] = {}
        try:
            value = cast(T, model_access.structured(request))
            raw_proposal, response_metrics = model_access.response_artifacts(value)
            value = schema.model_validate(references.translate(value.model_dump(mode="json"), mapping))
        except model_access.InvalidModelOutput as error:
            self.remember("model_error:" + key, str(error), commit=False)
            self.remember("invalid_submission:" + key, {"raw": error.raw, "error": str(error)}, commit=False)
            raw_proposal, response_metrics = model_access.response_artifacts(error)
            tasks.settle_shared(self.db, self.run.id, reservation, status="rejected", raw_output=raw_proposal, metrics=response_metrics)
            self.event(
                f"model_end:{number}",
                self.run.stage,
                "model_failed",
                "The structured response failed validation.",
                {"error": str(error)},
            )
            raise
        except Exception as error:
            error_raw, error_metrics = model_access.response_artifacts(error)
            if error_raw is not None:
                raw_proposal = error_raw
            response_metrics.update(error_metrics)
            tasks.settle_shared(self.db, self.run.id, reservation, status="failed",
                                raw_output=raw_proposal,
                                metrics={**response_metrics, "error_type": type(error).__name__, "error": str(error)[:4000]})
            self.event(f"model_end:{number}", self.run.stage, "model_failed",
                       "Provider invocation failed; its charged attempt is retained for bounded recovery.",
                       {"call": number, "error_type": type(error).__name__, "error": str(error)[:4000]})
            raise
        tasks.settle_shared(self.db, self.run.id, reservation, status="accepted", receipt=value.model_dump(mode="json"), raw_output=raw_proposal, metrics=response_metrics)
        self.remember(cache, value.model_dump(mode="json"), commit=False)
        self.event(
            f"model_end:{number}",
            self.run.stage,
            "model_completed",
            f"Structured proposal received for {key}.",
            {"call": number, "output_hash": fingerprint(value.model_dump(mode="json"))},
        )
        return schema.model_validate(value.model_dump())

    def review(
        self,
        phase: Literal["target", "plan", "report"],
        proposal: BaseModel,
        packet: dict[str, Any],
        gates: list[GateCheck],
    ) -> list[GateCheck]:
        """Review meaning only after structural acceptance, without editing the proposal.

        All proposal hashes share the existing review:<phase> allowance. A
        malformed review may use its one correction; neither hashes nor replay
        grant additional provider calls. Transport recovery remains in ask().
        """
        if phase not in {"target", "plan", "report"}:
            raise ValueError("Semantic review supports target, plan and report only.")
        if not all(gate.passed for gate in gates) or (isinstance(proposal, Decision) and proposal.status != "ready"):
            return gates
        candidate = proposal.model_dump(mode="json")
        base = f"review:{phase}:{fingerprint(candidate)}"
        payload = {
            **{key: deepcopy(value) for key, value in packet.items() if not key.startswith("_") and key != "correction"},
            "question": self.run.question,
            "review_phase": phase,
            "candidate": candidate,
        }
        verdict: CodeReview | None = None
        failure = ""
        identity = base
        for attempt in range(2):
            identity = base if attempt == 0 else base + ":correction"
            saved_error = self.saved.get("model_error:" + identity)
            if saved_error is not None:
                failure = str(saved_error)
            else:
                try:
                    verdict = self.ask(identity, CodeReview, str(self.run.inputs["semantic_review"]), payload)
                    break
                except model_access.InvalidModelOutput as error:
                    failure = str(error)
            payload = {
                **payload,
                "review_correction": references.correction(
                    {}, [{"rule": "typed_review", "passed": False, "detail": failure}], identity,
                ),
            }
        passed = verdict is not None and verdict.accepted and not verdict.issues
        detail = (
            f"Review {identity}: {verdict.summary}" + (" Issues: " + "; ".join(verdict.issues) if verdict.issues else "")
            if verdict is not None
            else f"Review {identity} could not establish a valid semantic judgment after one correction: {failure}"
        )
        return [*gates, GateCheck(rule="semantic_review_" + phase, kind="semantic", passed=passed, detail=detail)]

    def decide(
        self,
        checkpoint: str,
        schema: type[T],
        packet: dict[str, Any],
        validator: Callable[[T], list[GateCheck]],
        *,
        prompt: str,
        key: str | None = None,
        complete: bool = True,
        normalize: Callable[[T], T] | None = None,
    ) -> T | None:
        base = key or checkpoint
        if "decision:" + base in self.saved:
            return schema.model_validate(self.saved["decision:" + base])
        mapping = {item["reference"]: item["key"] for item in packet.get("evidence_records", [])}
        for attempt in range(2):
            identity = f"{base}:{attempt}"
            prior_record = self.saved.get("checkpoint:" + identity)
            if prior_record:
                if prior_record["status"] in {"blocked", "exhausted"}:
                    return None
                packet = {
                    **packet,
                    "correction": references.correction(prior_record["output"], prior_record["checks"], prior_record["id"]),
                }
                continue
            self.start(checkpoint, attempt, packet, key=identity)
            output: dict[str, Any] = {}
            try:
                proposal = self.ask(identity, schema, prompt, packet)
                if isinstance(proposal, Decision):
                    snapshot = packet.get("snapshot", {})
                    offered_prior = packet.get("decisions", {
                        **snapshot.get("frame", packet.get("frame", {})),
                        "plan": snapshot.get("plan", {}),
                    })
                    proposal = schema.model_validate(domain.resolve(proposal, mapping, known_assumptions=packets.assumption_ids(offered_prior)).model_dump())
                if normalize:
                    proposal = normalize(proposal)
                output = proposal.model_dump(mode="json")
                gates = validator(proposal)
                deterministic_ok = all(g.passed for g in gates if g.kind == "deterministic")
                status = (
                    "blocked"
                    if deterministic_ok and isinstance(proposal, Decision) and proposal.status != "ready"
                    else None
                )
                if all(g.passed for g in gates):
                    self.remember("decision:" + base, output, commit=False)
                receipt = self.finish(
                    checkpoint,
                    attempt,
                    packet,
                    output,
                    gates,
                    key=identity,
                    terminal=status,
                    complete=complete,
                )
                if receipt.status == "accepted":
                    return proposal
                if receipt.status == "blocked":
                    return None
            except ValueError as error:
                if isinstance(error, model_access.InvalidModelOutput):
                    output = {"invalid_submission": error.raw}
                gates = [
                    GateCheck(rule="typed_submission", kind="deterministic", passed=False, detail=str(error))
                ]
                status = (
                    "exhausted"
                    if isinstance(error, BudgetExhausted)
                    else "blocked"
                    if isinstance(error, packets.PacketOverflow)
                    else None
                )
                self.finish(
                    checkpoint,
                    attempt,
                    packet,
                    {"error": str(error), **output},
                    gates,
                    key=identity,
                    terminal=status,
                    complete=complete,
                )
                if status:
                    return None
            packet = {
                **packet,
                "correction": references.correction(output, [g.model_dump() for g in gates], identity),
            }
        self.run.status = "blocked"
        self.run.error = (
            "The checkpoint failed its acceptance gate after one correction. Inspect its saved attempts."
        )
        self.db.commit()
        return None



