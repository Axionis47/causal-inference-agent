"""Application adapter for the desk. PostgreSQL owns all conversational state."""

import hashlib
import json
import math
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from causal_agent.common import trace
from causal_agent.common.contracts import Handoff
from causal_agent.desk import material, pipeline
from causal_agent.explore import client as exploration
from causal_agent.memory import artifacts, ops
from causal_agent.memory.context.projection import bindings
from causal_agent.profile.profiler import profile
from causal_agent.server.conversations import schema as S
from causal_agent.server.conversations.commands import Repository
from causal_agent.server.conversations.publication import Repository as ArtifactRepository
from causal_agent.server.ingestion.service import Missing


class Runtime:
    def __init__(self, repository: Repository, settings):
        self.repo, self.settings = repository, settings
        self.conversation = repository.conversation
        self.csv = settings.root / "data" / "conversations" / self.conversation / "input.csv"
        self.artifacts = ArtifactRepository(repository.storage, repository.owner, self.conversation)

    def state(self):
        return self.repo.state()

    def operation(self, operation_id):
        with self.repo.storage.engine.connect() as db:
            return self.repo.operation(db, operation_id)

    def revision(self):
        return self.repo.context.read(self.state()["revision_id"])

    def original_account(self):
        return self.repo.chat["context"]

    def memory(self):
        revision = self.revision()
        memory = ops.seed(self.conversation, profile(self.csv), csv=str(self.csv))
        bindings(revision.knowledge, memory)
        # Human-reviewed column meanings are data semantics, not causal relationships.
        for col in revision.knowledge.columns:
            statements = [
                s
                for s in revision.knowledge.statements
                if s.scope == "column" and col.column_id in s.column_ids and s.status == "reviewed" and s.binding is None
            ]
            if statements:
                text = "; ".join(f"{s.field.replace('_', ' ')}: {s.value}" for s in statements)
                ops.apply(
                    memory,
                    [
                        ops.Update(
                            address=f"col:{col.key}.meaning",
                            value=text,
                            status="confirmed",
                            source="user:turn:" + revision.id,
                            said=text,
                            reason="Reviewed column statements",
                        )
                    ],
                )
        account = self.repo.chat["context"]
        if account.strip():
            ops.apply(
                memory,
                [
                    ops.Update(
                        address="claim:story.text",
                        value=account,
                        status="confirmed",
                        source="user:turn:" + self.state()["review_id"],
                        said=account,
                        verbatim=True,
                    )
                ],
            )
        memory.version = revision.number
        return memory

    def context_for(self, references):
        knowledge = self.revision().knowledge
        selected = [r.id for r in references if r.kind == "column"]
        if selected:
            return list({s.id: s for cid in selected for s in knowledge.for_column(cid)}.values())[:100]
        return [s for s in knowledge.statements if s.status != "withdrawn"][:100]

    def artifact_index(self):
        return [
            {"id": a.id, "summary": a.summary, "elements": [e.model_dump() for e in a.elements], "design": a.design}
            for a in self.artifacts.list(self.conversation, limit=30)
        ]

    def history(self):
        return [{"role": m["role"], "text": m["text"][:4000]} for m in reversed(self.repo.messages(limit=12))]

    def exchanges(self):
        from causal_agent.desk.contracts import Exchange

        result: list[Exchange] = []
        user = ""
        for message in self.history():
            if message["role"] == "user":
                user = message["text"]
            elif user:
                result.append(Exchange(turn=len(result) + 1, user=user, assistant=message["text"]))
                user = ""
        return result

    def reference(self, reference, revision_id):
        return self.repo.reference(reference, revision_id)

    def correct(self, *args):
        revision = self.repo.correct(*args)
        self.reassess()
        return revision

    def answer(self, *args):
        revision = self.repo.answer(*args)
        self.reassess()
        return revision

    def controls(self, *args):
        message = self.repo.controls(*args)
        if self.operation(args[0])["body"]["kind"] == "refresh":
            self.reassess()
        return message

    def set_question_frame(self, frame):
        with self.repo.storage.engine.begin() as db:
            self.repo.update(db, frame=frame.model_dump(mode="json"))

    def set_readiness(self, status, missing=0):
        with self.repo.storage.engine.begin() as db:
            self.repo.update(db, readiness={"status": status, "missing": missing})

    def reassess(self):
        from causal_agent.common.contracts import QuestionFrame
        from causal_agent.desk.graph import assess

        frame = self.state()["frame"]
        if frame:
            assess(self, QuestionFrame.model_validate(frame))

    def finish(self, *args):
        return self.repo.finish(*args)

    def save_interpretation(self, operation_id, plan):
        self.save_progress(operation_id, interpretation=plan)

    def save_progress(self, operation_id, **values):
        with self.repo.storage.engine.begin() as db:
            self.repo.state(db, lock=True)
            operation = self.repo.operation(db, operation_id)
            db.execute(S.commands.update().where(S.commands.c.id == operation_id).values(result={**(operation["result"] or {}), **values}))

    def pause_for_topic(self, continuing):
        with self.repo.storage.engine.begin() as db:
            self.repo.update(db, fast_track=continuing)

    def set_preparation(self, questions, handoff):
        with self.repo.storage.engine.begin() as db:
            state = self.repo.state(db, lock=True)
            pending = []
            for question in questions:
                if question.revision_id != state["revision_id"]:
                    raise ValueError("Context changed while preparing questions.")
                db.execute(
                    insert(S.questions)
                    .values(
                        id=question.id,
                        owner_id=self.repo.owner,
                        conversation_id=self.conversation,
                        revision_id=question.revision_id,
                        target=question.target,
                        body=question.model_dump(),
                    )
                    .on_conflict_do_nothing(index_elements=["conversation_id", "revision_id", "target"])
                )
                pending.append(question.id)
            preparation = None
            if handoff:
                handoff.design_id = (
                    db.scalar(
                        select(func.count())
                        .select_from(S.commands)
                        .where(
                            S.commands.c.conversation_id == self.conversation,
                            S.commands.c.owner_id == self.repo.owner,
                            S.commands.c.body["kind"].as_string() == "authorize",
                        )
                    )
                    or 0
                ) + 1
                raw = handoff.model_dump(mode="json")
                identity = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
                preparation = {
                    "id": identity,
                    "revision_id": state["revision_id"],
                    "handoff": raw,
                    "family": handoff.family,
                    "question": state["question"],
                    "assumptions": handoff.chosen_assumption,
                }
            self.repo.update(db, pending=pending, preparation=preparation)

    def material(self, refs):
        mat = material.Material()
        mat.add("chat.question", self.state()["question"])
        analyses = [self.artifacts.load(self.conversation, r.id) for r in refs if r.kind == "artifact"]
        analyses = [a for a in analyses if a.origin == "analysis"]
        if len(analyses) > 1 and (
            any(not a.comparison_basis for a in analyses) or any(a.comparison_basis != analyses[0].comparison_basis for a in analyses[1:])
        ):
            mat.add(
                "comparison.policy",
                "These causal results do not establish matching targets, populations, contrasts, scales and support. Do not compare their numbers. Select one result to discuss its evidence, or select results with matching definitions.",
            )
            mat.close()
            return mat
        for s in self.context_for(refs):
            passages = "\n".join(c.quote for c in s.citations)
            mat.add(
                "context:" + s.id,
                f"{s.value} ({s.status}). {s.qualifications} {s.conditions} {s.timing}\nSource passages: {passages}\nSupplied correction: {s.source.text}",
            )
        for ref in refs:
            self.reference(ref, self.state()["revision_id"])
            if ref.kind != "artifact":
                continue
            a = self.artifacts.load(self.conversation, ref.id)
            mat.add(a.address, a.render())
            for key, value in a.facts.items():
                mat.add(f"{a.address}.{key}", f"{key}: {value}", value)
            # Load actual registered evidence on demand, not only list thumbnails.
            for name in a.files:
                if name in {a.table, "facts.json", "result.json", "figures.json", "analysis.json"}:
                    raw = self.artifacts.read_file(a.id, name)
                    element = next((e for e in a.elements if e.id == ref.element and e.file == name), None)
                    if element and element.kind == "table_column":
                        from causal_agent.memory.artifact_tables import rows as table_rows

                        rows = table_rows(name, raw)
                        index = int(element.selector)
                        raw = json.dumps(
                            {
                                "column": element.label,
                                "rows_total": max(0, len(rows) - 1),
                                "preview_only": True,
                                "first_rows": [r[index] for r in rows[1:201] if len(r) > index],
                            }
                        ).encode()
                    elif element and element.kind == "chart_series":
                        i, j = (int(x) for x in element.selector.split(":"))
                        figure = json.loads(raw)[i]
                        raw = json.dumps({**figure, "series": [figure["series"][j]]}).encode()
                    if len(raw) <= 200000:
                        mat.add(f"{a.address}.file.{name}", raw.decode("utf-8"))
                    else:
                        mat.add(f"{a.address}.file.{name}", "This file is too large for one discussion request. Select a supported element to narrow it.")
        mat.close()
        return mat

    def products_for(self, operation_id):
        from causal_agent.memory.context.conversation import Reference

        request_id = str(uuid5(NAMESPACE_URL, operation_id + ":exploration-request"))
        with self.repo.storage.engine.connect() as db:
            records = db.scalars(
                select(S.results.c.record)
                .join(S.attempts, S.attempts.c.id == S.results.c.attempt_id)
                .where(S.attempts.c.request_id == request_id, S.results.c.owner_id == self.repo.owner, S.results.c.conversation_id == self.conversation)
            )
            return [Reference(kind="artifact", id=r["id"]) for r in records]

    def explore_once(self, operation_id, request, references):
        progress = self.operation(operation_id).get("result") or {}
        if progress.get("exploration_result"):
            return progress["exploration_result"]
        request_id = str(uuid5(NAMESPACE_URL, operation_id + ":exploration-request"))
        with self.repo.storage.engine.connect() as db:
            saved = db.scalar(
                select(S.results.c.record)
                .join(S.attempts, S.attempts.c.id == S.results.c.attempt_id)
                .where(S.attempts.c.request_id == request_id, S.results.c.owner_id == self.repo.owner, S.results.c.conversation_id == self.conversation)
            )
        if saved:
            return saved["summary"]
        if progress.get("exploration_started"):
            return "The previous visualization attempt was interrupted. It has not been resubmitted. Check its recorded attempt before requesting a new visualization."
        self.save_progress(operation_id, exploration_started=True)
        self.artifacts.operation_id = operation_id
        memory = self.memory()
        with artifacts.using(self.artifacts):
            inputs = exploration.artifact_inputs(self.conversation, [r.id for r in references if r.kind == "artifact"], self.settings.root)
            product, decline, _ = exploration.explore(
                dataset=self.conversation,
                ask=request,
                context="\n".join(s.value for s in self.context_for(references)),
                csv=self.csv,
                design=None,
                memory_version=memory.version,
                data_version=memory.data_version,
                artifact_refs=inputs,
                thread_id=self.conversation,
                root=self.settings.root,
            )
        text = product.summary if product else (decline.reason if decline else "Visualization produced no published artifact.")
        self.save_progress(operation_id, exploration_result=text)
        return text

    def run_authorized(self, command):
        state = self.state()
        preparation = state["preparation"]
        if not preparation or preparation["id"] != command.snapshot_id or preparation["revision_id"] != state["revision_id"]:
            raise ValueError("The analysis review is stale. Prepare and review the current context before running.")
        progress = self.operation(command.submission_id).get("result") or {}
        if progress.get("analysis_result"):
            return progress["analysis_result"]
        handoff = Handoff.model_validate(preparation["handoff"])
        directory = self.settings.root / "data" / "conversations" / self.conversation / "analyses" / command.submission_id
        directory.mkdir(parents=True, exist_ok=True)
        if progress.get("analysis_started"):
            result_file = directory / "result.json"
            if not result_file.exists():
                return "Analysis was interrupted without a completed result. It has not been restarted. Prepare and authorize a new attempt when ready."
            result = json.loads(result_file.read_text())
            record = pipeline.record(
                self.conversation, state["question"], handoff.design_id, handoff, result, {}, "Recovered completed analysis", str(directory)
            )
        else:
            from causal_agent.memory.context.snapshot import publish

            reference = publish(directory, handoff, state["revision_id"])
            self.save_progress(command.submission_id, analysis_started=True, snapshot=str(reference))
            record = pipeline.run(reference, handoff.design_id, self.conversation, state["question"], thread_id=self.conversation, snapshot=True)
        trace.write(directory / "record.json", record.model_dump_json(indent=2))
        self.publish_analysis(command, preparation, record, directory)
        text = f"Analysis finished with status: {record.status}."
        if record.specialist_result and record.specialist_result.get("context_deficiency"):
            text += " Essential context was missing. This attempt has stopped. Review the deficiency and authorize a new snapshot after correcting it."
        self.save_progress(command.submission_id, analysis_result=text, record=record.model_dump(mode="json"))
        return text

    def publish_analysis(self, command, preparation, record, directory):
        """Catalogue publication follows result assembly and never reopens reasoning."""
        from causal_agent.memory.artifacts import Artifact

        product_id = uuid5(NAMESPACE_URL, command.submission_id + ":analysis-product").hex
        try:
            self.artifacts.load(self.conversation, product_id)
            return
        except Missing:
            pass
        progress = self.operation(command.submission_id).get("result") or {}
        request_id, attempt_id = progress.get("publication_request"), progress.get("publication_attempt")
        if not request_id:
            self.artifacts.operation_id = command.submission_id
            request_id, attempt_id, _ = self.artifacts.begin(
                self.conversation, self.state()["question"], "Authorized causal analysis", self.repo.context.info["sha256"], []
            )
            self.save_progress(command.submission_id, publication_request=request_id, publication_attempt=attempt_id)
        assert request_id is not None and attempt_id is not None
        result = record.specialist_result or {}
        handoff = Handoff.model_validate(preparation["handoff"])
        design = (result.get("designs") or {}).get(record.lead_design_id) or {}
        comparison = None
        if design.get("target"):
            comparison = {
                "dataset": handoff.dataset_identity,
                "outcome": handoff.outcome,
                "treatment": handoff.treatment,
                "endpoint": design.get("endpoint_id"),
                "target": design["target"],
                "contrast": design.get("contrasts", design.get("contrast")),
                "support": design.get("support", design.get("row_ids", design.get("rows"))),
                "weights": design.get("weights_policy", design.get("weight_policy", design.get("weights"))),
            }
        facts = {}
        for i, estimate in enumerate(result.get("estimates") or []):
            for field in ("value", "ci_low", "ci_high", "p_value"):
                value = estimate.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    facts[f"estimate_{i}_{field}"] = value
        trace.write(directory / "facts.json", json.dumps(facts))
        trace.write(directory / "result.json", json.dumps(result, default=str))
        trace.write(directory / "figures.json", json.dumps(record.figures, default=str))
        # A bounded projection of the actual assembled result, with explicit
        # truncation markers and hashes. Full evidence remains in result.json.
        from causal_agent.memory.artifact_tables import compact_evidence

        discussion = {
            key: result.get(key)
            for key in (
                "status",
                "lead_design_id",
                "lead_selection",
                "context_id",
                "context_deficiency",
                "estimates",
                "checks",
                "declines",
                "interpretations",
                "feasibility",
                "terminal",
                "execution_attempts",
                "comparisons",
            )
        }
        discussion["designs"] = {
            key: {
                field: design.get(field)
                for field in ("endpoint_id", "target", "contrast", "assumptions", "support", "weights_policy", "inference", "estimator")
            }
            for key, design in (result.get("designs") or {}).items()
        }
        trace.write(directory / "analysis.json", json.dumps(compact_evidence(discussion), default=str))
        created = progress.get("publication_time") or datetime.now(UTC).isoformat()
        self.save_progress(command.submission_id, publication_time=created)
        a = Artifact(
            id=product_id,
            dataset=self.conversation,
            owner_id=self.repo.owner,
            dataset_version_id=self.state()["version_id"],
            origin="analysis",
            format="result",
            design=record.index,
            memory_version=self.revision().number,
            data_version=0,
            ask=self.state()["question"],
            summary=f"Causal analysis: {record.status}",
            method=record.family or "No admitted family",
            findings=["The registered result preserves the requested target, lead selection and all recorded attempts."],
            limitations=["Interpret estimates together with the identification assumptions, support checks and limitations in the registered result."],
            tags=[record.family or "analysis"],
            source_sha256=self.repo.context.info["sha256"],
            parent_refs=[],
            conversation_id=self.conversation,
            request_id=request_id,
            attempt_id=attempt_id,
            facts=facts,
            files=["result.json", "record.json", "facts.json", "figures.json", "analysis.json"],
            made_at=created,
            comparison_basis=comparison,
        )
        self.artifacts.publish(a, directory)
