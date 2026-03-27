"""Chat execution graph. Checkpoints route recovery; immutable receipts prevent repeated work."""

import re
from typing import Any, TypedDict, cast
from uuid import UUID, uuid5

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import model_access, numerical
from backend.chat import evidence, exploration, prompts
from backend.chat.contracts import Interpretation
from backend.chat.service import artifact_record
from backend.context.service import read_context
from backend.handoffs import AnalysisBinding, AnalysisDiscussionReader, AnalysisMaterial
from backend.models import ChatArtifact, ChatTurn, Conversation, DatasetVersion, now
from backend.responses import Reply
from backend.storage import files


class State(TypedDict):
    turn_id: str


class Runtime:
    def __init__(self, db: Session, turn: ChatTurn, analysis_reader: AnalysisDiscussionReader):
        self.db, self.turn = db, turn
        self.analysis_reader = analysis_reader
        conversation = db.get(Conversation, turn.conversation_id)
        assert conversation is not None
        self.conversation = conversation
        version = db.get(DatasetVersion, conversation.version_id)
        assert version is not None
        self.version = version
        selected_revision = turn.inputs.get("context_revision_id", conversation.context_revision_id)
        self.context_revision_id = UUID(str(selected_revision)) if selected_revision else None
        self.receipts: dict[str, Any] = dict(turn.receipts)
        self.context = (
            read_context(db, version.id, self.context_revision_id, []).model_dump(mode="json")
            if self.context_revision_id
            else {}
        )
        previous = list(
            db.scalars(
                select(ChatTurn)
                .where(ChatTurn.conversation_id == conversation.id, ChatTurn.number < turn.number)
                .order_by(ChatTurn.number)
            )
        )
        self.history = [
            {
                "turn": row.number,
                "user": row.text,
                "assistant": row.reply,
                "status": row.status,
                "error": row.error,
            }
            for row in previous
        ]
        self.artifacts = list(
            db.scalars(
                select(ChatArtifact)
                .join(ChatTurn)
                .where(ChatTurn.conversation_id == conversation.id, ChatTurn.number < turn.number)
                .order_by(ChatTurn.number)
            )
        )
        saved_bindings = turn.inputs.get("analysis_bindings")
        self.analyses = analysis_reader.views(db, conversation.id,
            [UUID(str(a)) for a in cast(list[str], turn.inputs.get("analysis_ids", []))],
            cast(dict[str, AnalysisBinding], saved_bindings) if isinstance(saved_bindings, dict) else None)

    def remember(self, key: str, value: Any) -> None:
        self.receipts[key] = value
        self.turn.receipts = dict(self.receipts)
        self.turn.updated_at = now()
        self.db.commit()

    def ask(
        self, key: str, schema: type[BaseModel], prompt: str, payload: dict[str, object], images: list[bytes],
        *, encoded_schema: dict[str, Any] | None = None,
    ) -> BaseModel:
        receipt = "model:" + key
        if receipt in self.receipts:
            return schema.model_validate(self.receipts[receipt])
        invalid = self.receipts.get("invalid_model:" + key)
        if invalid:
            raise ValueError(str(invalid["error"]))
        if self.turn.calls >= 24:
            raise ValueError(
                "This turn exhausted its 24-call allowance. Send a new message; previous work remains recorded."
            )
        self.turn.calls += 1
        self.turn.stage, self.turn.updated_at = key, now()
        self.db.commit()  # Reserve before the provider, including calls interrupted by process exit.
        try:
            value = model_access.structured(model_access.prepare(
                schema, prompt, payload, str(self.turn.inputs["model"]), images=images, encoded_schema=encoded_schema
            ))
        except model_access.InvalidModelOutput as error:
            self.remember("invalid_model:" + key, {"raw": error.raw, "error": str(error)})
            raise
        self.remember(receipt, value.model_dump(mode="json"))
        return value

    def packet(self, *, analyses: list[AnalysisMaterial] | None = None, catalogue: bool = False) -> dict[str, object]:
        return {
            "mode": self.turn.inputs.get("mode", "chat"),
            "current_message": self.turn.text,
            "history": self.history,
            "dataset": {
                "version_id": str(self.version.id),
                "sha256": self.version.sha256,
                "columns": self.version.columns,
                "rows": self.version.row_count,
            },
            "context": self.context,
            "artifacts": [{"id": str(a.id), **a.record} for a in self.artifacts],
            "selected_artifact_ids": self.turn.inputs["artifact_ids"],
            "analyses": [
                {
                    "id": str(a.id),
                    "question": a.question,
                    "artifact_files": list(a.binding["files"]),
                    "status_at_submission": a.binding["status"],
                    **({"result_available": a.result is not None} if catalogue else {"snapshot": a.snapshot, "result": a.result}),
                }
                for a in (self.analyses if analyses is None else analyses)
            ],
        }

    def interpret(self, state: State) -> dict[str, object]:
        if self.turn.inputs["contract"] not in {
            "chat-exploration-2", "chat-exploration-3", "chat-exploration-4", "chat-exploration-5", "chat-exploration-7", "chat-exploration-8", "chat-exploration-9", prompts.VERSION,
        }:
            raise ValueError("This turn uses an older execution contract. Start a new message.")
        value = Interpretation.model_validate(
            self.ask(
                "interpret", Interpretation, str(self.turn.inputs["interpret"]), self.packet(catalogue=True), [],
                encoded_schema=model_access.encode_schema(Interpretation.model_json_schema(), choices={
                    "artifact_ids": [str(item.id) for item in self.artifacts],
                    "analysis_ids": [str(item.id) for item in self.analyses],
                    "analysis_id": [str(item.id) for item in self.analyses],
                    "filename": sorted({name for item in self.analyses for name in item.binding["files"]}),
                    **({"analysis_files": []} if not self.analyses else {}),
                }),
            ).model_dump()
        )
        if value.exploration and (not value.passage.strip() or value.passage not in self.turn.text):
            raise ValueError(
                "The visualization request was not supported by an exact passage from this message."
            )
        if self.turn.inputs.get("mode") == "context" and value.exploration:
            raise ValueError("Context discussion does not execute code.")
        permitted = {a.id for a in self.artifacts}
        if not set(value.artifact_ids) <= permitted:
            raise ValueError("The model selected an artifact outside this conversation.")
        analysis_records = {record.id: record for record in self.analyses}
        if not set(value.analysis_ids) <= analysis_records.keys():
            raise ValueError("Select saved analyses from this turn's exact analysis catalogue.")
        selected_files = []
        total = 0
        for selected in value.analysis_files:
            record = analysis_records.get(selected.analysis_id)
            if record is None or selected.filename not in record.binding["files"]:
                raise ValueError("Selected analysis artifact is outside the turn's bound analysis records.")
            artifact = self.analysis_reader.file(self.db, self.conversation.id, record, selected.filename)
            raw = artifact.raw
            total += len(raw)
            if total > 20 * 1024 * 1024:
                raise ValueError("Selected analysis artifacts exceed the turn's input allowance.")
            selected_files.append(
                {
                    "analysis_id": str(record.id),
                    "filename": selected.filename,
                    "sha256": artifact.sha256,
                }
            )
        self.remember("analysis_files", selected_files)
        return {}

    def visualize(self, state: State) -> dict[str, object]:
        interpreted = Interpretation.model_validate(self.receipts["model:interpret"])
        if not interpreted.exploration:
            return {}
        selected = {UUID(str(value)) for value in cast(list[str], self.turn.inputs["artifact_ids"])} | set(
            interpreted.artifact_ids
        )
        supplied: dict[str, bytes] = {}
        for aid in selected:
            artifact = artifact_record(self.db, self.conversation.id, aid)
            for name, meta in artifact.files.items():
                if name.endswith((".csv", ".json", ".png", ".txt")) and isinstance(meta, dict):
                    supplied[f"artifact_{aid}/{name}"] = files().read(str(meta["key"]), str(meta["sha256"]))
        bound = {str(record.id): record for record in self.analyses}
        for selected_file in self.receipts.get("analysis_files", []):
            supplied[f"analysis_{selected_file['analysis_id']}/{selected_file['filename']}"] = self.analysis_reader.file(
                self.db, self.conversation.id, bound[selected_file["analysis_id"]], selected_file["filename"]
            ).raw
        if sum(map(len, supplied.values())) > 20 * 1024 * 1024:
            raise ValueError("Selected artifacts exceed the 20 MiB input budget. Select fewer artifacts.")
        exploration.explore(
            self.turn.id,
            interpreted.exploration,
            self.context,
            files().read(self.version.storage_key, self.version.sha256),
            supplied,
            self.turn.inputs,
            self.receipts,
            self.ask,
            self.remember,
        )
        return {}

    def respond(self, state: State) -> dict[str, object]:
        interpreted = Interpretation.model_validate(self.receipts["model:interpret"])
        selected = set(interpreted.analysis_ids) | {item.analysis_id for item in interpreted.analysis_files}
        # Explicit user references are selections too; never reinterpret an identity.
        selected.update(item.id for item in self.analyses if str(item.id) in self.turn.text)
        analyses = [item for item in self.analyses if item.id in selected] if self.turn.inputs["contract"] in {"chat-exploration-9", prompts.VERSION} else self.analyses
        numeric = evidence.sources(
            self.context,
            [{"id": str(a.id), **a.record} for a in self.artifacts],
            self.receipts.get("exploration"),
            self.history,
            self.turn.text,
            self.version.row_count,
        )
        for analysis in analyses:
            numeric[f"analysis.{analysis.id}.question"] = {
                "basis": "prespecified_user_request",
                "value": analysis.question,
            }
            if result := analysis.result:
                facts = result.get("facts", {})
                if isinstance(facts, dict):
                    numeric.update(
                        {
                            f"analysis.{analysis.id}.{key}": {"basis": "computed", "value": value}
                            for key, value in facts.items()
                        }
                    )
        analysis_records = {str(record.id): record for record in analyses}
        file_evidence = []
        analysis_images = []
        for selected_file in self.receipts.get("analysis_files", []):
            artifact = self.analysis_reader.file(self.db, self.conversation.id,
                analysis_records[selected_file["analysis_id"]], selected_file["filename"], expected_hash=selected_file["sha256"])
            raw = artifact.raw
            if selected_file["filename"].endswith(".png"):
                analysis_images.append(raw)
                file_evidence.append({**selected_file, "content": "Attached image"})
            else:
                text = raw[:64000].decode("utf-8", errors="replace")
                file_evidence.append(
                    {
                        **selected_file,
                        "content": text,
                        "truncated": len(raw) > 64000,
                    }
                )
                if selected_file["filename"].endswith(".py") and len(raw) <= 64000:
                    numeric.update(
                        numerical.program_literals(
                            f"analysis.{selected_file['analysis_id']}.file.{selected_file['filename']}",
                            text,
                        )
                    )
                numeric.update(
                    {
                        f"analysis.{selected_file['analysis_id']}.file.{selected_file['filename']}.{key}": {
                            "basis": "computed",
                            "value": value,
                        }
                        for key, value in artifact.facts.items()
                    }
                )
        analysis_references = {
            *[f"analysis.{record.id}" for record in analyses],
            *[key for key in numeric if key.startswith("analysis.")],
            *[f"analysis.{record['analysis_id']}.file.{record['filename']}" for record in file_evidence],
        }
        # The model selects short packet-local keys. Persist exact canonical references,
        # without asking it to repeatedly transcribe UUIDs and code positions.
        numeric_aliases, numeric_options = evidence.numeric_options(numeric)
        analysis_aliases = {
            f"a{index}": key
            for index, key in enumerate(
                sorted(
                    {f"analysis.{record.id}" for record in analyses}
                    | {
                        f"analysis.{record['analysis_id']}.file.{record['filename']}"
                        for record in file_evidence
                    }
                ),
                1,
            )
        }
        packet = {
            **self.packet(analyses=analyses),
            "analysis_file_evidence": file_evidence,
            "exploration_result": self.receipts.get("exploration"),
            "numeric_evidence": numeric_options,
            "analysis_references": analysis_aliases,
        }
        for attempt in range(2):
            key = "reply" if attempt == 0 else "reply-correction"
            reply: Reply | None = None
            try:
                reply = Reply.model_validate(
                    self.ask(
                        key, Reply, str(self.turn.inputs["discuss"]), packet, analysis_images
                    ).model_dump()
                )
                reply = numerical.render_handles(reply)
                for entry in reply.numbers:
                    if entry.source in numeric_options:
                        numerical.validate(Reply(text=str(entry.value), numbers=[entry]), {entry.source: numeric_options[entry.source]})
                inline_refs = set(re.findall(r"<(a\d+)>", reply.text))
                if not inline_refs <= analysis_aliases.keys() & set(reply.analysis_references):
                    raise ValueError("Inline analysis handles must select offered analysis references.")
                reply.text = re.sub(r"<(a\d+)>", "saved analysis", reply.text)
                reply.numbers = [
                    entry.model_copy(update={"source": numeric_aliases.get(entry.source, entry.source)})
                    for entry in reply.numbers
                ]
                reply.analysis_references = [
                    analysis_aliases.get(key, key) for key in reply.analysis_references
                ]
                # Namespaces are authoritative: route exact model-selected analysis keys
                # without treating them as context edits or inventing a source address.
                errors = []
                selected_analysis_refs = set(reply.analysis_references) | {
                    key for key in reply.context_keys if key.startswith("analysis.")
                }
                if not selected_analysis_refs <= analysis_references:
                    errors.append(
                        "Analysis references must select the exact turn-bound analysis catalogue. "
                        f"Unknown references: {sorted(selected_analysis_refs - analysis_references)}"
                    )
                reply.analysis_references = sorted(selected_analysis_refs)
                reply.context_keys = [key for key in reply.context_keys if not key.startswith("analysis.")]
                reply = numerical.bind_literals(
                    reply,
                    {
                        key: str(value["value"])
                        for key, value in numeric.items()
                        if value["basis"] == "prespecified_user_request"
                    },
                )
                try:
                    numerical.validate(reply, numeric)
                except ValueError as error:
                    errors.append(str(error))
                try:
                    self.validate_proposal(reply)
                except ValueError as error:
                    errors.append(str(error))
                try:
                    if reply.context_keys:
                        if not self.context_revision_id:
                            raise ValueError("The reply cited context without a selected revision.")
                        read_context(self.db, self.version.id, self.context_revision_id, reply.context_keys)
                except ValueError as error:
                    errors.append(str(error))
                if errors:
                    raise ValueError("; ".join(errors))
                break
            except ValueError as error:
                if attempt:
                    raise ValueError(
                        "The reply failed evidence verification after one correction. "
                        f"Send a new message. Details: {error}"
                    ) from error
                packet = {
                    **packet,
                    "rejected_reply": reply.model_dump()
                    if reply
                    else self.receipts.get("invalid_model:" + key),
                    "validation_error": str(error),
                }
        assert reply is not None
        self.turn.reply = reply.model_dump(mode="json")
        result = self.receipts.get("exploration")
        if result and result["exploration"]["status"] == "completed":
            identity = uuid5(self.turn.id, "artifact")
            if self.db.get(ChatArtifact, identity) is None:
                record = {key: value for key, value in result.items() if key != "files"}
                interpreted = Interpretation.model_validate(self.receipts["model:interpret"])
                record.update(
                    {
                        "dataset_version_id": str(self.version.id),
                        "context_revision_id": str(self.context_revision_id)
                        if self.context_revision_id
                        else None,
                        "source_sha256": self.version.sha256,
                        "parent_artifact_ids": sorted(
                            set(cast(list[str], self.turn.inputs["artifact_ids"]))
                            | {str(identity) for identity in interpreted.artifact_ids}
                        ),
                    }
                )
                self.db.add(
                    ChatArtifact(id=identity, turn_id=self.turn.id, record=record, files=result["files"])
                )
        self.turn.status, self.turn.stage, self.turn.error, self.turn.updated_at = (
            "succeeded",
            "complete",
            None,
            now(),
        )
        self.db.commit()
        return {}

    def validate_proposal(self, reply: Reply) -> None:
        from backend.context.contracts import ContextContent
        from backend.context.domain import apply_brief, apply_edits, validate_evidence
        from backend.context.service import revision_record, revision_sources
        from backend.models import Dataset

        if not reply.context_edits and reply.analysis_brief is None:
            if reply.proposal_passage:
                raise ValueError("A proposal passage must accompany actual proposed edits.")
            return
        if self.turn.inputs.get("mode") != "context" or not self.context_revision_id:
            raise ValueError("Context changes require Update context mode and a selected revision.")
        if not reply.proposal_passage.strip() or reply.proposal_passage not in self.turn.text:
            raise ValueError(
                "Proposed context changes need an exact supporting passage from the current message."
            )
        dataset = self.db.get(Dataset, self.version.dataset_id)
        assert dataset is not None
        content = apply_edits(
            ContextContent.model_validate(self.context["content"]),
            reply.context_edits,
            dataset.owner_id,
            uuid5(self.turn.id, "proposal-validation"),
        )
        if reply.analysis_brief is not None:
            content = apply_brief(
                content, reply.analysis_brief, reply.context_edits, dataset.owner_id,
                uuid5(self.turn.id, "proposal-validation"), source_text=reply.proposal_passage,
            )
        revision = revision_record(self.db, self.version.id, self.context_revision_id)
        sources = revision_sources(self.db, revision)
        validate_evidence(
            content, {str(s.id): files().read(s.storage_key, s.sha256).decode("utf-8") for s in sources}
        )


def build(runtime: Runtime, saver: BaseCheckpointSaver[Any]) -> Any:
    graph = StateGraph(State)
    graph.add_node("interpret", runtime.interpret)
    graph.add_node("visualize", runtime.visualize)
    graph.add_node("respond", runtime.respond)
    graph.add_edge(START, "interpret")
    graph.add_edge("interpret", "visualize")
    graph.add_edge("visualize", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=saver)
