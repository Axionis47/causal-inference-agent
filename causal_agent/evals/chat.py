"""Opt-in Gemini chat acceptance on the recorded Uruguay study account.

Uses disposable PostgreSQL/private storage and the production application adapter.
All questions, replies, operations and trace trees are retained for semantic review.
"""

import argparse
import io
import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from pytest import MonkeyPatch

from causal_agent.common import trace
from causal_agent.memory.context.conversation import ChatCommand, Reference
from causal_agent.memory.context.review import ColumnDisposition
from causal_agent.server.context import service as context_service
from causal_agent.server.context.repository import Repository as Context
from causal_agent.server.conversations import service
from causal_agent.server.conversations.commands import Repository
from causal_agent.server.conversations.manager import Manager
from causal_agent.server.profiling import service as profiling
from causal_agent.server.settings import Settings
from causal_agent.server.tests.ingestion_resources import cluster, database_and_bucket

ROOT = Path(__file__).resolve().parents[2]


def verify_trace(trace_id, out, required_node="chat:interpret"):
    from langsmith import Client

    from causal_agent.evals.trace import runs_of, tree

    trace.flush()
    client = Client()
    runs = []
    for _ in range(10):
        runs = runs_of(client, trace_id)
        if runs and all(r.end_time for r in runs):
            break
        time.sleep(2)
    trace.write(out / "trace.json", json.dumps([r.model_dump(mode="json") for r in runs], default=str))
    trace.write(out / "trace.txt", tree(runs, thoughts=True))
    ids = {str(r.id) for r in runs}
    if not runs or str(trace_id) not in ids or any(not r.end_time for r in runs):
        raise ValueError("The acceptance trace is incomplete.")
    if any(r.parent_run_id and str(r.parent_run_id) not in ids for r in runs):
        raise ValueError("The acceptance trace has a broken parent chain.")
    if not any(r.name == required_node for r in runs):
        raise ValueError("The trace is missing the required node: " + required_node)
    return {"runs": len(runs), "errors": [{"name": r.name, "error": r.error} for r in runs if r.error]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preparation-trace", type=Path, help="Reuse a validated real Phase 4 extraction for these exact dataset bytes")
    parser.add_argument(
        "--context-only", action="store_true", help="Verify extraction, review, discussion and corrections without unrelated visualization or lane work"
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not trace.on():
        raise ValueError("Real acceptance requires tracing. No model was called.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/context/gov_transfers.md"
    csv = ROOT / "data/raw/gov-transfers-uruguay/gov_transfers.csv"
    record = {"source": str(source), "csv": str(csv), "turns": [], "status": "running"}
    trace_id = None
    try:
        with trace.span("acceptance:reviewed-chat", inputs={"source": str(source), "csv": str(csv)}) as span:
            trace_id = str(span.run.id) if span.run else None
            with tempfile.TemporaryDirectory(prefix="chat-acceptance-") as scratch, MonkeyPatch.context() as patch, cluster() as resources:
                patch.setenv("RUN_DIR", str(Path(scratch) / "runs"))
                patch.setenv("PROFILE_CACHE_DIR", str(Path(scratch) / "profiles"))
                with database_and_bucket(resources, patch) as storage:
                    owner = str(uuid4())
                    data = storage.ingest(owner, str(uuid4()), "Uruguay government transfers", source.read_text(), csv.name, io.BytesIO(csv.read_bytes()))
                    record["dataset"] = {k: data[k] for k in ("dataset_id", "version_id", "sha256")}
                    profiling.generate(storage, owner, data["dataset_id"])
                    if args.preparation_trace:
                        from causal_agent.evals.chat_preparation import reuse

                        record["retained_preparation"] = reuse(storage, owner, data, args.preparation_trace.resolve())
                    else:
                        context_service.generate(storage, owner, data["dataset_id"])
                    context = Context(storage, owner, data["dataset_id"])
                    extracted_revision = context_service.view(storage, owner, data["dataset_id"]).pack.revision_id
                    review = context.start_review("Did receiving the transfer raise support for the government?")
                    if review["starting_revision_id"] != extracted_revision:
                        raise ValueError("Review forked a second context instead of selecting the extracted revision")
                    record["extracted_revision_id"] = extracted_revision
                    # An explicit unresolved disposition never claims scientific knowledge.
                    context.save_review(review["id"], 1, [], [ColumnDisposition(column_id=c.column_id, status="unresolved") for c in review["columns"]])
                    manager = Manager(Settings(root=Path(scratch)))
                    try:
                        conversation = service.create(storage, owner, data["dataset_id"], manager, review["id"], 2)["id"]
                        repo = Repository(storage, owner, conversation)
                        column = next(c for c in review["columns"] if c.name == "Support")

                        def send(kind="message", **kwargs):
                            command = ChatCommand(submission_id=str(uuid4()), revision_id=repo.state()["revision_id"], kind=kind, **kwargs)
                            before = repo.state()["revision_id"]
                            manager.send(conversation, command)
                            manager.futures[conversation].result(timeout=900)
                            view = manager.view(conversation)
                            record["turns"].append({"command": command.model_dump(), "before_revision": before, "overview": view.model_dump(mode="json")})
                            trace.write(out / "conversation.json", json.dumps(record, indent=2))
                            if view.status != "idle":
                                raise ValueError(view.error or "Turn did not complete")
                            if kind == "message" and "I could not ground an answer" in view.prompt:
                                raise ValueError("The conversation failed its evidence gate; this is not successful semantic acceptance.")
                            return view

                        send("focus", references=[Reference(kind="column", id=column.column_id)])
                        initial = repo.state()["revision_id"]
                        send(text="What does this column measure, and when was it recorded?")
                        if repo.state()["revision_id"] != initial:
                            raise ValueError("Ordinary discussion changed context")
                        send(
                            text="Suppose support had been measured before the programme. Explain how that would change the question, without changing the stored context."
                        )
                        if repo.state()["revision_id"] != initial:
                            raise ValueError("Hypothetical discussion changed context")
                        send(
                            text="Correct the stored description of Support: it is the respondent's stated support for the government in the 2007 survey, scored 0, 0.5, or 1, measured after the programme."
                        )
                        if repo.state()["revision_id"] == initial:
                            raise ValueError("The explicit correction was not committed")
                        current = context_service.view(storage, owner, data["dataset_id"]).pack
                        if current.revision_id != repo.state()["revision_id"]:
                            raise ValueError("Dataset browsing and chat disagree about the corrected context")
                        current_column = context_service.column(storage, owner, data["dataset_id"], current.revision_id, column.column_id)
                        if not any(s.source.kind == "message" and "2007" in s.value for s in current_column.statements):
                            raise ValueError("The conversational correction is missing from the dataset's column context")
                        record["corrected_revision_id"] = current.revision_id
                        products = []
                        if not args.context_only:
                            send(
                                text="Draw a bar chart showing the number of households at each recorded Support value. Describe it without making a causal claim."
                            )
                            from causal_agent.server.conversations.publication import Repository as Products

                            products = Products(storage, owner, conversation).list(conversation)
                            if not products or not any(p.figure and p.sandbox == "docker" for p in products):
                                raise ValueError("No verified Docker visualization was published")
                            send("focus", references=[Reference(kind="artifact", id=products[0].id)])
                            send(text="Explain the counts in this artifact and its limitations.")
                            ready = send("fast_track")
                            unknown_target = None
                            if ready.questions:
                                unknown_target = ready.questions[0].target
                                send("answer", question_id=ready.questions[0].id, option_id="unknown")
                            send("pause")
                            pinned = repo.state()["revision_id"]
                            manager.pool.shutdown(wait=True)
                            manager.connection.close()
                            manager = Manager(Settings(root=Path(scratch)))
                            manager.bind(conversation, storage, owner)
                            if manager.view(conversation).revision_id != pinned:
                                raise ValueError("Restart changed the selected revision")
                            continued = send("fast_track")
                            if unknown_target and any(q.target == unknown_target for q in continued.questions):
                                raise ValueError("Fast-track repeated an answered unknown")
                            if continued.analysis_review:
                                raise ValueError("The deliberately unresolved review unexpectedly became ready")
                            send("pause")
                            if Products(storage, owner, conversation).read_file(products[0].id, "facts.json") is None:
                                raise ValueError("Artifact evidence did not survive restart")
                        record["messages"] = repo.messages(limit=100)
                        record["artifacts"] = [p.model_dump(mode="json") for p in products]
                        record["status"] = "context_passed" if args.context_only else "conversation_passed"
                    finally:
                        manager.pool.shutdown(wait=True)
                        manager.connection.close()
            span.end(outputs={"status": record["status"], "turns": len(record["turns"])})
        record["trace_id"] = trace_id
        record["trace_checks"] = verify_trace(trace_id, out)
        record["semantic_review"] = "required"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", trace_id=trace_id)
        raise
    finally:
        if trace_id and not (out / "trace.json").exists():
            try:
                record["trace_checks"] = verify_trace(trace_id, out)
            except Exception as exc:
                record["trace_readback_error"] = str(exc)
        trace.write(out / "conversation.json", json.dumps(record, indent=2, default=str))
        print(f"Chat acceptance evidence: {out / 'conversation.json'}", flush=True)


if __name__ == "__main__":
    main()
