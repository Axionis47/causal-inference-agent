"""Shared setup for conversation integration tests. No semantic acceptance claim."""

from causal_agent.common.llm import set_llm
from causal_agent.memory.context.contracts import Assessment, ColumnContext, DraftContext
from causal_agent.memory.context.review import ColumnDisposition
from causal_agent.memory.context.tests.builders import Scripted
from causal_agent.server.context import service
from causal_agent.server.context.repository import Repository
from causal_agent.server.profiling import service as profiling


def review_body(storage, owner, dataset, question):
    if service.view(storage, owner, dataset["dataset_id"]).status != "prepared":
        profiling.generate(storage, owner, dataset["dataset_id"])
        with storage.engine.connect() as db:
            columns = service.column_index(db, owner, dataset["version_id"])
        set_llm(Scripted([DraftContext(columns=[ColumnContext(column_id=c.column_id) for c in columns]), Assessment()]))
        service.generate(storage, owner, dataset["dataset_id"])
    context = Repository(storage, owner, dataset["dataset_id"])
    review = context.start_review(question)
    context.save_review(review["id"], 1, [], [ColumnDisposition(column_id=c.column_id, status="unresolved") for c in review["columns"]])
    return {"review_id": review["id"], "generation": 2}
