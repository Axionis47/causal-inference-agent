"""Context reuses the isolated PostgreSQL/S3 harness and shared semantic input builders."""

import io
from uuid import uuid4

import pytest

from causal_agent.common.llm import set_llm
from causal_agent.memory.context.tests.builders import SOURCE, Scripted, happy
from causal_agent.server.context import service
from causal_agent.server.profiling import service as profiling


@pytest.fixture(autouse=True)
def model_guard():
    set_llm(Scripted([]))
    yield
    set_llm(None)


@pytest.fixture
def prepared(storage):
    def create(owner, source=SOURCE):
        record = storage.ingest(owner, str(uuid4()), "Context study", source, "study.csv", io.BytesIO(b"code,amount\r\n001,10\r\n002,20\r\n"))
        profiling.generate(storage, owner, record["dataset_id"])
        with storage.engine.connect() as db:
            columns = service.column_index(db, owner, record["version_id"])
        set_llm(happy(columns, source) if source.strip() else Scripted([]))
        return record

    return create
