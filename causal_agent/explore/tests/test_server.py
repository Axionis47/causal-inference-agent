"""The agent as an A2A server, driven in process: the card at its well-known path, one task round trip with a scripted model
whose artifacts carry the record and its products, and a refused exploration as a failed task that says why."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import proto_helpers as H
from a2a.types import Role, SendMessageRequest, TaskState

from causal_agent.common.llm import set_llm
from causal_agent.explore import server as S
from causal_agent.explore.contracts import Exploration, ExploreRequest
from causal_agent.explore.tests.fakes import FakeLLM
from causal_agent.explore.tests.test_agent import ANSWER, TABLE, request

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
URL = "http://explore"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    yield
    set_llm(None)


def _request(tmp_path):
    return request(tmp_path)


async def _round_trip(req: ExploreRequest):
    app = S.app(URL + "/")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=URL, timeout=None) as http:
        card = await A2ACardResolver(http, URL).get_agent_card()
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)
        msg = H.new_data_message(req.model_dump(mode="json"), role=Role.ROLE_USER)
        task = None
        async for resp in client.send_message(SendMessageRequest(message=msg)):
            if resp.HasField("task"):
                task = resp.task
        return card, task


def test_the_card_names_the_agent_and_its_one_skill(tmp_path):
    async def go():
        app = S.app(URL + "/")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=URL) as http:
            r = await http.get("/.well-known/agent-card.json")
            return r.status_code, r.json()

    status, card = asyncio.run(go())
    assert status == 200 and card["name"] == S.NAME and [s["id"] for s in card["skills"]] == ["explore"]
    assert card["supportedInterfaces"][0]["url"] == URL + "/" and card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"


def test_one_task_round_trip_carries_the_record_and_its_products(tmp_path):
    set_llm(
        FakeLLM(
            [[TABLE], []],
            [ANSWER],
        )
    )
    _card, task = asyncio.run(_round_trip(_request(tmp_path)))
    assert task is not None and task.status.state == TaskState.TASK_STATE_COMPLETED
    names = [a.name for a in task.artifacts]
    assert names == ["exploration", "facts", "summary", "table", "code", "thoughts", "trail"]
    found = {a.name: a for a in task.artifacts}
    assert H.get_data_parts(found["facts"].parts)[0] == {"reading_standard": 71.65}  # only the facts stood behind
    assert H.get_text_parts(found["summary"].parts)[0].startswith("Standard lunch") and b"lunch" in H.get_raw_parts(found["table"].parts)[0]
    assert "pd.read_csv" in H.get_text_parts(found["code"].parts)[0] and H.get_data_parts(found["trail"].parts)[0]["calls"] == 1


def test_a_refused_exploration_is_a_failed_task_that_says_why(tmp_path):
    bad = Exploration(summary="x", step=1, facts=["reading_standard"])
    set_llm(FakeLLM([[], [], []], [bad, bad, bad]))
    _card, task = asyncio.run(_round_trip(_request(tmp_path)))
    assert task is not None and task.status.state == TaskState.TASK_STATE_FAILED
    assert "no script has run yet" in H.get_message_text(task.status.message) and [a.name for a in task.artifacts] == ["thoughts", "trail"]
