"""Isolated native tool loops. Only accepted domain handoffs leave an agent."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict, cast
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ValidationError

from backend import model_access
from backend.analysis import authority, packets, provider
from backend.analysis.agent_store import LIMITS, AgentLimit, Store
from backend.analysis.records import programs, requested_revision
from backend.models import AnalysisTask, AnalysisTaskAttempt

AGENT_INPUT_LIMIT = 192_000  # Complete multi-turn transport, including native thought signatures.


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    execute: Callable[[Any, str], dict[str, Any]]
    claims: bool = False

    def declaration(self) -> dict[str, Any]:
        choices = {"_json_schema": ["native"]}
        if self.claims:
            choices["_authority_policy"] = [authority.POLICY]
            choices["_proposal_defaults"] = ["empty"]
        schema = provider.provider_schema(self.arguments, choices)
        return {"name": self.name, "description": self.description, "parameters_json_schema": schema}

    def invoke(self, args: dict[str, Any], key: str) -> dict[str, Any]:
        value = authority.proposal_decoder(allow_empty_defaults=True)(args) if self.claims else args
        return self.execute(self.arguments.model_validate(value), key)


class State(TypedDict, total=False):
    attempt_id: str
    done: bool
    paused: bool


def calls(content: dict[str, Any]) -> list[dict[str, Any]]:
    return [part["function_call"] for part in content.get("parts", []) if part.get("function_call")]


class Agent:
    def __init__(self, store: Store, task: AnalysisTask, tools: list[Tool], system: str, model: str):
        self.store, self.task, self.system, self.model = store, task, system, model
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Agent tool names must be unique.")

    def contents(self, turns: list[AnalysisTaskAttempt]) -> list[dict[str, Any]]:
        packet: dict[str, Any] = dict(self.task.packet)
        packet["accepted_handoffs"] = [self.store.handoff(UUID(item["producer"]), expected_hash=item["hash"])
                                        for item in cast(list[dict[str, Any]], packet.get("handoffs", []))]
        for handoff in packet["accepted_handoffs"]:
            if "snapshot" in handoff["records"]:
                handoff["records"]["snapshot"] = packets.science(handoff["records"]["snapshot"])
            if "column_inventory" in handoff["records"]:
                handoff["records"]["column_inventory_count"] = len(handoff["records"].pop("column_inventory"))
                handoff["records"]["column_inventory_reference"] = handoff["id"]
        result: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": json.dumps(packets.model_view(packet), ensure_ascii=False)}]}]
        for turn in turns:
            if turn.provider_content is None:
                continue
            result.append(dict(turn.provider_content))
            selected = calls(dict(turn.provider_content))
            responses: list[dict[str, Any]] = []
            for index, call in enumerate(selected):
                receipt: Any = turn.tool_receipts.get(str(index))
                if not receipt or not receipt.get("complete"):
                    raise ValueError("A pending tool must settle before another model turn.")
                responses.append({"function_response": {"name": call["name"],
                    **({"id": call["id"]} if call.get("id") else {}), "response": receipt["result"]}})
            result.append({"role": "user", "parts": responses or [{"text": "Use the available tools to inspect, revise, or publish your work. A prose response does not publish a handoff."}]})
        run = self.store.locked_run()
        saved = programs(run)
        feedback = run.receipts.get(f"revision_request:{saved[-1]['revision']}") if saved and requested_revision(run) else None
        result.append({"role": "user", "parts": [{"text": json.dumps({"run_status": run.status,
            "review_feedback": feedback if self.task.agent == "computation" else None,
            "remaining_model_calls":
            max(0, min(int(self.task.model_limit or 0) - len(turns), LIMITS[str(self.task.agent)] - self.store.counts(str(self.task.agent)))),
            "handoff_submissions_remaining": max(0, 3 - self.task.submissions)})}]})
        return result

    def model_turn(self, state: State) -> State:
        self.store.db.refresh(self.task)
        run = self.store.locked_run()
        if self.task.status in {"accepted", "blocked", "exhausted"} or run.status == "blocked":
            self.store.db.commit()
            return {"done": True, "paused": False}
        if run.status == "awaiting_approval":
            self.store.db.commit()
            return {"done": False, "paused": True}
        turns = self.store.turns(self.task)
        for turn in turns:
            if turn.provider_content is not None:
                expected = range(max(1, len(calls(dict(turn.provider_content)))))
                if any(not cast(dict[str, Any], turn.tool_receipts.get(str(index), {})).get("complete") for index in expected):
                    self.store.db.commit()
                    return {"attempt_id": str(turn.id), "done": False, "paused": False}
        request = model_access.PreparedToolRequest(model=self.model, system=self.system,
            contents_json=json.dumps(self.contents(turns), ensure_ascii=False),
            functions_json=json.dumps([tool.declaration() for tool in self.tools.values()], ensure_ascii=False),
            output_limit=16384 if self.task.agent == "computation" else 8192)
        if len(request.system) + len(request.contents_json) + len(request.functions_json) > AGENT_INPUT_LIMIT:
            raise packets.PacketOverflow("This agent's complete request exceeds its input limit. No evidence was truncated.")
        attempt = self.store.reserve(self.task, request)
        try:
            response = model_access.invoke_tools(request)
        except Exception as error:
            self.store.settle(attempt, None, error)
            raise
        self.store.settle(attempt, response)
        return {"attempt_id": str(attempt.id), "done": False, "paused": False}

    def execute_tools(self, state: State) -> State:
        attempt = self.store.db.get(AnalysisTaskAttempt, UUID(state["attempt_id"]))
        if attempt is None or attempt.task_id != self.task.id or attempt.provider_content is None:
            raise ValueError("The agent tool step must reference its saved provider response.")
        selected = calls(dict(attempt.provider_content))
        for index, call in enumerate(selected or [{"name": "__missing_tool", "args": {}}]):
            name, args = str(call.get("name", "")), call.get("args") or {}

            def execute(key: str, name: str = name, args: Any = args) -> dict[str, Any]:
                tool = self.tools.get(name)
                if tool is None:
                    return {"ok": False, "error": "Select one of the available tools.", "available": list(self.tools)}
                self.store.db.refresh(self.task)
                if self.task.status == "accepted":
                    return {"ok": False, "error": "The handoff is already sealed. Further edits are not permitted."}
                try:
                    result = tool.invoke(args, key)
                    if "handoff" in result:
                        accepted = result["handoff"]
                        result["handoff"] = {name: accepted[name] for name in ("id", "producer", "hash", "agent", "summary")}
                    return {"ok": True, **result}
                except AgentLimit:
                    raise
                except ValidationError as error:
                    return {"ok": False, "error": "Correct the tool arguments.",
                            "issues": error.errors(include_url=False, include_input=False)}
                except (ValueError, SyntaxError) as error:
                    return {"ok": False, "error": str(error)[:6000], "action": "Revise the affected record using the frozen evidence, then validate again."}

            self.store.tool(self.task, attempt, index, name, args, execute)
        self.store.db.refresh(self.task)
        run = self.store.locked_run()
        done = self.task.status in {"accepted", "blocked", "exhausted"} or run.status == "blocked"
        paused = run.status == "awaiting_approval"
        self.store.db.commit()
        return {"done": done, "paused": paused}

    def graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        graph = StateGraph(State)
        graph.add_node("model", self.model_turn)
        graph.add_node("tools", self.execute_tools)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", lambda state: END if state.get("done") or state.get("paused") else "tools", [END, "tools"])
        graph.add_conditional_edges("tools", lambda state: END if state.get("done") or state.get("paused") else "model", [END, "model"])
        # The parent provides PostgreSQL persistence; each named agent gets a private namespace.
        return graph.compile()
