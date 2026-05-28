"""Four native agent loops through PostgreSQL, approval, execution and reopening."""

import json
from uuid import UUID, uuid4

import pytest
from analysis_context_fixture import create, provider_proposal, ready_revision
from test_analysis import approval, legacy_scripted, prepare, read

from backend import model_access
from backend.analysis import jobs
from backend.analysis.contracts import CausalAccount, Plan, Program, Target, Treatments
from backend.config import settings
from backend.execution import runner as sandbox


def tool_turn(name, args):
    content = {"role": "model", "parts": [{"function_call": {"name": name, "args": args}}]}
    return model_access.ToolTurn(content, json.dumps(content), {"input_tokens": 100, "output_tokens": 50})


def selected_call(turn):
    return turn.content["parts"][0]["function_call"]


def agent_detail(owner, saved, agent):
    identity = next(item["id"] for item in saved["agents"] if item["agent"] == agent)
    response = owner.get(f"/api/analyses/{saved['id']}/agents/{identity}")
    assert response.status_code == 200
    return response.json()


def native_script(request, factory=legacy_scripted):
    contents = json.loads(request.contents_json)
    packet = json.loads(contents[0]["parts"][0]["text"])
    names = [part["function_response"]["name"] for message in contents for part in message.get("parts", [])
             if part.get("function_response", {}).get("response", {}).get("ok")
             and part["function_response"]["name"] not in {"read_context", "inspect_work", "delegate_evidence"}]
    available = {item["name"] for item in json.loads(request.functions_json)}
    if "submit_findings" in available:
        name, args = "submit_findings", {"summary": "Scoped frozen evidence inspected.", "references": [item["key"] for item in packet["evidence_records"]]}
    elif "submit_review" in available:
        name, args = "submit_review", {"accepted": True, "summary": "Controlled independent review.", "issues": []}
    elif "set_target" in available:
        target = provider_proposal(Target, packet, factory(Target, "", packet, "test"))
        treatment = provider_proposal(Treatments, packet, factory(Treatments, "", packet, "test"))
        steps = [
            ("set_target", {key: target[key] for key in ("outcome", "measurement", "population", "unit", "timing", "requested_effects", "summary")}),
            *[("set_factor", {"treatment": item, "values": [v for v in treatment["values"] if v["variable_id"] == item["variable"]["id"]]}) for item in treatment["treatments"]],
            *[("set_assignment", {"process": item}) for item in treatment["assignments"]],
            *[("set_comparison", {"effect_id": item["id"], **{key: item[key] for key in ("treatment_ids", "estimand", "comparison")}}) for item in treatment["effects"]],
            ("publish_handoff", {"summary": "Study bound to supplied evidence."}),
        ]
        name, args = steps[len(names)]
    elif "set_structure" in available:
        study = packet["accepted_handoffs"][0]["records"]
        payload = {**packet, "decisions": study}
        causal = provider_proposal(CausalAccount, payload, factory(CausalAccount, "", payload, "test"))
        steps = [
            ("set_structure", {key: causal[key] for key in ("nodes", "edges", "identification", "threats")}),
            *[("set_identification", {"identification": item}) for item in causal["identifications"]],
            ("freeze_causal_account", {}),
            ("run_diagnostic", {"action": "run_code", "question": "Count available treatment records.", "consequence": "Check observations exist.", "code": "import json\njson.dump({'rows': 3}, open('facts.json', 'w'))"}),
            ("set_method", {"plan": factory(Plan, "", packet, "test").model_dump(mode="json")}),
            ("publish_handoff", {"summary": "Frozen executable design."}),
        ]
        name, args = steps[len(names)]
    elif "submit_program" in available:
        replies = [part["function_response"]["response"] for message in contents for part in message.get("parts", [])
                   if part.get("function_response", {}).get("name") == "inspect_computation"]
        if not names or names[-1] == "submit_program":
            name, args = "inspect_computation", {}
        elif not replies[-1]["current_program"]:
            name, args = "submit_program", factory(Program, "", packet, "test").model_dump(mode="json")
        elif replies[-1].get("review_feedback"):
            name, args = "submit_program", {"code": replies[-1]["current_program"]["program"]["code"] + "\n# Corrected after human review\n", "summary": "Review correction"}
        elif replies[-1]["programming_failure"] and replies[-1]["repairs_remaining"]:
            name, args = "submit_program", {"code": replies[-1]["current_program"]["program"]["code"] + "\n# Minimal programming correction\n", "summary": "Programming correction only"}
        elif not replies[-1]["execution"]:
            name, args = "stop_analysis", {"outcome": "unidentified", "reason": "The approved computation did not produce the required valid results."}
        else:
            name, args = "publish_handoff", {"summary": "Approved computation validated."}
    else:
        steps = [("inspect_results", {}), ("set_report", {"report": {"text": "The estimate is 2, conditional on the frozen assumptions.",
                    "numbers": [{"source": "execution.primary_estimate", "value": 2}]}, "citations": ["execution"]}),
                 ("publish_handoff", {"summary": "Grounded explanation saved."})]
        name, args = steps[len(names)]
    content = {"role": "model", "parts": [{"function_call": {"name": name, "args": args, "id": str(len(names))}}]}
    return model_access.ToolTurn(content, json.dumps(content), {"input_tokens": 100, "output_tokens": 50})


@pytest.fixture
def native_boundary(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(settings(), "storage_dir", tmp_path)
    monkeypatch.setattr(model_access, "invoke_tools", native_script)
    monkeypatch.setattr(sandbox, "ready", lambda: {"runtime_hash": "runtime-test"})
    def execute(identity, code, csv, inputs, runtime):
        calls.append({"id": str(identity), "csv": csv, "inputs": inputs})
        return {"ok": True, "stdout": "saved", "stderr": "", "files": {},
                "facts": {"primary_estimate": 2.0, "primary_ci_low": 1.0, "primary_ci_high": 3.0, "primary_n": 3.0} if inputs else {"rows": 3.0}}
    monkeypatch.setattr(sandbox, "run", execute)
    return calls


@pytest.mark.parametrize("extra_columns", [0, 20, 198])
def test_native_agents_complete_once_and_reopen_with_constant_call_budget(owner, native_boundary, extra_columns):
    if extra_columns:
        names = ["group", *[f"opaque_{i}" for i in range(extra_columns)], "score"]
        raw = (",".join(names) + "\n" + "\n".join(
            ",".join([group, *[str(i) for i in range(extra_columns)], score]) for group, score in [("a", "2"), ("a", "4"), ("b", "8")]) + "\n").encode()
        data = owner.post("/api/datasets", data={"title": "Wide native fixture", "submission_id": str(uuid4())}, files={"file": ("wide.csv", raw, "text/csv")}).json()
        revision = ready_revision(owner, data["version"]["id"])
        chat = owner.post(f"/api/versions/{data['version']['id']}/conversations", json={"title": "Native agents", "submission_id": str(uuid4()), "context_revision_id": revision["id"]}).json()
    else:
        _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "awaiting_approval", saved["error"]
    assert saved["checkpoints"] == []
    assert [agent["status"] for agent in saved["agents"]] == ["accepted", "accepted", "awaiting_approval", "pending"]
    assert len(saved["frame"]["roles"]["assignments"]) == extra_columns
    assert len(native_boundary) == 1 and not native_boundary[0]["inputs"]
    assert len(saved["snapshot"]["input_projection"]["columns"]) == 2
    complete = approval(owner, run["id"])
    assert complete["status"] == "succeeded", complete["error"]
    assert complete["result"]["facts"]["primary_estimate"] == 2
    assert complete["calls"] == 22
    assert all(agent["status"] == "accepted" for agent in complete["agents"])
    assert len(native_boundary) == 2
    jobs.process(UUID(run["id"]))
    assert read(owner, run["id"]) == complete
    assert len(native_boundary) == 2
    for agent in complete["agents"]:
        detail = owner.get(f"/api/analyses/{run['id']}/agents/{agent['id']}")
        assert detail.status_code == 200
        assert detail.json()["handoff_hash"] == agent["handoff_hash"]
