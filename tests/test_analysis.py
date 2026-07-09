"""Real PostgreSQL/LangGraph journeys with controlled model and numerical boundaries."""

import csv
import io
import json
from copy import deepcopy
from uuid import UUID, uuid4

import pytest
import test_chat
from analysis_context_fixture import create, provider_proposal
from conftest import PASSWORD
from historical_analysis.work import ColumnScreen
from model_double import model_double
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from backend import model_access
from backend.analysis import delivery, domain, jobs
from backend.analysis.contracts import (
    AnalysisReport,
    CausalAccount,
    Claim,
    CodeReview,
    Conclusion,
    ContrastBinding,
    DiagnosticAction,
    Plan,
    Program,
    Roles,
    Target,
    Treatments,
    TreatmentValue,
)
from backend.chat import jobs as chat_jobs
from backend.chat.contracts import Interpretation
from backend.config import settings
from backend.context.contracts import ContextEdit, EditStatement, empty_content
from backend.db import engine
from backend.execution import runner as sandbox
from backend.integration.readers import Analyses
from backend.models import AnalysisRun, ChatContextSelection, ChatTurn
from backend.responses import Reply
from backend.validation import ReplyNumber

context_chat = test_chat.context_chat


@pytest.mark.parametrize("extra_columns", [10, 58, 198])
def test_column_count_scopes_reasoning_and_accounts_for_every_column(
    owner, boundaries, monkeypatch, extra_columns
):
    from backend.analysis import packets

    observed = []

    def infer(schema, system, payload, model, **kwargs):
        observed.append((schema, deepcopy(payload)))
        return scripted(schema, system, payload, model, **kwargs)

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    names = ["group", *[f"unrelated_{i}" for i in range(extra_columns)], "score"]
    raw = (
        ",".join(names)
        + "\n"
        + "\n".join(
            ",".join([group, *[str(i) for i in range(extra_columns)], score])
            for group, score in [("a", "2"), ("a", "4"), ("b", "8")]
        )
        + "\n"
    ).encode()
    data = owner.post(
        "/api/datasets",
        data={"title": "Wide fixture", "submission_id": str(uuid4())},
        files={"file": ("wide.csv", raw, "text/csv")},
    ).json()
    from analysis_context_fixture import ready_revision
    revision = ready_revision(owner, data["version"]["id"])
    response = owner.post(
        f"/api/versions/{data['version']['id']}/conversations",
        json={"title": "Column-count fixture", "submission_id": str(uuid4()), "context_revision_id": revision["id"]},
    )
    run, _ = prepare(owner, response.json())
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "awaiting_approval", json.dumps({"error": saved["error"], "failed": [(c["id"], a["status"], [g for g in a["checks"] if not g["passed"]]) for c in saved["checkpoints"] for a in c["attempts"] if a["status"] != "accepted"], "tasks": [(t["key"], t["status"], [g for a in t["attempts"] for g in a["checks"] if not g["passed"]]) for t in saved["tasks"] if t["status"] != "accepted"][:3]})
    assert len(saved["frame"]["roles"]["assignments"]) == extra_columns
    assert saved["checkpoints"] == []
    assert saved["calls"] == 16
    assert all(task["kind"] == "agent" for task in saved["tasks"])
    for schema, payload in observed:
        if schema in {Target, Treatments, CausalAccount, Plan}:
            assert len(payload["column_identities"]) == 2
        if schema is Program:
            assert len(payload["input_columns"]) == 2
            assert payload["snapshot"]["frame"]["roles"]["assignments"] == []
        if "evidence_scope" in payload:
            assert payload["evidence_scope"]["evidence_characters"] <= packets.EVIDENCE_LIMIT
    assert len(next(csv.reader(io.StringIO(boundaries[0]["csv"].decode())))) == 1
    before = deepcopy(saved)
    jobs.process(UUID(run["id"]))
    assert read(owner, run["id"]) == before
    completed = approval(owner, run["id"])
    assert completed["status"] == "succeeded" and completed["budgets"]["shared_calls"] <= 27


def test_receipt_input_is_detached_from_mutable_retrieval_history(owner, boundaries):
    from backend.analysis.agent_store import Store
    from backend.context.service import fingerprint
    saved = prepared(owner)
    with Session(engine()) as db:
        store = Store(db, UUID(saved["id"]))
        packet = {"frozen_inputs": {}, "handoffs": [], "history": [{"requested_keys": ["dataset"]}]}
        expected = fingerprint(packet)
        task = store.open("study", packet, key="mutation-regression")
        packet["history"][0]["requested_keys"].append("relationships")
        db.refresh(task)
        assert task.input_hash == fingerprint(task.packet) == expected
        assert task.packet["history"] == [{"requested_keys": ["dataset"]}]


def test_oversized_model_packet_blocks_without_spending_call(owner, boundaries, monkeypatch):
    from backend.analysis import agent_runtime
    monkeypatch.setattr(agent_runtime, "AGENT_INPUT_LIMIT", 1)
    _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "blocked" and saved["calls"] == 0
    assert "input limit" in saved["error"] and not boundaries


def test_report_contract_rejects_chat_actions_and_forged_references():
    from pydantic import ValidationError

    from backend.analysis.reporting import contract

    schema = contract({"execution", "question"}, {"execution.effect_estimate", "question"})
    valid = {**decision(), "citations": ["execution"], "report": {"text": "Estimate 2.", "numbers": [{"value": 2, "source": "execution.effect_estimate"}]}}
    assert schema.model_validate(valid).citations == ["execution"]
    for changes in [
        {"citations": ["execution.forged-id"]},
        {"report": {**valid["report"], "analysis_references": ["execution"]}},
        {"report": {**valid["report"], "numbers": [{"value": 2, "source": "execution.forged_fact"}]}},
    ]:
        with pytest.raises(ValidationError):
            schema.model_validate({**valid, **changes})


@pytest.mark.parametrize(
    "finish_before_reply,select_future_file", [(False, False), (True, False), (True, True)]
)
def test_chat_binds_frozen_program_before_approval_and_excludes_future_results(
    owner, boundaries, monkeypatch, finish_before_reply, select_future_file
):
    from backend.chat.contracts import AnalysisFile

    saved = prepared(owner)
    request = owner.post(
        f"/api/conversations/{saved['conversation_id']}/turns",
        json={
            "submission_id": str(uuid4()),
            "expected_turn": 0,
            "text": "Inspect the frozen program and design before approval. Do not execute anything.",
        },
    )
    assert request.status_code == 202
    if finish_before_reply:
        approval(owner, saved["id"])
    executions = len(boundaries)

    def discuss(schema, system, payload, model, **kwargs):
        assert payload["analyses"][0]["status_at_submission"] == "awaiting_approval"
        if schema is Interpretation:
            assert payload["analyses"][0]["result_available"] is False
            assert "snapshot" not in payload["analyses"][0] and "result" not in payload["analyses"][0]
            assert "program-0.py" in payload["analyses"][0]["artifact_files"]
            assert "execution-0-facts.json" not in payload["analyses"][0]["artifact_files"]
            return Interpretation(
                analysis_files=[
                    AnalysisFile(
                        analysis_id=UUID(saved["id"]),
                        filename="execution-0-facts.json" if select_future_file else "program-0.py",
                    )
                ]
            )
        assert payload["analyses"][0]["result"] is None
        assert payload["analysis_file_evidence"][0]["content"] == saved["programs"][0]["program"]["code"]
        assert not any(v["basis"] == "computed" for v in payload["numeric_evidence"].values())
        return Reply(text="This is the frozen program; no result was available when this message was sent.")

    monkeypatch.setattr(model_access, "structured", model_double(discuss))
    chat_jobs.process(UUID(request.json()["id"]), Analyses())
    turn = owner.get(f"/api/conversations/{saved['conversation_id']}").json()["turns"][-1]
    assert turn["status"] == ("failed" if select_future_file else "succeeded"), turn
    if select_future_file:
        assert "outside the turn's bound analysis" in turn["error"]
    assert len(boundaries) == executions
    if not finish_before_reply:
        assert read(owner, saved["id"])["status"] == "awaiting_approval"


def test_packet_scoping_keeps_new_relevant_evidence_and_cross_column_dependencies():
    from backend.analysis import packets
    from backend.context.contracts import ColumnContext, SourceReference, Statement

    context = empty_content(uuid4(), ["group", "score", "confounder", "noise"])
    ids = {c.identity.name: c.identity.column_id for c in context.columns.values()}
    source = SourceReference(kind="review", id=uuid4(), author_id=uuid4())
    context.columns[ids["confounder"]].definition["confounder_fact"] = Statement(
        value="Common cause of assignment and outcome", source=source, status="reviewed"
    )
    target = scripted(
        Target, "", {"column_identities": [c.identity.model_dump() for c in context.columns.values()]}, "test"
    )
    treatments = scripted(
        Treatments,
        "",
        {"column_identities": [c.identity.model_dump() for c in context.columns.values()]},
        "test",
    )
    role = {
        "variable": variable(context.columns[ids["confounder"]].identity.model_dump()),
        "roles": ["confounder"],
    }
    prior = {
        "target": target.model_dump(),
        "treatments": treatments.model_dump(),
        "roles": {"assignments": [role]},
    }
    active = packets.active(prior)
    assert ids["confounder"] in active and ids["noise"] not in active
    evidence = packets.evidence_records(packets.scope(context, active))
    assert any(item["statement"]["value"] == "Common cause of assignment and outcome" for item in evidence)
    assert packets.decisions(prior)["roles"]["assignments"] == [role]
    # A large single-column account must block, never truncate its qualifications.
    col: ColumnContext = context.columns[ids["confounder"]]
    for i in range(5):
        col.qualifications[f"qualification_{i}"] = Statement(value="c" * 9000, source=source)
    with pytest.raises(packets.PacketOverflow, match="No evidence was truncated"):
        packets.bounded(packets.scope(context, {ids["confounder"]}))


def test_only_validated_execution_artifacts_supply_computed_chat_numbers():
    from backend.analysis.artifacts import numerical_facts

    run = AnalysisRun(
        receipts={
            "execution:0": {
                "valid": True,
                "facts": {"primary_estimate": 2.5},
                "files": {"facts.json": {"key": "saved/facts.json", "sha256": "bound-checksum"}},
            }
        }
    )
    assert numerical_facts(run, "execution-0-facts.json") == {"primary_estimate": 2.5}
    assert numerical_facts(run, "execution-1-facts.json") == {}
    assert numerical_facts(run, "snapshot.json") == {}
    run.receipts["execution:0"]["valid"] = False
    assert numerical_facts(run, "execution-0-facts.json") == {}


@pytest.fixture(autouse=True)
def storage(isolated, tmp_path, monkeypatch):
    monkeypatch.setattr(settings(), "storage_dir", tmp_path)


def claim(text="Supplied study account"):
    return {"statement": text, "basis": "supplied", "references": ["question"]}


def decision(summary="The supplied account establishes this decision."):
    return {
        "status": "ready",
        "summary": summary,
        "citations": ["question"],
        "changes": "The scientific question is now explicit.",
        "permits": "The next checkpoint can use this decision.",
    }


def variable(column):
    return {
        "id": column["column_id"],
        "column_ids": [column["column_id"]],
        "label": column["name"],
        "meaning": claim(),
    }


def neutral_roles(payload):
    """Controlled irrelevant dispositions cover the exact coordinator-owned scopes."""
    groups = {}
    for effect, treatments in payload["role_scope"]["effect_treatments"].items():
        groups.setdefault(tuple(sorted(treatments)), []).append(effect)
    established = {cid: item for item in payload.get("established_variables", []) for cid in item["column_ids"]}
    return [
        {
            "variable": established.get(column["column_id"], variable(column)),
            "roles": ["irrelevant"],
            "timing": "unknown",
            "account": claim("No causal or operational role in this controlled fixture scope."),
            "timing_evidence": {"statement": "No timing claim is required for this excluded variable.", "basis": "unknown"},
            "reason": "Explicitly excluded in this effect/event scope.",
            "effect_ids": effects,
            "treatment_ids": list(treatments),
            "time_scope": "Timing is unknown and does not affect this excluded scope.",
        }
        for column in payload["column_identities"] if column["column_id"] in payload["task_columns"]
        for treatments, effects in groups.items()
    ]


def static_comparison(column_id, prefix="group", kind="text", levels=("a", "b")):
    values = [TreatmentValue(id=f"{prefix}_{name}", variable_id=column_id,
                             value={"kind": kind, "literal": level}, meaning=claim())
              for name, level in zip(("control", "treated"), levels, strict=True)]
    binding = ContrastBinding(kind="static", evidence=claim(), static={"terms": [
        {"weight": 1, "value_ids": [values[1].id]},
        {"weight": -1, "value_ids": [values[0].id]},
    ]})
    return values, binding


def scripted(schema, system, payload, model, **kwargs):
    result = legacy_scripted(schema, system, payload, model, **kwargs)
    if payload.get("frozen_inputs") and schema in {ColumnScreen, Target, Treatments, Roles, CausalAccount}:
        assert callable(kwargs.get("reference_decoder")), "Current fixtures must cross the real provider decoder boundary."
        return schema.model_validate(kwargs["reference_decoder"](provider_proposal(schema, payload, result)))
    return result


def legacy_scripted(schema, system, payload, model, **kwargs):
    if schema is ColumnScreen:
        return ColumnScreen(
            summary="This column has been screened.", citations=["question"],
            variable=variable(next(c for c in payload["column_identities"] if c["column_id"] in payload["task_columns"])),
            target_candidate=any(c["name"] == "score" and c["column_id"] in payload["task_columns"] for c in payload["column_identities"]),
            treatment_candidate=any(c["name"] in {"group", "second"} and c["column_id"] in payload["task_columns"] for c in payload["column_identities"]),
        )
    if schema is Target:
        column = next(c for c in payload["column_identities"] if c["name"] == "score")
        return Target(
            **decision(),
            outcome=variable(column),
            measurement=claim(),
            population=claim(),
            unit=claim(),
            timing=claim(),
            effect_question="Average effect on score",
            requested_effects=[{"id": "primary", "question": claim("Average effect on score"), "priority": "primary"}],
        )
    if schema is Treatments:
        column = next(c for c in payload["column_identities"] if c["name"] == "group")
        values, comparison = static_comparison(column["column_id"])
        return Treatments(
            **decision(),
            values=values,
            treatments=[
                {"variable": variable(column), "change": claim(), "timing": claim(), "assignment": claim()}
            ],
            assignments=[{"id": "assignment", "treatment_ids": [column["column_id"]], "kind": "point", "unit": claim(), "timing": claim(), "mechanism": claim()}],
            effects=[
                {
                    "id": "primary",
                    "treatment_ids": [column["column_id"]],
                    "contrast": claim(),
                    "estimand": "Average effect",
                    "priority": "primary",
                    "comparison": comparison,
                }
            ],
        )
    if schema is Roles:
        return Roles(**decision(), assignments=neutral_roles(payload))
    if schema is CausalAccount:
        prior = payload["decisions"]
        y = prior["target"]["outcome"]
        t = prior["treatments"]["treatments"][0]["variable"]
        return CausalAccount(
            **decision(),
            identification="Randomized contrast, conditional on no interference.",
            identifications=[{"effect_id": "primary", "strategy": "source_assignment", "evidence": claim(), "adjustment": []}],
            nodes=[
                {"id": v["id"], "label": v["label"], "column_ids": v["column_ids"], "roles": [role]}
                for v, role in ((t, "treatment"), (y, "outcome"))
            ],
            edges=[
                {
                    "source": t["id"],
                    "target": y["id"],
                    "basis": "supplied",
                    "references": ["question"],
                    "explanation": "Declared temporal intervention.",
                }
            ],
        )
    if schema is DiagnosticAction:
        return DiagnosticAction(
            action="finish" if payload["computations"] else "run_code",
            question="Count available treatment records.",
            consequence="Check that observations exist.",
            code="import json\njson.dump({'rows': 3}, open('facts.json', 'w'))",
        )
    if schema is Plan:
        return Plan(
            **decision(),
            effects=[
                {
                    "effect_id": "primary",
                    "estimator": "Difference in means",
                    "inference": "Welch interval",
                    "adjustment_reason": "Randomized assignment",
                }
            ],
            population="Study sample",
            missingness="Exclude missing scores",
            exclusions="None otherwise",
            multiplicity="One primary effect",
            reporting="Report effect and interval regardless of significance",
            limitations=["Conditional on randomization."],
        )
    if schema is Program:
        return Program(
            code="import json\njson.dump({'primary_estimate': 2, 'primary_ci_low': 1, 'primary_ci_high': 3, 'primary_n': 3}, open('facts.json', 'w'))",
            summary="Primary comparison",
        )
    if schema is CodeReview:
        return CodeReview(accepted=True, summary="Controlled test review accepted.")
    if issubclass(schema, Conclusion):
        failed = not payload["execution"]["valid"]
        return Conclusion(
            **decision("The conclusion follows the recorded execution."),
            report=AnalysisReport(
                text="The approved computation failed."
                if failed
                else "The primary estimate is 2, conditional on the frozen assumptions.",
                numbers=[] if failed else [ReplyNumber(source="execution.primary_estimate", value=2)],
            ),
        )
    if schema is Interpretation:
        return Interpretation()
    assert schema is Reply
    return Reply(text="The saved analysis remains conditional on its assumptions.")


@pytest.fixture
def boundaries(monkeypatch):
    from test_agents_journey import native_script

    calls = []
    monkeypatch.setattr(model_access, "structured", model_double(scripted))
    monkeypatch.setattr(model_access, "invoke_tools", native_script)
    monkeypatch.setattr(sandbox, "ready", lambda: {"runtime_hash": "runtime-test"})

    def run(identity, code, csv, inputs, runtime_hash):
        calls.append({"id": identity, "code": code, "csv": csv, "inputs": inputs})
        return {
            "ok": True,
            "stdout": "saved",
            "stderr": "",
            "files": {},
            "facts": {
                "primary_estimate": 2.0,
                "primary_ci_low": 1.0,
                "primary_ci_high": 3.0,
                "primary_n": 3.0,
            }
            if inputs
            else {"rows": 3.0},
        }

    monkeypatch.setattr(sandbox, "run", run)
    return calls


def prepare(owner, chat):
    readiness_response = owner.get(f"/api/conversations/{chat['id']}/analysis-readiness")
    assert readiness_response.status_code == 200, readiness_response.text
    readiness = readiness_response.json()
    assert readiness["status"] != "needs_context", readiness
    body = {
        "submission_id": str(uuid4()),
        "question": readiness["question"],
        "expected_context_revision_id": readiness["context_revision_id"],
        "expected_readiness_fingerprint": readiness["fingerprint"],
    }
    response = owner.post(f"/api/conversations/{chat['id']}/analyses", json=body)
    assert response.status_code == 202, response.text
    return response.json(), body


def read(owner, identity):
    response = owner.get(f"/api/analyses/{identity}")
    assert response.status_code == 200, response.text
    return response.json()


def approval(owner, identity):
    saved = read(owner, identity)
    result = owner.post(
        f"/api/analyses/{identity}/approve",
        json={"snapshot_hash": saved["snapshot_hash"], "program_hash": saved["programs"][-1]["hash"]},
    )
    assert result.status_code == 202, result.text
    jobs.process(UUID(identity))
    return read(owner, identity)


def prepared(owner):
    _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "awaiting_approval", json.dumps({"error": saved["error"], "failed": [(c["id"], a["status"], [g for g in a["checks"] if not g["passed"]]) for c in saved["checkpoints"] for a in c["attempts"] if a["status"] != "accepted"], "tasks": [(t["key"], t["status"]) for t in saved["tasks"] if t["status"] != "accepted"]})
    return saved


def test_approval_checkpoint_semantics_execution_and_reopen(owner, boundaries):
    saved = prepared(owner)
    assert saved["checkpoints"] == []
    assert [agent["status"] for agent in saved["agents"]] == ["accepted", "accepted", "awaiting_approval", "pending"]
    assert len(boundaries) == 1 and not boundaries[0]["inputs"]
    diagnostic_rows = list(csv.reader(io.StringIO(boundaries[0]["csv"].decode())))
    outcome = saved["frame"]["target"]["outcome"]["column_ids"][0]
    assert outcome not in diagnostic_rows[0] and all(len(row) == 1 for row in diagnostic_rows)
    assert all(agent["handoff_hash"] for agent in saved["agents"][:2])
    result = approval(owner, saved["id"])
    assert result["status"] == "succeeded", result
    assert len(boundaries) == 2 and "snapshot.json" in boundaries[1]["inputs"]
    assert result["snapshot"] == saved["snapshot"]
    jobs.process(UUID(saved["id"]))
    assert read(owner, saved["id"]) == result
    approval(owner, saved["id"])
    assert len(boundaries) == 2
    for name in ("code.py", "program-0.py", "trajectory.json", "graph.json"):
        assert owner.get(f"/api/analyses/{saved['id']}/files/{name}").status_code == 200


def test_rejected_gate_is_not_completion_and_stops_before_computation(owner, boundaries, monkeypatch):
    from test_agents_journey import tool_turn
    monkeypatch.setattr(model_access, "invoke_tools", lambda request: tool_turn("stop_analysis", {"outcome": "unsupported", "reason": "The requested intervention is not defined by the frozen study.", "limitations": []}))
    _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "blocked" and not boundaries
    assert saved["agents"][0]["status"] == "blocked"
    assert saved["agents"][1]["status"] == "pending"
    assert saved["outcome"] == "unsupported"


def test_two_bad_submissions_preserve_rejections(owner, boundaries, monkeypatch):
    from test_agents_journey import agent_detail, tool_turn
    monkeypatch.setattr(model_access, "invoke_tools", lambda request: tool_turn("publish_handoff", {"summary": "Premature submission"}))
    _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "blocked" and not boundaries
    detail = agent_detail(owner, saved, "study")
    receipts = [r for session in detail["sessions"] for turn in session["attempts"] for r in turn["tool_receipts"].values() if r.get("complete")]
    assert len(receipts) == 3 and all(not r["result"]["ok"] for r in receipts)
    assert saved["agents"][0]["submissions"] == 3


def test_snapshot_receipts_and_approval_cannot_change(owner, boundaries):
    saved = prepared(owner)
    assert (
        owner.post(
            f"/api/analyses/{saved['id']}/approve",
            json={"snapshot_hash": saved["snapshot_hash"], "program_hash": "0" * 64},
        ).status_code
        == 409
    )
    with Session(engine()) as db:
        row = db.get(AnalysisRun, UUID(saved["id"]))
        changed = deepcopy(row.snapshot)
        changed["question"] = "Different scientific target"
        for values in ({"snapshot": changed}, {"calls": 0}, {"receipts": {}}, {"approved_hash": "x" * 64}):
            with pytest.raises(DBAPIError):
                db.execute(update(AnalysisRun).where(AnalysisRun.id == row.id).values(**values))
                db.commit()
            db.rollback()


def test_report_recovery_never_refits(owner, boundaries, monkeypatch):
    from test_agents_journey import native_script, selected_call
    saved = prepared(owner)
    def fail_report(request):
        response = native_script(request)
        if selected_call(response)["name"] == "set_report":
            raise OSError("Provider interrupted")
        return response
    monkeypatch.setattr(model_access, "invoke_tools", fail_report)
    failed = approval(owner, saved["id"])
    assert failed["status"] == "failed" and failed["retryable"] and len(boundaries) == 2
    monkeypatch.setattr(model_access, "invoke_tools", native_script)
    assert owner.post(f"/api/analyses/{saved['id']}/retry").status_code == 202
    jobs.process(UUID(saved["id"]))
    recovered = read(owner, saved["id"])
    assert recovered["status"] == "succeeded" and len(boundaries) == 2
    assert recovered["calls"] == failed["calls"] + 3
    assert [a["model_calls"] for a in recovered["agents"][:3]] == [a["model_calls"] for a in failed["agents"][:3]]


def test_scientific_failure_has_no_repair(owner, boundaries, monkeypatch):
    from test_agents_journey import native_script, selected_call, tool_turn
    saved = prepared(owner)
    original = sandbox.run
    monkeypatch.setattr(sandbox, "run", lambda *args: {**original(*args), "ok": False, "stderr": "Singular primary design", "facts": {}})
    def unsupported(request):
        response = native_script(request)
        if selected_call(response)["name"] == "publish_handoff" and "submit_program" in request.functions_json:
            return tool_turn("stop_analysis", {"outcome": "unidentified", "reason": "The approved primary design is singular.", "limitations": []})
        return response
    monkeypatch.setattr(model_access, "invoke_tools", unsupported)
    result = approval(owner, saved["id"])
    assert result["status"] == "blocked" and not result["retryable"]
    assert len(result["programs"]) == 1 and len(boundaries) == 2
    assert not result["programs"][0]["execution"]["valid"]
    assert result["result"] is None and result["agents"][-1]["status"] == "pending"


def test_programming_repair_requires_new_exact_approval(owner, boundaries, monkeypatch):
    saved = prepared(owner)
    original = sandbox.run

    def fail(*args):
        return {
            **original(*args),
            "ok": False,
            "stderr": "NameError: name 'missing' is not defined",
            "facts": {},
        }

    monkeypatch.setattr(sandbox, "run", fail)
    packets = []

    def repair_model(schema, system, payload, model, **kwargs):
        packets.append(payload)
        result = scripted(schema, system, payload, model, **kwargs)
        if schema is Program:
            result.code += "\n# Programming correction only\n"
        return result

    monkeypatch.setattr(model_access, "structured", model_double(repair_model))
    revised = approval(owner, saved["id"])
    assert revised["status"] == "awaiting_approval", revised
    assert revised["snapshot"] == saved["snapshot"] and len(revised["programs"]) == 2
    assert revised["programs"][1]["diff"] and revised["programs"][0]["execution"]["ok"] is False
    assert all("execution" not in packet for packet in packets)
    assert (
        owner.post(
            f"/api/analyses/{saved['id']}/approve",
            json={"snapshot_hash": saved["snapshot_hash"], "program_hash": saved["programs"][0]["hash"]},
        ).status_code
        == 409
    )
    assert len(boundaries) == 2
    monkeypatch.setattr(sandbox, "run", original)
    result = approval(owner, saved["id"])
    assert result["status"] == "succeeded", result
    assert len(boundaries) == 3 and boundaries[1]["id"] != boundaries[2]["id"]


def test_sse_replay_and_owner_isolation(owner, context_chat, boundaries):
    from analysis_context_fixture import ready_revision

    revision = ready_revision(owner, context_chat["version_id"], context_chat["context_revision_id"])
    context_chat = owner.post(
        f"/api/versions/{context_chat['version_id']}/conversations",
        json={"title": "Bound SSE analysis", "submission_id": str(uuid4()), "context_revision_id": revision["id"]},
    ).json()
    run, _ = prepare(owner, context_chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    last = saved["events"][-1]["sequence"]
    response = owner.get(f"/api/analyses/{run['id']}/events?after={last - 1}")
    assert f"id: {last}\n" in response.text and "id: 1\n" not in response.text
    assert "event: settled" in response.text
    assert saved["context"]["revision_id"] == context_chat["context_revision_id"]
    owner.post("/api/auth/login", json={"email": "other@example.com", "password": PASSWORD})
    for suffix in ("", "/events", "/files/trajectory.json", "/files/context.json"):
        assert owner.get(f"/api/analyses/{run['id']}{suffix}").status_code == 404


def test_queue_failure_is_atomic(owner, monkeypatch):
    _, _, chat = create(owner)
    readiness = owner.get(f"/api/conversations/{chat['id']}/analysis-readiness").json()
    monkeypatch.setattr(delivery, "enqueue", lambda *args: (_ for _ in ()).throw(OSError("Queue unavailable")))
    response = owner.post(
        f"/api/conversations/{chat['id']}/analyses",
        json={"submission_id": str(uuid4()), "question": readiness["question"], "expected_context_revision_id": readiness["context_revision_id"], "expected_readiness_fingerprint": readiness["fingerprint"]},
    )
    assert response.status_code == 503
    assert owner.get(f"/api/conversations/{chat['id']}/analyses").json() == []


def test_assumption_dependencies_and_cycle_gates():
    context = empty_content(uuid4(), ["group", "score"])
    packet = {"column_identities": [c.identity.model_dump() for c in context.columns.values()]}
    target = scripted(Target, "", packet, "test")
    target.measurement = Claim(
        statement="Assumed measurement", basis="assumption", references=["assumption:measurement"]
    )
    gates = domain.checks(target, context, {})
    assert not all(g.passed for g in gates)
    target.measurement = Claim(**claim())
    prior = {"target": target.model_dump()}
    prior["treatments"] = scripted(Treatments, "", packet, "test").model_dump()
    prior["roles"] = Roles(**decision()).model_dump()
    causal = scripted(CausalAccount, "", {"decisions": prior}, "test")
    assert all(g.passed for g in domain.checks(causal, context, prior))
    causal.edges.append(
        causal.edges[0].model_copy(
            update={"source": causal.edges[0].target, "target": causal.edges[0].source}
        )
    )
    assert not next(
        g for g in domain.checks(causal, context, prior) if g.rule == "causal_graph_consistency"
    ).passed


def test_invalid_context_action_has_one_correction_and_preserves_failure(owner, boundaries, monkeypatch):
    from test_agents_journey import agent_detail, native_script, tool_turn
    called = 0
    def invalid_once(request):
        nonlocal called
        called += 1
        if called == 1:
            return tool_turn("add_context", {"text": "An invented source"})
        return native_script(request)
    monkeypatch.setattr(model_access, "invoke_tools", invalid_once)
    saved = prepared(owner)
    first = agent_detail(owner, saved, "study")["sessions"][0]["attempts"][0]
    assert first["tool_receipts"]["0"]["result"]["ok"] is False
    assert "add_context" not in first["tool_receipts"]["0"]["result"]["available"]
    assert saved["agents"][0]["status"] == "accepted"
    assert saved["calls"] == 17


def test_field_citations_and_namespaced_ledger_ids_preserve_authority():
    from backend.analysis.contracts import Assumption

    context = empty_content(uuid4(), ["group", "score"])
    packet = {"column_identities": [c.identity.model_dump() for c in context.columns.values()]}
    target = scripted(Target, "", packet, "test")
    target.citations = []
    assert all(g.passed for g in domain.checks(target, context, {}))
    target.timing = Claim(statement="Before follow-up", basis="assumption", references=["assumption:timing"])
    gates = domain.checks(target, context, {})
    assert not gates[0].passed and "assumption:timing" in gates[0].detail
    target.assumptions = [
        Assumption(
            id="assumption:timing",
            statement="Before follow-up",
            rationale="Working premise",
            affects=["timing"],
        )
    ]
    resolved = domain.resolve(target, {})
    assert resolved.assumptions[0].id == "timing"
    resolved.timing.references.append("question")
    assert all(g.passed for g in domain.checks(resolved, context, {}))
    resolved.timing.basis = "supplied"
    assert not domain.checks(resolved, context, {})[0].passed


def test_report_request_literals_do_not_authorize_unregistered_results():
    from historical_analysis.runtime import bind_requested_numbers

    from backend import numerical as evidence

    question = "Estimate assignment 1 versus 0 with a 95% interval."
    value = Conclusion(
        **decision(), report=AnalysisReport(text="Compare 1 with 0 using a 95% interval; estimate 9.25.")
    )
    resolved = bind_requested_numbers(value, question)
    assert {n.value for n in resolved.report.numbers} == {0, 1, 95}
    with pytest.raises(ValueError, match="matching numbers metadata"):
        evidence.validate(resolved.report, {"question": {"value": question}})
    value.citations = []
    assert not bind_requested_numbers(value, question).report.numbers


def test_code_repair_cannot_add_statistical_fallback_or_change_contrast():
    original = "estimate = treated.mean() - control.mean()\nalpha = .05\n"
    domain.validate_repair(original, "import pandas as pd\n" + original)
    domain.validate_repair(original, "input_columns = [{'position': 0, 'name': 'a'}]\n" + original)
    for changed in (
        original.replace(".05", ".10"),
        original.replace(".mean()", ".median()"),
        "if len(treated) > 5:\n    " + original.replace("\n", "\n    "),
        original + "if variance == 0:\n    ci_low = estimate\n",
    ):
        with pytest.raises(ValueError, match="Code-only repair"):
            domain.validate_repair(original, changed)


def test_recovery_after_freeze_commit_restores_approval_event_without_new_work(
    owner, boundaries, monkeypatch
):
    from backend.analysis import agents

    original = agents.emit

    def interrupt_event(db, row, key, stage, kind, *args, **kwargs):
        if kind == "approval_requested":
            raise OSError("Interrupted after the scientific receipt committed")
        return original(db, row, key, stage, kind, *args, **kwargs)

    monkeypatch.setattr(agents, "emit", interrupt_event)
    _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    failed = read(owner, run["id"])
    assert failed["status"] == "failed" and len(failed["programs"]) == 1
    monkeypatch.setattr(agents, "emit", original)
    assert owner.post(f"/api/analyses/{run['id']}/retry").status_code == 202
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "awaiting_approval" and saved["calls"] == failed["calls"]
    assert len(boundaries) == 1 and saved["snapshot_hash"] == failed["snapshot_hash"]
    assert sum(e["kind"] == "approval_requested" for e in saved["events"]) == 1


def test_context_mode_debates_then_applies_and_binds_future_turns(owner, context_chat, monkeypatch):
    cid = context_chat["id"]
    original = context_chat["context_revision_id"]

    def infer(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        if payload["current_message"].startswith("Please record"):
            return Reply(
                text="This is a proposed correction. Apply changes to save it.",
                proposal_passage=payload["current_message"],
                context_edits=[
                    ContextEdit(
                        action="replace",
                        key="relationships.known",
                        statement=EditStatement(
                            value="Group assignment preceded the score measurement.",
                            participants=[
                                {"kind": "column", "id": c} for c in payload["context"]["content"]["columns"]
                            ],
                        ),
                        reason="Explicit correction",
                    )
                ],
            )
        return Reply(text="We can discuss this without changing the supplied account.")

    monkeypatch.setattr(model_access, "structured", model_double(infer))

    def turn(text, expected):
        response = owner.post(
            f"/api/conversations/{cid}/turns",
            json={"submission_id": str(uuid4()), "expected_turn": expected, "text": text, "mode": "context"},
        )
        assert response.status_code == 202, response.text
        chat_jobs.process(UUID(response.json()["id"]), Analyses())
        return response.json()["id"]

    turn("Could timing affect the interpretation?", 0)
    assert owner.get(f"/api/conversations/{cid}").json()["context_revision_id"] == original
    tid = turn("Please record that assignment preceded score measurement.", 1)
    before = owner.get(f"/api/conversations/{cid}").json()
    assert before["turns"][-1]["status"] == "succeeded", before
    assert before["context_revision_id"] == original
    response = owner.post(f"/api/conversations/{cid}/turns/{tid}/apply-context")
    assert response.status_code == 200, response.text
    new = response.json()["context_revision_id"]
    assert new != original and response.json()["turns"][0]["context_revision_id"] == original
    assert (
        owner.post(f"/api/conversations/{cid}/turns/{tid}/apply-context").json()["context_revision_id"] == new
    )
    next_id = turn("What does the updated account mean?", 2)
    with Session(engine()) as db:
        assert db.get(ChatTurn, UUID(next_id)).inputs["context_revision_id"] == new
        assert len(list(db.scalars(select(ChatContextSelection)))) == 1


def test_chat_mode_cannot_publish_context_edits(owner, context_chat, monkeypatch):
    def infer(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        return Reply(
            text="Proposed edit.",
            proposal_passage=payload["current_message"],
            context_edits=[
                ContextEdit(action="withdraw", key="relationships.known", reason="Unsupported correction")
            ],
        )

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    cid = context_chat["id"]
    response = owner.post(
        f"/api/conversations/{cid}/turns",
        json={"submission_id": str(uuid4()), "expected_turn": 0, "text": "Discuss this", "mode": "chat"},
    )
    chat_jobs.process(UUID(response.json()["id"]), Analyses())
    saved = owner.get(f"/api/conversations/{cid}").json()
    assert saved["turns"][0]["status"] == "failed" and saved["turns"][0]["reply"] is None
    assert saved["context_revision_id"] == context_chat["context_revision_id"]


def test_multiple_treatments_and_all_requested_effects_are_frozen():
    context = empty_content(uuid4(), ["group", "score", "second_offer"])
    columns = [c.identity.model_dump() for c in context.columns.values()]
    packet = {"column_identities": columns}
    target = scripted(Target, "", packet, "test")
    treatment = scripted(Treatments, "", packet, "test")
    second = variable(next(c for c in columns if c["name"] == "second_offer"))
    treatment.treatments.append(
        treatment.treatments[0].model_copy(update={"variable": domain.Variable.model_validate(second)})
    )
    treatment.effects[0].treatment_ids.append(second["id"])
    treatment.effects.append(
        treatment.effects[0].model_copy(
            update={"id": "second", "priority": "secondary", "treatment_ids": [second["id"]]}
        )
    )
    prior = {"target": target.model_dump()}
    assert all(g.passed for g in domain.checks(treatment, context, prior))
    prior["treatments"] = treatment.model_dump()
    roles = Roles(**decision())
    assert all(g.passed for g in domain.checks(roles, context, prior))
    prior["roles"] = roles.model_dump()
    plan = scripted(Plan, "", {}, "test")
    assert not all(g.passed for g in domain.checks(plan, context, prior))
    plan.effects.append(plan.effects[0].model_copy(update={"effect_id": "second"}))
    assert all(g.passed for g in domain.checks(plan, context, prior))


def test_safe_projection_preserves_pre_outcomes_and_withholds_proxies():
    from backend.analysis.contracts import RoleAssignment

    context = empty_content(uuid4(), ["group", "score", "baseline", "leak", "id"])
    columns = [c.identity.model_dump() for c in context.columns.values()]
    packet = {"column_identities": columns}
    target = scripted(Target, "", packet, "test")
    treatments = scripted(Treatments, "", packet, "test")
    roles = Roles(
        **decision(),
        assignments=[
            RoleAssignment(
                variable=variable(c),
                roles=["pre_outcome"] if c["name"] == "baseline" else ["unknown"],
                timing="before_treatment" if c["name"] != "id" else "unknown",
                account=claim(),
                timing_evidence=claim(),
                outcome_proxy=c["name"] == "leak",
                reason="Fixture timing account",
            )
            for c in columns
            if c["name"] in {"baseline", "leak", "id"}
        ],
    )
    causal = CausalAccount(**decision(), identification="Fixture projection test")
    frame = domain.Frame(
        target=target, treatments=treatments, roles=roles, causal=causal, graph_hash=domain.graph_hash(causal)
    )
    raw, manifest = domain.projection(
        frame, context, b"group,score,baseline,leak,id\n1,99999,3,88888,a\n0,77777,4,66666,b\n"
    )
    assert b"99999" not in raw and b"88888" not in raw and b"77777" not in raw
    assert len(manifest["columns"]) == 2 and len(manifest["withheld"]) == 3
    assert {
        c.identity.name for c in context.columns.values() if c.identity.column_id in manifest["columns"]
    } == {"group", "baseline"}


@pytest.mark.parametrize("missing_secondary", [False, True])
def test_two_treatments_use_the_same_approval_and_execution_gate(
    owner, boundaries, monkeypatch, missing_secondary
):
    def two_effects(schema, system, payload, model, **kwargs):
        value = legacy_scripted(schema, system, payload, model, **kwargs)
        if schema is Target:
            value.requested_effects.append(value.requested_effects[0].model_copy(update={"id": "secondary", "priority": "secondary"}))
        if schema is Treatments:
            column = next(c for c in payload["column_identities"] if c["name"] == "second")
            values, comparison = static_comparison(column["column_id"], "second", "number", ("0", "1"))
            value.values.extend(values)
            item = value.treatments[0].model_copy(deep=True)
            item.variable = domain.Variable.model_validate(variable(column))
            value.treatments.append(item)
            value.assignments.append(value.assignments[0].model_copy(update={"id": "second_assignment", "treatment_ids": [item.variable.id]}))
            value.effects.append(
                value.effects[0].model_copy(
                    update={"id": "secondary", "priority": "secondary", "treatment_ids": [item.variable.id], "comparison": comparison}
                )
            )
        if schema is CausalAccount:
            item = payload["decisions"]["treatments"]["treatments"][1]["variable"]
            value.nodes.append(
                value.nodes[0].model_copy(
                    update={"id": item["id"], "column_ids": item["column_ids"], "label": item["label"]}
                )
            )
            value.edges.append(value.edges[0].model_copy(update={"source": item["id"]}))
            value.identifications.append(value.identifications[0].model_copy(update={"effect_id": "secondary"}))
        if schema is Plan:
            value.effects.append(value.effects[0].model_copy(update={"effect_id": "secondary"}))
            value.multiplicity = "Report both prespecified intervals as marginal intervals; make no familywise significance claim."
        return value

    from test_agents_journey import native_script
    monkeypatch.setattr(model_access, "invoke_tools", lambda request: native_script(request, factory=two_effects))
    original = sandbox.run

    def compute(*args):
        result = original(*args)
        if args[3] and not missing_secondary:
            result["facts"].update(
                {"secondary_estimate": 4, "secondary_ci_low": 2, "secondary_ci_high": 6, "secondary_n": 3}
            )
        return result

    monkeypatch.setattr(sandbox, "run", compute)
    uploaded = owner.post(
        "/api/datasets",
        data={"title": "Two treatments", "submission_id": str(uuid4())},
        files={"file": ("two.csv", b"group,second,score\na,0,2\na,1,4\nb,0,8\n", "text/csv")},
    ).json()
    from analysis_context_fixture import ready_revision

    revision = ready_revision(owner, uploaded["version"]["id"], question="Estimate the separate group and second intervention effects on score.", treatment_names=("group", "second"))
    chat = owner.post(
        f"/api/versions/{uploaded['version']['id']}/conversations",
        json={"title": "Two effects", "submission_id": str(uuid4()), "context_revision_id": revision["id"]},
    ).json()
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    from test_agents_journey import agent_detail
    assert saved["status"] == "awaiting_approval", [r.get("result") for session in agent_detail(owner, saved, "design")["sessions"] for turn in session["attempts"] for r in turn["tool_receipts"].values() if not r.get("result", {}).get("ok")][-2:]
    assert len(saved["snapshot"]["plan"]["effects"]) == 2
    assert len(next(csv.reader(io.StringIO(boundaries[0]["csv"].decode())))) == 2
    result = approval(owner, run["id"])
    assert result["snapshot_hash"] == saved["snapshot_hash"] and len(boundaries) == 2
    assert result["status"] == ("blocked" if missing_secondary else "succeeded")
    if missing_secondary:
        assert (
            "effect secondary"
            in result["programs"][0]["execution"]["validation_error"]
        )


def test_diagnostic_budget_includes_failed_runs_and_cannot_expand(owner, boundaries, monkeypatch):
    original = sandbox.run
    monkeypatch.setattr(
        sandbox, "run", lambda *args: {**original(*args), "ok": False, "facts": {}, "stderr": "Probe failed"}
    )

    from test_agents_journey import native_script, tool_turn
    def endless(request):
        contents = json.loads(request.contents_json)
        if "set_method" in request.functions_json and any(
            part.get("function_response", {}).get("name") == "freeze_causal_account" and part["function_response"]["response"].get("ok")
            for message in contents for part in message.get("parts", [])
        ):
            return tool_turn("run_diagnostic", {"action": "run_code", "question": "Repeated feasibility probe", "consequence": "Inspect feasibility", "code": "import json\njson.dump({},open('facts.json','w'))"})
        return native_script(request)
    monkeypatch.setattr(model_access, "invoke_tools", endless)
    _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "blocked" and len(boundaries) == 4
    assert saved["budgets"]["diagnostic_runs"] == 4 and saved["snapshot"] is None
    assert all(not record["result"]["ok"] for record in saved["diagnostics"])
    jobs.process(UUID(run["id"]))
    assert len(boundaries) == 4


def test_checkpoint_acceptance_and_event_roll_back_together(owner, boundaries, monkeypatch):
    from backend.analysis import agents
    _, _, chat = create(owner)
    run, _ = prepare(owner, chat)
    original = agents.emit
    def unavailable(db, row, key, stage, kind, *args, **kwargs):
        if stage == "study" and kind == "handoff_accepted":
            raise OSError("Event persistence unavailable")
        return original(db, row, key, stage, kind, *args, **kwargs)
    monkeypatch.setattr(agents, "emit", unavailable)
    jobs.process(UUID(run["id"]))
    saved = read(owner, run["id"])
    assert saved["status"] == "failed" and not boundaries
    assert saved["agents"][0]["status"] == "accepted"
    original_hash = saved["agents"][0]["handoff_hash"]
    monkeypatch.setattr(agents, "emit", original)
    assert owner.post(f"/api/analyses/{run['id']}/retry").status_code == 202
    jobs.process(UUID(run["id"]))
    recovered = read(owner, run["id"])
    assert recovered["status"] == "awaiting_approval"
    assert recovered["agents"][0]["handoff_hash"] == original_hash
    assert recovered["agents"][0]["model_calls"] == saved["agents"][0]["model_calls"]


def test_main_chat_reads_exact_saved_analysis_code_without_execution(owner, boundaries, monkeypatch):
    from backend.chat.contracts import AnalysisFile

    saved = prepared(owner)
    completed = approval(owner, saved["id"])
    observed = []

    def discuss(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation(
                analysis_files=[AnalysisFile(analysis_id=UUID(saved["id"]), filename="program-0.py")]
            )
        observed.append(payload)
        assert payload["analysis_file_evidence"][0]["content"] == completed["programs"][0]["program"]["code"]
        literals = {
            key: value
            for key, value in payload["numeric_evidence"].items()
            if value["basis"] == "program_literal_not_result"
        }
        source = next(key for key, value in literals.items() if value["value"] == 2)
        file_source = next(
            key
            for key, value in payload["analysis_references"].items()
            if value == f"analysis.{saved['id']}.file.program-0.py"
        )
        return Reply(
            text="The saved fixture code contains the literal 2.",
            context_keys=[f"analysis.{saved['id']}.primary_estimate"],
            analysis_references=[file_source],
            numbers=[ReplyNumber(value=2, source=source)],
        )

    monkeypatch.setattr(model_access, "structured", model_double(discuss))
    response = owner.post(
        f"/api/conversations/{saved['conversation_id']}/turns",
        json={
            "submission_id": str(uuid4()),
            "expected_turn": 0,
            "text": "Inspect the saved analysis code without computing again.",
        },
    )
    assert response.status_code == 202
    chat_jobs.process(UUID(response.json()["id"]), Analyses())
    turn = owner.get(f"/api/conversations/{saved['conversation_id']}").json()["turns"][-1]
    assert turn["status"] == "succeeded", turn
    assert turn["reply"]["context_keys"] == []
    assert turn["reply"]["numbers"][0]["source"].startswith(f"analysis.{saved['id']}.file.program-0.py.line_")
    assert turn["reply"]["analysis_references"] == [
        f"analysis.{saved['id']}.file.program-0.py",
        f"analysis.{saved['id']}.primary_estimate",
    ]
    assert observed and len(boundaries) == 2

    def forged(schema, *args, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        return Reply(
            text="An unbound analysis claim.", analysis_references=[f"analysis.{uuid4()}.primary_estimate"]
        )

    monkeypatch.setattr(model_access, "structured", model_double(forged))
    response = owner.post(
        f"/api/conversations/{saved['conversation_id']}/turns",
        json={"submission_id": str(uuid4()), "expected_turn": 1, "text": "Discuss the saved estimate."},
    )
    chat_jobs.process(UUID(response.json()["id"]), Analyses())
    turn = owner.get(f"/api/conversations/{saved['conversation_id']}").json()["turns"][-1]
    assert turn["status"] == "failed" and "turn-bound" in turn["error"]
    assert len(boundaries) == 2
