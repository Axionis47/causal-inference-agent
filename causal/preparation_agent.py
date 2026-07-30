"""Bounded ReAct investigator for unknown, possibly multi-table datasets.

The model can only call read-only inspection tools or write to an advisory
plan. Repairs and joins are never executed here. If no model credential is
configured, the same tool surface is exercised by a deterministic fallback so
the product stays testable and usable offline.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .design import options
from .profile import profile
from .prompt_registry import load
from .studio_prep import PII_NAME, propose_repairs, quality_summary


PROMPT_ID = "preparation-agent"
PROMPT_VERSION = "1.1.0"
MODEL = os.getenv("PREPARATION_MODEL", os.getenv("MODEL", "gemini-2.5-flash"))
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCP_PROJECT_ID", "plotpointe"))
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", os.getenv("GCP_LOCATION", "global"))
PROVIDER = "vertex-ai"
MAX_TURNS = 10


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower()) if len(t) > 2}


@dataclass
class PreparationPlan:
    primary_table: str = ""
    primary_reason: str = ""
    context_draft: dict[str, Any] = field(default_factory=dict)
    proposed_repairs: list[str] = field(default_factory=list)
    lane_readiness: list[dict[str, Any]] = field(default_factory=list)
    eligible_lanes: list[str] = field(default_factory=list)
    recommended_lane: str = ""
    recommendation_reason: str = ""
    unresolved_questions: list[str] = field(default_factory=list)
    escalation_reasons: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    prompt: dict[str, str] = field(default_factory=dict)
    provider: str = PROVIDER
    project: str = ""
    location: str = ""
    model: str = ""
    mode: str = ""
    tables_seen: int = 0
    failed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Workspace:
    def __init__(self, tables: dict[str, pd.DataFrame], question: str, description: str):
        if not tables:
            raise ValueError("Preparation requires at least one table")
        self.tables = tables
        self.question = question.strip()
        self.description = description.strip()
        self.selected = ""
        self.selected_reason = ""
        self.context: dict[str, Any] = {
            "question": self.question,
            "description": self.description,
        }
        self.repairs: list[str] = []
        self.unresolved: list[str] = []
        self.escalations: list[str] = []
        self.readiness: list[dict[str, Any]] = []
        self.recommended_lane = ""
        self.recommendation_reason = ""
        self.trace: list[dict[str, Any]] = []
        self.finalized = False

    def _record(self, name: str, arguments: dict[str, Any], result: Any, started: float) -> Any:
        summary = result
        if not isinstance(result, (str, int, float, bool, type(None))):
            summary = json.dumps(result, default=str)[:4000]
        self.trace.append({
            "turn": len(self.trace) + 1,
            "tool": name,
            "arguments": arguments,
            "result": summary,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": "ok",
        })
        return result

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        started = time.perf_counter()
        try:
            fn = getattr(self, f"tool_{name}")
            result = fn(**arguments)
            return self._record(name, arguments, result, started)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            self._record(name, arguments, result, started)
            self.trace[-1]["status"] = "error"
            return result

    def tool_list_tables(self) -> list[dict[str, Any]]:
        out = []
        for name, df in self.tables.items():
            q = quality_summary(df)
            out.append({
                "table": name,
                "rows": q["rows"],
                "columns": q["columns"],
                "missing_cells": q["missing_cells"],
                "duplicates": q["duplicate_rows"],
                "possible_pii": q["possible_pii_columns"],
                "column_names": [str(c) for c in df.columns[:60]],
            })
        return out

    def tool_inspect_table(self, table: str) -> dict[str, Any]:
        df = self.tables[table]
        p = profile(df)
        return {
            "table": table,
            "rows": len(df),
            "columns": [asdict(c) for c in p.columns[:100]],
            "repair_catalogue": propose_repairs(df),
            "quality": quality_summary(df),
        }

    def tool_inspect_column(self, table: str, column: str) -> dict[str, Any]:
        df = self.tables[table]
        if column not in df.columns:
            raise KeyError(f"{table} has no column {column!r}")
        p = profile(df)[column]
        series = df[column]
        sensitive = bool(PII_NAME.search(str(column)))
        samples: list[Any] = []
        if not sensitive:
            samples = [str(x)[:80] for x in series.dropna().drop_duplicates().head(8)]
        return asdict(p) | {
            "table": table,
            "sample_values": "[redacted: possible PII]" if sensitive else samples,
            "top_counts": {} if sensitive else {str(k)[:80]: int(v) for k, v in series.value_counts(dropna=False).head(8).items()},
        }

    def tool_find_join_keys(self, left: str, right: str) -> list[dict[str, Any]]:
        a, b = self.tables[left], self.tables[right]
        shared = [str(c) for c in a.columns if c in b.columns]
        candidates = []
        for column in shared[:30]:
            av, bv = a[column].dropna(), b[column].dropna()
            if av.empty or bv.empty:
                continue
            aset, bset = set(map(str, av.head(100_000))), set(map(str, bv.head(100_000)))
            overlap = len(aset & bset) / max(1, min(len(aset), len(bset)))
            candidates.append({
                "column": column,
                "left_unique_fraction": round(float(av.nunique() / len(av)), 4),
                "right_unique_fraction": round(float(bv.nunique() / len(bv)), 4),
                "value_overlap": round(float(overlap), 4),
                "requires_human_approval": True,
            })
        return sorted(candidates, key=lambda x: x["value_overlap"], reverse=True)

    def tool_select_primary_table(self, table: str, reason: str) -> dict[str, Any]:
        if table not in self.tables:
            raise KeyError(f"No table {table!r}")
        self.selected, self.selected_reason = table, reason.strip()
        return {"selected": table, "reason": self.selected_reason}

    def tool_draft_context(self, **updates: Any) -> dict[str, Any]:
        table = self.selected or next(iter(self.tables))
        columns = set(map(str, self.tables[table].columns))
        for key in ("outcome", "treatment", "group", "period", "time_column", "running_variable"):
            value = str(updates.get(key, "")).strip()
            if value and value not in columns:
                raise ValueError(f"Context names nonexistent {key} column {value!r}")
        allowed = {
            "unit", "assignment", "timing", "population", "intended_use", "outcome",
            "treatment", "group", "period", "time_column", "running_variable", "cutoff",
        }
        self.context.update({k: v for k, v in updates.items() if k in allowed and v not in (None, "")})
        return self.context

    def tool_propose_repair(self, repair_id: str, reason: str) -> dict[str, Any]:
        table = self.selected or next(iter(self.tables))
        catalogue = {r["id"]: r for r in propose_repairs(self.tables[table])}
        if repair_id not in catalogue:
            raise ValueError(f"Repair {repair_id!r} is not in the deterministic catalogue")
        if repair_id not in self.repairs:
            self.repairs.append(repair_id)
        return catalogue[repair_id] | {"agent_reason": reason, "requires_approval": True}

    def tool_check_lane_readiness(self) -> list[dict[str, Any]]:
        table = self.selected or next(iter(self.tables))
        df = self.tables[table]
        p = profile(df)
        treatment = self.context.get("treatment")
        outcome = self.context.get("outcome")
        self.readiness = [
            {
                "lane": item.lane,
                "structurally_available": item.available,
                "reason": item.reason,
                "assumption": item.assumption,
                "needs": item.needs,
            }
            for item in options(df, p, treatment=treatment, outcome=outcome)
        ]
        return self.readiness

    def tool_add_human_question(self, question: str, reason: str) -> dict[str, Any]:
        text = question.strip()
        if text and text not in self.unresolved:
            self.unresolved.append(text)
        if reason.strip() and reason.strip() not in self.escalations:
            self.escalations.append(reason.strip())
        return {"question": text, "reason": reason, "recorded": True}

    def tool_recommend_lane(self, lane: str, reason: str) -> dict[str, Any]:
        allowed = {"observational", "matching", "iv", "survival", "did", "rdd", "mediation", "time_series"}
        if lane not in allowed:
            raise ValueError(f"Unknown lane {lane!r}")
        if not self.readiness:
            self.tool_check_lane_readiness()
        structural = next(r for r in self.readiness if r["lane"] == lane)
        self.recommended_lane = lane
        self.recommendation_reason = reason.strip()
        return {
            "lane": lane,
            "reason": self.recommendation_reason,
            "structurally_available": structural["structurally_available"],
            "needs": structural["needs"],
            "requires_human_confirmation": True,
        }

    def tool_finalize_plan(self, summary: str = "") -> dict[str, Any]:
        if not self.selected:
            self.tool_select_primary_table(next(iter(self.tables)), "Only/default table")
        if not self.readiness:
            self.tool_check_lane_readiness()
        if not self.recommended_lane:
            available = [r["lane"] for r in self.readiness if r["structurally_available"]]
            fallback = "observational" if "observational" in available else (available[0] if available else "observational")
            self.tool_recommend_lane(fallback, "Simplest structurally available design; human confirmation required.")
        self.finalized = True
        return {"finalized": True, "summary": summary, "unresolved": self.unresolved}


TOOL_DECLARATIONS = [
    {
        "name": "list_tables",
        "description": "List tables, shapes, columns, and coarse quality signals.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "inspect_table",
        "description": "Inspect schema, quality, and allowed repair catalogue for one table.",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string"}},
            "required": ["table"],
        },
    },
    {
        "name": "inspect_column",
        "description": "Inspect one real column; possible PII samples are redacted.",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string"}, "column": {"type": "string"}},
            "required": ["table", "column"],
        },
    },
    {
        "name": "find_join_keys",
        "description": "Compare shared columns as possible join keys; never performs the join.",
        "parameters": {
            "type": "object",
            "properties": {"left": {"type": "string"}, "right": {"type": "string"}},
            "required": ["left", "right"],
        },
    },
    {
        "name": "select_primary_table",
        "description": "Select the proposed primary analysis table.",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["table", "reason"],
        },
    },
    {
        "name": "draft_context",
        "description": "Draft supported semantic fields using exact column names.",
        "parameters": {
            "type": "object",
            "properties": {
                key: {"type": "string"}
                for key in (
                    "unit", "assignment", "timing", "population", "intended_use",
                    "outcome", "treatment", "group", "period", "time_column",
                    "running_variable", "cutoff",
                )
            },
        },
    },
    {
        "name": "propose_repair",
        "description": "Add one repair from the deterministic catalogue to the advisory plan.",
        "parameters": {
            "type": "object",
            "properties": {"repair_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["repair_id", "reason"],
        },
    },
    {
        "name": "check_lane_readiness",
        "description": "Check structural readiness for all eight analysis lanes.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "add_human_question",
        "description": "Record an unresolved semantic, join, repair, or design question.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["question", "reason"],
        },
    },
    {
        "name": "recommend_lane",
        "description": "Recommend one of the eight lanes with a reason; never binds execution.",
        "parameters": {
            "type": "object",
            "properties": {"lane": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["lane", "reason"],
        },
    },
    {
        "name": "finalize_plan",
        "description": "Finish the advisory plan after inspecting enough evidence.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}},
    },
]


def _plan(workspace: Workspace, prompt, *, mode: str, failed: str = "") -> PreparationPlan:
    table = workspace.selected or next(iter(workspace.tables))
    if not workspace.readiness:
        workspace.call("check_lane_readiness", {})
    eligible = [r["lane"] for r in workspace.readiness if r["structurally_available"]]
    trace = [
        call | {
            "provider": PROVIDER,
            "project": PROJECT,
            "location": LOCATION,
            "model": MODEL,
            "prompt_id": prompt.prompt_id,
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.sha256,
        }
        for call in workspace.trace
    ]
    return PreparationPlan(
        primary_table=table,
        primary_reason=workspace.selected_reason,
        context_draft=workspace.context,
        proposed_repairs=workspace.repairs,
        lane_readiness=workspace.readiness,
        eligible_lanes=eligible,
        recommended_lane=workspace.recommended_lane,
        recommendation_reason=workspace.recommendation_reason,
        unresolved_questions=workspace.unresolved,
        escalation_reasons=workspace.escalations,
        trace=trace,
        prompt=prompt.lineage(),
        provider=PROVIDER,
        project=PROJECT,
        location=LOCATION,
        model=MODEL if mode == "react" else "deterministic",
        mode=mode,
        tables_seen=len(workspace.tables),
        failed=failed,
    )


def deterministic_fallback(workspace: Workspace, prompt, failed: str = "") -> PreparationPlan:
    inventory = workspace.call("list_tables", {})
    words = _tokens(workspace.question + " " + workspace.description)
    scored = []
    for item in inventory:
        names = _tokens(" ".join(item["column_names"]))
        score = 10 * len(words & names) + min(item["rows"], 1_000_000) / 1_000_000
        scored.append((score, item["rows"], item["table"]))
    _, _, table = max(scored)
    workspace.call("inspect_table", {"table": table})
    workspace.call("select_primary_table", {
        "table": table,
        "reason": "Highest deterministic match between question terms, columns, and usable row count.",
    })
    df = workspace.tables[table]
    columns = [str(c) for c in df.columns]
    mentioned = [c for c in columns if c.lower() in words]
    outcome = next(
        (c for c in mentioned if re.search(r"outcome|target|label|result|response", c, re.I)),
        mentioned[-1] if mentioned else "",
    )
    treatment = next(
        (c for c in mentioned if re.search(r"treat|intervention|exposure|variant|group", c, re.I) and c != outcome),
        next(
            (c for c in mentioned if c != outcome),
            next((c for c in columns if re.search(r"treat|intervention|exposure|variant|group", c, re.I) and c != outcome), ""),
        ),
    )
    updates = {"outcome": outcome, "treatment": treatment}
    workspace.call("draft_context", updates)
    for repair in propose_repairs(df):
        if repair["safe_default"]:
            workspace.call("propose_repair", {"repair_id": repair["id"], "reason": "Mechanical, reversible normalization."})
    for question, reason in (
        ("What does one row represent?", "Unit of analysis cannot be inferred reliably from values."),
        ("How was treatment assigned, and when was the outcome measured?", "Assignment and temporal order determine causal validity."),
    ):
        workspace.call("add_human_question", {"question": question, "reason": reason})
    if len(workspace.tables) > 1:
        workspace.call("add_human_question", {
            "question": "Does the analysis require joining another table to the proposed primary table?",
            "reason": "A multi-table join changes row grain and must be approved.",
        })
    workspace.call("check_lane_readiness", {})
    available = [r["lane"] for r in workspace.readiness if r["structurally_available"]]
    text = (workspace.question + " " + workspace.description).lower()
    preferred = ""
    if re.search(r"through|mediate|mechanism|pathway", text):
        preferred = "mediation"
    elif re.search(r"threshold|cutoff|crossing", text):
        preferred = "rdd"
    elif re.search(r"surviv|time until|hazard|death|churn", text):
        preferred = "survival"
    elif re.search(r"before|after|policy change|difference.in.difference", text):
        preferred = "did" if "did" in available else "time_series"
    if preferred not in available:
        preferred = "observational" if "observational" in available else (available[0] if available else preferred or "observational")
    workspace.call("recommend_lane", {
        "lane": preferred,
        "reason": "Question shape plus deterministic structural readiness; human confirmation required.",
    })
    workspace.call("finalize_plan", {"summary": "Deterministic offline preparation draft."})
    return _plan(workspace, prompt, mode="deterministic", failed=failed)


def _client():
    from google import genai
    from google.genai.types import HttpOptions

    return genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )


def run(tables: dict[str, pd.DataFrame], question: str, description: str) -> PreparationPlan:
    """Run the ReAct investigator, falling back deterministically when offline."""
    prompt = load(PROMPT_ID, PROMPT_VERSION)
    workspace = Workspace(tables, question, description)
    if os.getenv("PREPARATION_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return deterministic_fallback(workspace, prompt, "Vertex AI disabled by PREPARATION_OFFLINE")
    try:
        client = _client()
    except Exception as exc:
        return deterministic_fallback(
            workspace,
            prompt,
            failed=f"Vertex AI client unavailable; deterministic fallback used: {type(exc).__name__}: {exc}"[:300],
        )

    try:
        from google.genai import types

        contents: list[Any] = [types.Content(
            role="user",
            parts=[types.Part(text=(
                f"Question: {question or '(not supplied)'}\n"
                f"Vague dataset description: {description or '(not supplied)'}\n"
                "Investigate this unknown bundle and finalize a preparation plan."
            ))],
        )]
        config = types.GenerateContentConfig(
            system_instruction=prompt.text,
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        for _ in range(MAX_TURNS):
            response = client.models.generate_content(model=MODEL, contents=contents, config=config)
            candidate = response.candidates[0]
            calls = [p.function_call for p in candidate.content.parts or [] if getattr(p, "function_call", None)]
            if not calls:
                break
            contents.append(candidate.content)
            replies = []
            for call in calls:
                arguments = dict(call.args or {})
                result = workspace.call(call.name, arguments)
                replies.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=replies))
            if workspace.finalized:
                break
        if not workspace.finalized:
            workspace.call("finalize_plan", {"summary": "Turn budget reached; plan finalized with current evidence."})
        return _plan(workspace, prompt, mode="react")
    except Exception as exc:
        # Preserve the failed calls, then complete a usable deterministic plan.
        return deterministic_fallback(
            workspace,
            prompt,
            failed=f"ReAct failed; deterministic fallback used: {type(exc).__name__}: {exc}"[:300],
        )
