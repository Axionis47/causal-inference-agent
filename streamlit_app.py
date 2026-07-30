"""Causal Studio — governed causal analysis for an arbitrary uploaded dataset."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langgraph.types import Command

from causal.design import ASSUMPTION, options
from causal.kaggle import KaggleError, fetch as fetch_kaggle
from causal.monitoring import (
    append_interaction,
    read_analysis_runs,
    read_design_contracts,
    save_analysis_run,
    save_design_contract,
)
from causal.preparation_agent import run as run_preparation_agent
from causal.profile import profile
from causal.studio_eda import (
    apply_cohort,
    chart_columns,
    chart_sample,
    column_summary,
    grouped_trends,
    kaplan_meier_frame,
    standardized_differences,
)
from causal.studio_export import build_bundle
from causal.studio_prep import (
    apply_repairs,
    build_data_version,
    bundle_fingerprint,
    context_readiness,
    propose_repairs,
    quality_summary,
    read_bundle,
)
from causal.studio_protocols import (
    build_contract,
    contract_hash,
    design_dot,
    get_protocol,
    protocol_as_dict,
    run_preflight,
)
from causal.studio_workflow import build
from causal.suggest import for_lane


ROOT = Path(__file__).parent
RUNS = ROOT / ".studio_runs"
LANES = (
    "observational",
    "matching",
    "iv",
    "survival",
    "did",
    "rdd",
    "mediation",
    "time_series",
)


st.set_page_config(
    page_title="Causal Studio",
    page_icon="∴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1380px; padding-top: 2rem;}
      [data-testid="stMetric"] {background: #f6f7f9; border: 1px solid #e6e8eb;
        padding: .8rem 1rem; border-radius: .6rem;}
      .policy-allow {border-left: 5px solid #138a5b; padding: .7rem 1rem; background: #f0fbf6;}
      .policy-review {border-left: 5px solid #c17b00; padding: .7rem 1rem; background: #fff8e8;}
      .policy-block {border-left: 5px solid #bd2c2c; padding: .7rem 1rem; background: #fff1f1;}
      .subtle {color: #5f6670; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def workflow():
    return build()


@st.cache_data(show_spinner=False)
def _cached_column_summary(dataset_id: str, df: pd.DataFrame) -> pd.DataFrame:
    del dataset_id  # included only as an explicit cache namespace
    return column_summary(df)


def _record_interaction(
    kind: str,
    stage: str,
    payload: dict[str, Any] | None = None,
    *,
    dedupe_slot: str = "",
    run_id: str = "",
) -> dict[str, Any] | None:
    """Append one meaningful server event, avoiding Streamlit rerun duplicates."""
    if not st.session_state.get("interaction_session_id"):
        return None
    safe_payload = payload or {}
    if dedupe_slot:
        signature = json.dumps([kind, safe_payload], sort_keys=True, default=str)
        signatures = st.session_state.setdefault("interaction_signatures", {})
        if signatures.get(dedupe_slot) == signature:
            return None
        signatures[dedupe_slot] = signature
    record = append_interaction(
        session_id=st.session_state.interaction_session_id,
        dataset_id=st.session_state.get("file_id", ""),
        run_id=run_id or st.session_state.get("run_id", ""),
        kind=kind,
        stage=stage,
        payload=safe_payload,
        parent_event_id=st.session_state.get("last_interaction_event_id", ""),
    )
    st.session_state.last_interaction_event_id = record["event_id"]
    st.session_state.setdefault("interaction_events", []).append(record)
    return record


def _contract_dataset_id(file_id: str | None = None, table: str | None = None) -> str:
    return f"{file_id or st.session_state.get('file_id', '')}:{table or st.session_state.get('primary_table', '')}"


def _invalidate_analysis(reason: str) -> None:
    """Archive the visible run and force the edited data back through design review."""
    run_id = st.session_state.get("run_id", "")
    if run_id:
        prior = st.session_state.setdefault("prior_run_ids", [])
        if run_id not in prior:
            prior.append(run_id)
        st.session_state.has_visible_result = True
    st.session_state.run_id = ""
    st.session_state.csv_path = ""
    st.session_state.active_design_contract = {}
    st.session_state.analysis_invalidated_reason = reason


def _reset_for_bundle(file_id: str, tables: dict[str, pd.DataFrame], source: str) -> None:
    primary = max(tables, key=lambda name: (len(tables[name]), len(tables[name].columns)))
    df = tables[primary]
    keep = {
        "file_id": file_id,
        "tables": tables,
        "primary_table": primary,
        "raw_df": df,
        "clean_df": df.copy(),
        "source": source,
        "interaction_session_id": uuid.uuid4().hex,
        "interaction_events": [],
        "interaction_signatures": {},
        "last_interaction_event_id": "",
        "cohort_spec": None,
        "data_version": {},
        "data_version_history": [],
        "prior_run_ids": [],
        "has_visible_result": False,
        "analysis_invalidated_reason": "",
        "design_contract_history": [],
        "active_design_contract": {},
    }
    for key in list(st.session_state):
        if key not in {"_streamlit_script_run_ctx"}:
            del st.session_state[key]
    st.session_state.update(keep)
    st.session_state.repair_plan = propose_repairs(df)
    st.session_state.repair_log = []
    st.session_state.preparation_approved = False
    st.session_state.context = {}
    st.session_state.preparation_plan = {}
    st.session_state.prompt_versions = {}
    st.session_state.run_id = ""
    history = read_design_contracts(_contract_dataset_id(file_id, primary))
    run_history = read_analysis_runs(_contract_dataset_id(file_id, primary))
    st.session_state.design_contract_history = history
    st.session_state.prior_run_ids = [item["run_id"] for item in run_history]
    st.session_state.has_visible_result = bool(run_history)
    # A remembered contract becomes active only after its exact prepared-data
    # fingerprint is recreated and matched during preparation approval.
    st.session_state.active_design_contract = {}
    _record_interaction(
        "dataset_loaded",
        "intake",
        {
            "source": source,
            "tables": len(tables),
            "primary_table": primary,
            "rows": len(df),
            "columns": len(df.columns),
        },
    )


def _choose_primary(table: str) -> None:
    previous = st.session_state.get("primary_table", "")
    if table != previous:
        _invalidate_analysis("The primary analysis table changed.")
    df = st.session_state.tables[table]
    st.session_state.primary_table = table
    st.session_state.raw_df = df
    st.session_state.clean_df = df.copy()
    st.session_state.repair_plan = propose_repairs(df)
    st.session_state.repair_log = []
    st.session_state.preparation_approved = False
    st.session_state.cohort_spec = None
    st.session_state.data_version = {}
    st.session_state.data_version_history = []
    history = read_design_contracts(_contract_dataset_id(table=table))
    run_history = read_analysis_runs(_contract_dataset_id(table=table))
    st.session_state.design_contract_history = history
    st.session_state.prior_run_ids = [item["run_id"] for item in run_history]
    st.session_state.has_visible_result = bool(run_history)
    st.session_state.active_design_contract = {}
    if table != previous:
        _record_interaction(
            "primary_table_changed",
            "intake",
            {"from": previous, "to": table, "rows": len(df), "columns": len(df.columns)},
        )


def _column_select(
    label: str,
    columns: list[str],
    default: str = "",
    *,
    key: str,
    optional: bool = False,
) -> str:
    choices = (["(none)"] if optional else []) + columns
    wanted = "(none)" if optional and not default else default
    index = choices.index(wanted) if wanted in choices else 0
    value = st.selectbox(label, choices, index=index, key=key)
    return "" if value == "(none)" else value


def _lane_arguments(lane: str, df: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    p = profile(df)
    columns = p.names()
    numeric = p.numeric_names()
    intake = {
        "outcome": context.get("outcome", ""),
        "treatment": context.get("treatment", ""),
        "group": "",
        "period": "",
        "running_variable": "",
        "cutoff": None,
        "time_column": "",
    }
    suggestion = for_lane(lane, p, intake, df)
    prefix = f"lane_{lane}"

    if lane in {"observational", "matching"}:
        outcome = _column_select("Outcome", numeric, suggestion.get("outcome", ""), key=f"{prefix}_outcome")
        treatment = _column_select("Treatment", columns, suggestion.get("treatment", ""), key=f"{prefix}_treatment")
        defaults = [c for c in suggestion.get("covariates", []) if c in numeric]
        covariates = st.multiselect("Pre-treatment covariates", numeric, default=defaults, key=f"{prefix}_covs")
        return {"outcome": outcome, "treatment": treatment, "covariates": covariates}

    if lane == "iv":
        outcome = _column_select("Outcome", numeric, suggestion.get("outcome", ""), key=f"{prefix}_outcome")
        treatment = _column_select("Treatment", numeric, suggestion.get("treatment", ""), key=f"{prefix}_treatment")
        instrument = _column_select("Instrument", columns, "", key=f"{prefix}_instrument")
        covariates = st.multiselect("Expert-approved controls (optional)", numeric, key=f"{prefix}_covs")
        return {"outcome": outcome, "treatment": treatment, "instrument": instrument, "covariates": covariates}

    if lane == "survival":
        treatment = _column_select("Treatment", columns, suggestion.get("treatment", ""), key=f"{prefix}_treatment")
        duration = _column_select("Follow-up duration", numeric, suggestion.get("duration", ""), key=f"{prefix}_duration")
        event = _column_select("Event flag (0/1)", columns, suggestion.get("event", ""), key=f"{prefix}_event")
        excluded = {treatment, duration, event}
        defaults = [c for c in suggestion.get("covariates", []) if c in numeric and c not in excluded]
        covariates = st.multiselect("Pre-treatment covariates", [c for c in numeric if c not in excluded], default=defaults, key=f"{prefix}_covs")
        return {"treatment": treatment, "duration": duration, "event": event, "covariates": covariates}

    if lane == "did":
        outcome = _column_select("Outcome", numeric, context.get("outcome", ""), key=f"{prefix}_outcome")
        group = _column_select("Treatment-group column", columns, "", key=f"{prefix}_group")
        period = _column_select("Before/after period column", columns, "", key=f"{prefix}_period")
        unit = _column_select("Unit identifier (optional)", columns, "", key=f"{prefix}_unit", optional=True)
        levels = sorted(map(str, df[group].dropna().unique())) if group else []
        treated_group = st.selectbox("Which group was treated?", levels or [""], key=f"{prefix}_treated")
        return {"outcome": outcome, "group": group, "period": period, "treated_group": treated_group, "unit": unit or None}

    if lane == "rdd":
        outcome = _column_select("Outcome", numeric, context.get("outcome", ""), key=f"{prefix}_outcome")
        running = _column_select("Running/assignment variable", numeric, suggestion.get("running", ""), key=f"{prefix}_running")
        default_cut = float(pd.to_numeric(df[running], errors="coerce").median()) if running else 0.0
        cutoff = st.number_input("Known policy cutoff", value=default_cut, key=f"{prefix}_cutoff")
        return {"outcome": outcome, "running": running, "cutoff": float(cutoff)}

    if lane == "mediation":
        outcome = _column_select("Outcome", numeric, context.get("outcome", ""), key=f"{prefix}_outcome")
        treatment = _column_select("Treatment", columns, context.get("treatment", ""), key=f"{prefix}_treatment")
        mediator = _column_select("Mediator", numeric, suggestion.get("mediator", ""), key=f"{prefix}_mediator")
        excluded = {outcome, treatment, mediator}
        covariates = st.multiselect("Pre-treatment covariates", [c for c in numeric if c not in excluded], key=f"{prefix}_covs")
        return {"outcome": outcome, "treatment": treatment, "mediator": mediator, "covariates": covariates}

    if lane == "time_series":
        outcome = _column_select("Outcome", numeric, context.get("outcome", ""), key=f"{prefix}_outcome")
        time = _column_select("Time column", columns, suggestion.get("time", ""), key=f"{prefix}_time")
        intervention = st.text_input("Intervention date", placeholder="YYYY-MM-DD", key=f"{prefix}_intervention")
        return {"outcome": outcome, "time": time, "intervention": intervention}

    return {}


REQUIRED_ARGS = {
    "observational": ("outcome", "treatment"),
    "matching": ("outcome", "treatment", "covariates"),
    "iv": ("outcome", "treatment", "instrument"),
    "survival": ("treatment", "duration", "event"),
    "did": ("outcome", "group", "period", "treated_group"),
    "rdd": ("outcome", "running", "cutoff"),
    "mediation": ("outcome", "treatment", "mediator"),
    "time_series": ("outcome", "time", "intervention"),
}


def _missing_lane_args(lane: str, kwargs: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_ARGS[lane] if kwargs.get(key) in (None, "", [])]


def _run_context(context: dict[str, Any], lane: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    updated = dict(context)
    if kwargs.get("outcome"):
        updated["outcome"] = kwargs["outcome"]
    if kwargs.get("treatment"):
        updated["treatment"] = kwargs["treatment"]
        updated["exposure"] = kwargs["treatment"]
    elif lane == "did":
        updated["exposure"] = f"{kwargs.get('group')} × {kwargs.get('period')}"
    elif lane == "rdd":
        updated["exposure"] = f"{kwargs.get('running')} crossing {kwargs.get('cutoff')}"
    elif lane == "time_series":
        updated["exposure"] = f"intervention at {kwargs.get('intervention')}"
    return updated


def _render_lane_eda(lane: str, df: pd.DataFrame, kwargs: dict[str, Any]) -> None:
    """Render descriptive, lane-aware views without making causal claims."""
    missing = _missing_lane_args(lane, kwargs)
    if missing:
        st.info("Choose the method fields above to activate design-specific EDA.")
        return
    role_payload = {
        key: value if not isinstance(value, list) else len(value)
        for key, value in kwargs.items()
        if not str(key).startswith("_")
    }
    _record_interaction(
        "lane_eda_rendered",
        "design",
        {"lane": lane, "roles": role_payload},
        dedupe_slot="lane_eda",
    )

    if lane in {"observational", "matching"}:
        outcome, treatment = kwargs["outcome"], kwargs["treatment"]
        sample = chart_sample(df, [outcome, treatment], limit=5_000).dropna()
        fig = px.box(sample, x=treatment, y=outcome, points=False, title="Outcome by treatment arm")
        st.plotly_chart(fig, width="stretch")
        balance = standardized_differences(df, treatment, kwargs.get("covariates", []))
        if not balance.empty:
            fig = px.bar(
                balance,
                x="standardised difference",
                y="covariate",
                orientation="h",
                title="Unadjusted covariate balance",
            )
            fig.add_vline(x=0.1, line_dash="dash", line_color="#c17b00")
            fig.add_vline(x=0.25, line_dash="dash", line_color="#bd2c2c")
            st.plotly_chart(fig, width="stretch")
            st.caption("0.10 is a common review threshold; 0.25 indicates substantial imbalance. This is descriptive, not proof of exchangeability.")
        return

    if lane == "iv":
        outcome, treatment, instrument = kwargs["outcome"], kwargs["treatment"], kwargs["instrument"]
        sample = chart_sample(df, [instrument, treatment, outcome], limit=5_000).dropna()
        if sample[instrument].nunique() <= 30:
            fig = px.box(sample, x=instrument, y=treatment, points=False, title="Treatment by instrument")
        else:
            fig = px.scatter(sample, x=instrument, y=treatment, opacity=0.35, title="Treatment by instrument")
        st.plotly_chart(fig, width="stretch")
        st.warning("This can show instrument relevance. It cannot establish the exclusion restriction or independence.")
        return

    if lane == "survival":
        curve = kaplan_meier_frame(df, kwargs["duration"], kwargs["event"], kwargs["treatment"])
        if curve.empty:
            st.warning("The selected survival fields did not produce a usable event curve.")
        else:
            fig = px.line(curve, x="time", y="survival", color="group", line_shape="hv", title="Kaplan–Meier view by treatment")
            fig.update_yaxes(range=[0, 1.02])
            st.plotly_chart(fig, width="stretch")
        return

    if lane == "did":
        trends = grouped_trends(df, kwargs["period"], kwargs["group"], kwargs["outcome"])
        fig = px.line(
            trends,
            x=kwargs["period"],
            y="mean outcome",
            color=kwargs["group"],
            markers=True,
            hover_data=["rows"],
            title="Group trends",
        )
        st.plotly_chart(fig, width="stretch")
        if trends[kwargs["period"]].nunique() < 3:
            st.warning("Fewer than three periods are visible, so a pre-trend cannot be inspected.")
        return

    if lane == "rdd":
        outcome, running, cutoff = kwargs["outcome"], kwargs["running"], float(kwargs["cutoff"])
        sample = chart_sample(df, [running, outcome], limit=8_000).dropna()
        fig = px.scatter(sample, x=running, y=outcome, opacity=0.3, title="Outcome around the assignment cutoff")
        fig.add_vline(x=cutoff, line_dash="dash", line_color="#bd2c2c")
        st.plotly_chart(fig, width="stretch")
        st.caption("A visible jump is not sufficient; density, bandwidth, continuity, and placebo-cutoff checks still govern the result.")
        return

    if lane == "mediation":
        treatment, mediator, outcome = kwargs["treatment"], kwargs["mediator"], kwargs["outcome"]
        sample = chart_sample(df, [treatment, mediator, outcome], limit=5_000).dropna()
        left, right = st.columns(2)
        left.plotly_chart(
            px.scatter(sample, x=treatment, y=mediator, opacity=0.35, title="Treatment → proposed mediator"),
            width="stretch",
        )
        right.plotly_chart(
            px.scatter(sample, x=mediator, y=outcome, opacity=0.35, title="Proposed mediator → outcome"),
            width="stretch",
        )
        st.warning("These relationships cannot confirm temporal order or absence of mediator–outcome confounding.")
        return

    if lane == "time_series":
        outcome, time_column = kwargs["outcome"], kwargs["time"]
        sample = chart_sample(df, [time_column, outcome], limit=10_000)
        sample[time_column] = pd.to_datetime(sample[time_column], errors="coerce")
        sample[outcome] = pd.to_numeric(sample[outcome], errors="coerce")
        sample = sample.dropna().sort_values(time_column)
        fig = px.line(sample, x=time_column, y=outcome, title="Outcome over time")
        try:
            fig.add_vline(x=pd.Timestamp(kwargs["intervention"]).timestamp() * 1000, line_dash="dash", line_color="#bd2c2c")
        except (TypeError, ValueError):
            st.warning("The intervention date could not be rendered.")
        st.plotly_chart(fig, width="stretch")
        st.caption("The intervention marker is user-supplied; the chart cannot establish that nothing else changed then.")


def _snapshot(run_id: str):
    return workflow().get_state({"configurable": {"thread_id": run_id}})


def _policy_card(policy: dict[str, Any]) -> None:
    decision = policy.get("decision", "block")
    css = "allow" if decision in {"allow", "approved"} else decision
    st.markdown(
        f'<div class="policy-{css}"><strong>Policy decision: {decision.upper()}</strong><br>'
        f'<span class="subtle">{policy.get("version", "")}</span></div>',
        unsafe_allow_html=True,
    )
    for finding in policy.get("findings", []):
        st.write(f"**{finding['severity']} · {finding['rule']}** — {finding['message']}")


st.title("Causal Studio")
st.caption("Upload → understand → prepare → choose a defensible design → test robustness → publish under policy")

with st.sidebar:
    st.subheader("Run state")
    has_data = "raw_df" in st.session_state
    has_agent_plan = bool(st.session_state.get("preparation_plan"))
    has_context = bool(st.session_state.get("context"))
    prepared = bool(st.session_state.get("preparation_approved"))
    has_run = bool(st.session_state.get("run_id"))
    stages = [
        ("1", "Dataset uploaded", has_data),
        ("2", "Preparation investigated", has_agent_plan),
        ("3", "Context confirmed", has_context),
        ("4", "Preparation approved", prepared),
        ("5", "Analysis started", has_run),
    ]
    for number, label, done in stages:
        st.write(f"{'✓' if done else '○'}  {number}. {label}")
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    st.caption(f"LangSmith tracing: {'on' if tracing else 'off (optional)'}")
    if st.button("Reset interface", width="stretch"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()


intake_tab, prep_tab, design_tab, result_tab, report_tab = st.tabs(
    ["1 · Intake", "2 · Understand & prepare", "3 · Design", "4 · Results", "5 · Report & audit"]
)

with intake_tab:
    uploaded = st.file_uploader(
        "Upload a Kaggle bundle or one or more tables",
        type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "pq", "zip"],
        accept_multiple_files=True,
        help="The raw upload is never overwritten. Analysis uses a versioned CSV derived from it.",
    )
    if uploaded:
        files = [(item.name, item.getvalue()) for item in uploaded]
        file_id = bundle_fingerprint(files)
        if st.session_state.get("file_id") != file_id:
            try:
                tables = read_bundle(files)
                _reset_for_bundle(file_id, tables, ", ".join(name for name, _ in files))
                st.success(f"Loaded {len(tables)} table(s) from {len(files)} upload(s)")
            except Exception as exc:
                st.error(f"Could not read this bundle: {exc}")

    with st.expander("Or fetch from Kaggle", expanded=False):
        kaggle_ref = st.text_input("Kaggle URL or owner/dataset slug", key="kaggle_ref")
        if st.button("Fetch Kaggle dataset"):
            try:
                with st.spinner("Downloading and inventorying Kaggle files…"):
                    fetched = fetch_kaggle(kaggle_ref)
                    files = [(str(path.relative_to(path.parent)), path.read_bytes()) for path in fetched.csv_paths]
                    tables = read_bundle(files)
                    _reset_for_bundle(bundle_fingerprint(files), tables, fetched.slug)
                st.success(f"Fetched {fetched.slug}: {len(tables)} CSV table(s)")
            except (KaggleError, Exception) as exc:
                st.error(str(exc))

    if "raw_df" in st.session_state:
        df = st.session_state.raw_df
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", f"{len(df.columns):,}")
        c.metric("Source", st.session_state.source)
        inventory = pd.DataFrame([
            {
                "table": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing cells": int(frame.isna().sum().sum()),
                "duplicates": int(frame.duplicated().sum()),
            }
            for name, frame in st.session_state.tables.items()
        ])
        st.subheader("Bundle inventory")
        st.dataframe(inventory, width="stretch", hide_index=True)
        primary_names = list(st.session_state.tables)
        selected_primary = st.selectbox(
            "Primary analysis table",
            primary_names,
            index=primary_names.index(st.session_state.primary_table),
            help="The preparation agent proposes one table. You remain the authority.",
        )
        if selected_primary != st.session_state.primary_table:
            if st.button("Use this primary table"):
                _choose_primary(selected_primary)
                st.rerun()
        st.dataframe(df.head(20), width="stretch", height=300)

        previous = st.session_state.get("context", {})
        columns = [str(c) for c in df.columns]
        st.subheader("Preparation investigator")
        draft_question = st.text_input(
            "Initial causal question",
            value=previous.get("question", ""),
            placeholder="Did the intervention change the outcome?",
            key="agent_question",
        )
        draft_description = st.text_area(
            "Vague dataset description",
            value=previous.get("description", ""),
            placeholder="Paste the Kaggle description or tell us what little you know.",
            key="agent_description",
        )
        if st.button("Investigate bundle and draft preparation plan", type="primary"):
            _record_interaction(
                "preparation_agent_requested",
                "intake",
                {"tables": len(st.session_state.tables), "question_length": len(draft_question)},
            )
            with st.spinner("The preparation investigator is inspecting tables, columns, repairs, and lane readiness…"):
                plan = run_preparation_agent(
                    st.session_state.tables, draft_question, draft_description
                ).to_dict()
            st.session_state.preparation_plan = plan
            st.session_state.prompt_versions = {
                plan["prompt"]["prompt_id"]: plan["prompt"]
            }
            if plan["primary_table"] in st.session_state.tables:
                _choose_primary(plan["primary_table"])
            drafted = dict(plan.get("context_draft") or {})
            drafted.setdefault("question", draft_question)
            drafted.setdefault("description", draft_description)
            drafted.setdefault("assignment", "")
            drafted.setdefault("unit", "")
            drafted.setdefault("timing", "")
            drafted.setdefault("population", "")
            drafted.setdefault("intended_use", "")
            drafted.setdefault("outcome", "")
            drafted.setdefault("treatment", "")
            drafted.setdefault("high_impact", False)
            st.session_state.context = drafted
            _record_interaction(
                "preparation_agent_completed",
                "intake",
                {
                    "mode": plan["mode"],
                    "model": plan["model"],
                    "prompt_version": plan["prompt"]["version"],
                    "tool_calls": len(plan["trace"]),
                    "recommended_lane": plan.get("recommended_lane", ""),
                    "unresolved_questions": len(plan.get("unresolved_questions", [])),
                },
            )
            st.rerun()

        if st.session_state.get("preparation_plan"):
            plan = st.session_state.preparation_plan
            mode_label = "Vertex AI ReAct" if plan["mode"] == "react" else "Vertex unavailable → deterministic fallback"
            st.info(
                f"Preparation mode: **{mode_label}** · prompt "
                f"`{plan['prompt']['prompt_id']}@{plan['prompt']['version']}` · "
                f"{len(plan['trace'])} tool calls · `{plan['project']}/{plan['location']}/{plan['model']}`"
            )
            st.write(f"**Proposed primary table:** `{plan['primary_table']}` — {plan['primary_reason']}")
            if plan.get("recommended_lane"):
                st.write(
                    f"**Recommended lane:** `{plan['recommended_lane']}` — "
                    f"{plan['recommendation_reason']}"
                )
            if plan.get("failed"):
                st.warning(plan["failed"])
            if plan.get("unresolved_questions"):
                st.write("**Questions requiring human context:**")
                for question in plan["unresolved_questions"]:
                    st.write(f"- {question}")
            with st.expander("Preparation tool trace"):
                st.dataframe(pd.DataFrame(plan["trace"]), width="stretch", hide_index=True)

        # The agent's table selection may have changed the active frame.
        df = st.session_state.raw_df
        columns = [str(c) for c in df.columns]
        previous = st.session_state.get("context", {})
        with st.form("context_contract"):
            st.subheader("Dataset context contract")
            question = st.text_input(
                "Causal question",
                value=previous.get("question", ""),
                placeholder="Did the intervention change the outcome?",
            )
            description = st.text_area(
                "What is this dataset and where did it come from?",
                value=previous.get("description", ""),
                placeholder="A vague description is fine; the fields below make it operational.",
            )
            col1, col2 = st.columns(2)
            unit = col1.text_input("What does one row represent?", value=previous.get("unit", ""))
            assignment_options = ["", "randomized", "rule / threshold", "policy at a date", "self-selected / observational", "observational / unknown"]
            old_assignment = previous.get("assignment", "")
            assignment = col2.selectbox(
                "How did treatment happen?",
                assignment_options,
                index=assignment_options.index(old_assignment) if old_assignment in assignment_options else 0,
            )
            timing = col1.text_input(
                "When were treatment and outcome measured?",
                value=previous.get("timing", ""),
                placeholder="Treatment at signup; outcome 30 days later",
            )
            population = col2.text_input("Target population", value=previous.get("population", ""))
            intended_use = st.text_input(
                "How will the result be used?", value=previous.get("intended_use", "")
            )
            map1, map2 = st.columns(2)
            old_outcome = previous.get("outcome", "")
            outcome = map1.selectbox(
                "Likely outcome column",
                [""] + columns,
                index=([""] + columns).index(old_outcome) if old_outcome in columns else 0,
            )
            old_treatment = previous.get("treatment", "")
            treatment = map2.selectbox(
                "Likely treatment column (optional for DiD/RDD/time series)",
                [""] + columns,
                index=([""] + columns).index(old_treatment) if old_treatment in columns else 0,
            )
            high_impact = st.checkbox(
                "This result may affect people, money, eligibility, or access",
                value=bool(previous.get("high_impact")),
            )
            submitted = st.form_submit_button("Confirm context", type="primary")
        if submitted:
            context = {
                "question": question.strip(),
                "description": description.strip(),
                "unit": unit.strip(),
                "assignment": assignment,
                "timing": timing.strip(),
                "population": population.strip(),
                "intended_use": intended_use.strip(),
                "outcome": outcome,
                "treatment": treatment,
                "high_impact": high_impact,
            }
            ready, missing = context_readiness(context)
            st.session_state.context = context
            _record_interaction(
                "context_confirmed",
                "intake",
                {
                    "ready": ready,
                    "missing_fields": missing,
                    "outcome": outcome,
                    "treatment": treatment,
                    "assignment": assignment,
                    "high_impact": high_impact,
                },
            )
            if ready:
                st.success("Context matches the approved shape.")
            else:
                st.warning("Still needed: " + ", ".join(missing))
    else:
        st.info("Upload a CSV, TSV, Excel, or Parquet file to begin.")


with prep_tab:
    if "raw_df" not in st.session_state:
        st.info("Upload a dataset first.")
    else:
        raw = st.session_state.raw_df
        quality = quality_summary(raw)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Missing cells", f"{quality['missing_cells']:,}")
        m2.metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
        m3.metric("High-missing columns", len(quality["high_missing_columns"]))
        m4.metric("Possible PII columns", len(quality["possible_pii_columns"]))

        groups = chart_columns(raw)
        overview_tab, distribution_tab, relationship_tab, time_tab, cohort_tab = st.tabs(
            ["Schema & missingness", "Distributions", "Relationships", "Time", "Cohort preview"]
        )

        with overview_tab:
            left, right = st.columns([1, 1.35])
            missing = raw.isna().mean().sort_values(ascending=False).head(20)
            with left:
                st.subheader("Missingness")
                if not missing.empty and float(missing.max()) > 0:
                    fig = px.bar(
                        x=missing.values * 100,
                        y=missing.index,
                        orientation="h",
                        labels={"x": "% missing", "y": "column"},
                    )
                    fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.success("No missing values detected.")
            with right:
                st.subheader("Column audit")
                st.dataframe(
                    _cached_column_summary(st.session_state.file_id, raw),
                    width="stretch",
                    hide_index=True,
                    height=390,
                )
            _record_interaction(
                "schema_eda_viewed",
                "data_understanding",
                {"rows": len(raw), "columns": len(raw.columns)},
                dedupe_slot="schema_eda",
            )

        with distribution_tab:
            available_views = []
            if groups["numeric"]:
                available_views.append("Numeric")
            if groups["categorical"]:
                available_views.append("Categorical")
            if not available_views:
                st.info("No numeric or low-cardinality categorical column is available.")
            else:
                distribution_kind = st.radio(
                    "Distribution type", available_views, horizontal=True, key="eda_distribution_kind"
                )
                if distribution_kind == "Numeric":
                    variable = st.selectbox("Numeric column", groups["numeric"], key="eda_numeric")
                    sample = chart_sample(raw, [variable], limit=20_000)
                    fig = px.histogram(sample, x=variable, nbins=40, marginal="box")
                else:
                    variable = st.selectbox("Categorical column", groups["categorical"], key="eda_category")
                    counts = (
                        raw[variable].astype("string").fillna("(missing)")
                        .value_counts(dropna=False).head(30).rename_axis(variable).reset_index(name="rows")
                    )
                    fig = px.bar(counts, x=variable, y="rows")
                fig.update_layout(height=390, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
                _record_interaction(
                    "eda_distribution_changed",
                    "data_understanding",
                    {"kind": distribution_kind.lower(), "column": variable},
                    dedupe_slot="distribution",
                )

        with relationship_tab:
            if len(groups["numeric"]) < 2:
                st.info("Two numeric columns are required for a relationship view.")
            else:
                context = st.session_state.get("context", {})
                numeric = groups["numeric"]
                default_x = context.get("treatment") if context.get("treatment") in numeric else numeric[0]
                default_y = context.get("outcome") if context.get("outcome") in numeric else numeric[1]
                x_col, y_col, color_col = st.columns(3)
                x = x_col.selectbox("X axis", numeric, index=numeric.index(default_x), key="eda_x")
                y = y_col.selectbox("Y axis", numeric, index=numeric.index(default_y), key="eda_y")
                color_choices = ["(none)"] + groups["categorical"]
                color = color_col.selectbox("Colour/group", color_choices, key="eda_color")
                if x == y:
                    st.warning("Choose different X and Y columns.")
                else:
                    columns = [x, y] + ([] if color == "(none)" else [color])
                    sample = chart_sample(raw, columns, limit=5_000).dropna(subset=[x, y])
                    fig = px.scatter(
                        sample,
                        x=x,
                        y=y,
                        color=None if color == "(none)" else color,
                        opacity=0.35,
                        render_mode="webgl",
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption("This view is descriptive. Association does not assign causal roles.")
                    _record_interaction(
                        "eda_relationship_changed",
                        "data_understanding",
                        {"x": x, "y": y, "group": "" if color == "(none)" else color},
                        dedupe_slot="relationship",
                    )

        with time_tab:
            if not groups["date"] or not groups["numeric"]:
                st.info("A date-like column and a numeric column are required for a time view.")
            else:
                time_left, time_right = st.columns(2)
                time_column = time_left.selectbox("Time column", groups["date"], key="eda_time")
                context_outcome = st.session_state.get("context", {}).get("outcome", "")
                default_value = context_outcome if context_outcome in groups["numeric"] else groups["numeric"][0]
                value_column = time_right.selectbox(
                    "Value", groups["numeric"], index=groups["numeric"].index(default_value), key="eda_time_value"
                )
                timeline = chart_sample(raw, [time_column, value_column], limit=20_000)
                timeline[time_column] = pd.to_datetime(timeline[time_column], errors="coerce")
                timeline[value_column] = pd.to_numeric(timeline[value_column], errors="coerce")
                timeline = timeline.dropna().sort_values(time_column)
                st.plotly_chart(px.line(timeline, x=time_column, y=value_column), width="stretch")
                _record_interaction(
                    "eda_time_view_changed",
                    "data_understanding",
                    {"time": time_column, "value": value_column},
                    dedupe_slot="time_view",
                )

        with cohort_tab:
            st.caption("Filters are previews only until explicitly committed. Committing creates a new analysis population and requires preparation approval again.")
            filter_choices = ["(all rows)"] + groups["all"]
            filter_column = st.selectbox("Preview one cohort filter", filter_choices, key="cohort_column")
            preview_spec: dict[str, Any] | None = None
            if filter_column != "(all rows)":
                numeric_values = pd.to_numeric(raw[filter_column], errors="coerce")
                numeric_fraction = float(numeric_values.notna().mean())
                if numeric_fraction > 0.9 and numeric_values.notna().any():
                    low = float(numeric_values.min())
                    high = float(numeric_values.max())
                    if low == high:
                        st.info("This column is constant and cannot define a useful cohort.")
                    else:
                        selected_range = st.slider(
                            "Included range",
                            min_value=low,
                            max_value=high,
                            value=(low, high),
                            key=f"cohort_range_{filter_column}",
                        )
                        preview_spec = {
                            "kind": "numeric_range", "column": filter_column,
                            "low": selected_range[0], "high": selected_range[1],
                        }
                else:
                    choices = list(raw[filter_column].astype("string").dropna().value_counts().head(50).index)
                    if raw[filter_column].nunique(dropna=True) > 50:
                        st.warning("Only the 50 most common values are offered; high-cardinality columns should not normally define a cohort here.")
                    selected_values = st.multiselect(
                        "Included values", choices, default=choices, key=f"cohort_values_{filter_column}"
                    )
                    preview_spec = {"kind": "categories", "column": filter_column, "values": selected_values}
            preview = apply_cohort(raw, preview_spec)
            kept_fraction = len(preview) / max(1, len(raw))
            st.metric("Preview rows", f"{len(preview):,}", delta=f"{kept_fraction:.1%} retained")
            if preview_spec:
                _record_interaction(
                    "cohort_filter_previewed",
                    "data_understanding",
                    {
                        "column": filter_column,
                        "kind": preview_spec["kind"],
                        "rows_before": len(raw),
                        "rows_after": len(preview),
                        "selected_values": len(preview_spec.get("values", [])),
                    },
                    dedupe_slot="cohort_preview",
                )
                if st.button("Commit this analysis cohort", type="primary"):
                    if preview_spec != st.session_state.get("cohort_spec"):
                        _invalidate_analysis("The committed analysis cohort changed.")
                    st.session_state.cohort_spec = preview_spec
                    st.session_state.data_version = {}
                    st.session_state.preparation_approved = False
                    _record_interaction(
                        "analysis_population_committed",
                        "data_understanding",
                        {
                            "column": filter_column,
                            "kind": preview_spec["kind"],
                            "rows_before": len(raw),
                            "rows_after": len(preview),
                        },
                    )
                    st.success("Cohort committed. Re-approve preparation below before analysis.")
            if st.session_state.get("cohort_spec"):
                st.write("**Committed cohort contract**")
                st.json(st.session_state.cohort_spec)
                if st.button("Clear committed cohort"):
                    _invalidate_analysis("The committed analysis cohort was cleared.")
                    st.session_state.cohort_spec = None
                    st.session_state.data_version = {}
                    st.session_state.preparation_approved = False
                    _record_interaction("analysis_population_cleared", "data_understanding", {})
                    st.rerun()

        st.subheader("Reviewable repair plan")
        plan = st.session_state.repair_plan
        agent_repairs = set(
            (st.session_state.get("preparation_plan") or {}).get("proposed_repairs", [])
        )
        selected: list[str] = []
        if not plan:
            st.success("No mechanical repair was proposed. The raw data can be versioned as-is.")
        for proposal in plan:
            checked = st.checkbox(
                f"{proposal['label']} · {proposal['affected']:,} affected",
                value=proposal["safe_default"] or proposal["id"] in agent_repairs,
                key=f"repair_{st.session_state.file_id}_{proposal['id']}",
                help=proposal["detail"],
            )
            st.caption(proposal["detail"])
            if checked:
                selected.append(proposal["id"])

        st.warning("No outcome imputation or automatic outlier removal is permitted in this layer.")
        if st.button("Apply selected repairs and approve preparation", type="primary"):
            cohort_spec = st.session_state.get("cohort_spec")
            cohort_frame = apply_cohort(raw, cohort_spec)
            clean, log = apply_repairs(cohort_frame, plan, selected)
            if cohort_spec:
                log.insert(0, {
                    "id": "analysis_cohort",
                    "label": f"Committed cohort on {cohort_spec['column']}",
                    "affected": len(raw) - len(cohort_frame),
                    "rows_before": len(raw),
                    "rows_after": len(cohort_frame),
                    "spec": cohort_spec,
                })
            candidate_version = build_data_version(
                raw,
                clean,
                cohort=cohort_spec,
                repairs=log,
            )
            version_history = st.session_state.setdefault("data_version_history", [])
            prior_version = next(
                (
                    item for item in version_history
                    if item.get("version_id") == candidate_version["version_id"]
                ),
                None,
            )
            if prior_version:
                candidate_version = prior_version
            else:
                candidate_version["revision"] = len(version_history) + 1
                version_history.append(candidate_version)
            current_version_id = (st.session_state.get("data_version") or {}).get("version_id")
            if current_version_id != candidate_version["version_id"]:
                _invalidate_analysis(
                    f"Prepared dataset changed to version {candidate_version['version_id']}."
                )
            st.session_state.clean_df = clean
            st.session_state.repair_log = log
            st.session_state.data_version = candidate_version
            matching_contracts = [
                contract for contract in st.session_state.get("design_contract_history", [])
                if (contract.get("data_version") or {}).get("version_id")
                == candidate_version["version_id"]
            ]
            st.session_state.active_design_contract = (
                matching_contracts[-1] if matching_contracts else {}
            )
            st.session_state.preparation_approved = True
            _record_interaction(
                "preparation_approved",
                "preparation",
                {
                    "repairs": selected,
                    "cohort_committed": bool(cohort_spec),
                    "rows_before": len(raw),
                    "rows_after": len(clean),
                    "data_version_id": candidate_version["version_id"],
                    "data_version_revision": candidate_version["revision"],
                },
            )
            st.success(
                f"Created analysis-ready version {candidate_version['revision']} · "
                f"`{candidate_version['version_id']}`: "
                f"{len(clean):,} rows × {len(clean.columns):,} columns"
            )

        if st.session_state.get("preparation_approved"):
            version = st.session_state.get("data_version") or {}
            if version:
                st.caption(
                    f"Prepared-data fingerprint: `{version['prepared_fingerprint'][:16]}` · "
                    f"manifest `{version['manifest_hash'][:16]}`"
                )
            st.dataframe(st.session_state.clean_df.head(20), width="stretch")
            if st.session_state.repair_log:
                st.json(st.session_state.repair_log)


with design_tab:
    if "clean_df" not in st.session_state:
        st.info("Upload and prepare a dataset first.")
    else:
        context = st.session_state.get("context", {})
        ready, missing_context = context_readiness(context)
        if not ready:
            st.warning("Context contract is incomplete: " + ", ".join(missing_context))
        if not st.session_state.get("preparation_approved"):
            st.warning("The preparation plan has not been approved.")

        df = st.session_state.clean_df
        p = profile(df)
        menu = options(df, p, treatment=context.get("treatment"), outcome=context.get("outcome"))
        st.subheader("Eight supported analysis designs")
        st.dataframe(
            pd.DataFrame([
                {
                    "analysis": item.lane,
                    "structurally available": item.available,
                    "why": item.reason,
                    "identifying assumption": item.assumption,
                }
                for item in menu
            ]),
            width="stretch",
            hide_index=True,
        )
        st.caption("A false structural verdict is not a permanent veto: RDD, IV, survival, and mediation require domain input the file cannot infer.")

        recommended = (st.session_state.get("preparation_plan") or {}).get("recommended_lane", "")
        lane = st.selectbox(
            "Analysis design",
            LANES,
            index=LANES.index(recommended) if recommended in LANES else 0,
            format_func=lambda x: x.replace("_", " ").title(),
            key="analysis_lane",
        )
        _record_interaction(
            "analysis_lane_selected",
            "design",
            {"lane": lane, "recommended": lane == recommended},
            dedupe_slot="analysis_lane",
        )
        st.info(f"**Assumption:** {ASSUMPTION[lane]}")
        kwargs = _lane_arguments(lane, df, context)
        missing_args = _missing_lane_args(lane, kwargs)
        if missing_args:
            st.warning("Method configuration still needs: " + ", ".join(missing_args))
        with st.expander("Design-specific exploratory view", expanded=True):
            _render_lane_eda(lane, df, kwargs)

        protocol = get_protocol(lane)
        preflight = run_preflight(df, lane, kwargs, context) if not missing_args else []
        base_contract = build_contract(
            dataset_id=_contract_dataset_id(),
            lane=lane,
            kwargs=kwargs,
            context=context,
            cohort=st.session_state.get("cohort_spec"),
            data_version=st.session_state.get("data_version"),
        )
        configuration_hash = contract_hash(base_contract)

        st.subheader("Pre-estimation protocol")
        protocol_left, protocol_right = st.columns(2)
        with protocol_left:
            st.write(f"**Required visual:** {protocol.visual}")
            st.write("**Before estimation:** " + ", ".join(protocol.pre_checks))
        with protocol_right:
            st.write("**After estimation:** " + ", ".join(protocol.post_checks))
            st.caption("The complete prespecified set runs; the system never selects the most favourable variant.")

        preflight_tab, roles_tab, map_tab = st.tabs(["Preflight findings", "Role ledger", "Design map"])
        with preflight_tab:
            if preflight:
                st.dataframe(pd.DataFrame(preflight), width="stretch", hide_index=True)
                actions = [item["remediation"] for item in preflight if item.get("remediation") and item.get("verdict") != "pass"]
                if actions:
                    st.write("**Bounded reviewer recommendations**")
                    for action in dict.fromkeys(actions):
                        st.write(f"- {action}")
            else:
                st.info("Complete the method configuration to run preflight checks.")
        with roles_tab:
            ledger = base_contract["role_ledger"]
            if ledger:
                st.dataframe(pd.DataFrame(ledger), width="stretch", hide_index=True)
            else:
                st.info("No column roles are currently configured.")
            st.caption("These roles become confirmed only when a human freezes the contract. Correlation alone cannot confirm a confounder.")
        with map_tab:
            st.graphviz_chart(design_dot(base_contract), width="stretch")
            st.caption("This is a proposed causal/design map from selected roles, not a graph learned from correlations.")

        active_contract = st.session_state.get("active_design_contract") or {}
        active_answers = active_contract.get("assumption_answers", {}) if active_contract.get("configuration_hash") == configuration_hash else {}
        has_preflight_failure = any(item.get("verdict") == "fail" for item in preflight)
        if (
            st.session_state.get("has_visible_result")
            and not active_contract.get("configuration_hash") == configuration_hash
        ):
            st.warning("A result already exists for an earlier contract. Freezing these changed fields creates a post-estimation exploratory revision; it cannot replace the original confirmatory result.")
        with st.form(f"design_review_{lane}_{configuration_hash}"):
            st.subheader("Human design review and freeze")
            st.caption("Answer the lane-specific questions before the effect is estimated. These answers are versioned with the contract.")
            answers: dict[str, str] = {}
            for index, question_text in enumerate(protocol.review_questions):
                answers[question_text] = st.text_area(
                    question_text,
                    value=active_answers.get(question_text, ""),
                    key=f"design_answer_{lane}_{index}_{configuration_hash}",
                )
            reviewer = st.text_input("Design reviewer name or role", value=(active_contract.get("approval") or {}).get("reviewer", "causal reviewer"))
            reason = st.text_input(
                "Revision reason",
                value="Initial design freeze" if not st.session_state.get("design_contract_history") else "Design fields or assumptions reviewed",
            )
            map_confirmed = st.checkbox("I reviewed the role ledger, design map, cohort and prespecified diagnostics.")
            freeze_submitted = st.form_submit_button(
                "Freeze design contract",
                type="primary",
                disabled=bool(missing_args or has_preflight_failure),
            )
        if freeze_submitted:
            unanswered = [question for question, answer in answers.items() if not answer.strip()]
            if unanswered or not map_confirmed or not reviewer.strip():
                st.error("Complete every design question, name the reviewer, and confirm the map before freezing.")
            else:
                frozen = build_contract(
                    dataset_id=_contract_dataset_id(),
                    lane=lane,
                    kwargs=kwargs,
                    context=context,
                    cohort=st.session_state.get("cohort_spec"),
                    data_version=st.session_state.get("data_version"),
                    answers={question: answer.strip() for question, answer in answers.items()},
                )
                frozen["configuration_hash"] = configuration_hash
                frozen["contract_hash"] = contract_hash(frozen)
                if active_contract.get("contract_hash") == frozen["contract_hash"]:
                    frozen = active_contract
                else:
                    frozen["revision"] = len(st.session_state.design_contract_history) + 1
                    frozen["parent_contract_hash"] = (
                        st.session_state.design_contract_history[-1]["contract_hash"]
                        if st.session_state.design_contract_history else ""
                    )
                    frozen["change_reason"] = reason.strip()
                    frozen["revision_timing"] = (
                        "post_estimation_exploratory"
                        if st.session_state.get("has_visible_result") else "pre_estimation"
                    )
                    frozen["approval"] = {
                        "approved": True,
                        "reviewer": reviewer.strip(),
                        "role_ledger_and_map_confirmed": True,
                    }
                    st.session_state.design_contract_history.append(frozen)
                    save_design_contract(frozen)
                st.session_state.active_design_contract = frozen
                _record_interaction(
                    "design_contract_frozen",
                    "design",
                    {
                        "lane": lane,
                        "revision": frozen["revision"],
                        "contract_hash": frozen["contract_hash"],
                        "preflight_reviews": sum(item.get("verdict") != "pass" for item in preflight),
                    },
                )
                st.success(f"Frozen revision {frozen['revision']} · {frozen['contract_hash']}")
                st.rerun()

        active_contract = st.session_state.get("active_design_contract") or {}
        contract_is_current = active_contract.get("configuration_hash") == configuration_hash
        if contract_is_current:
            st.success(f"Current design is frozen as revision {active_contract['revision']} · `{active_contract['contract_hash']}`")
        else:
            st.warning("The current fields are not covered by a frozen contract. Review and freeze them before estimation.")

        can_run = (
            ready
            and st.session_state.get("preparation_approved")
            and bool((st.session_state.get("data_version") or {}).get("version_id"))
            and not missing_args
            and not has_preflight_failure
            and contract_is_current
        )
        if st.button("Run analysis and policy checks", type="primary", disabled=not can_run):
            current_run_id = st.session_state.get("run_id", "")
            if current_run_id:
                prior_runs = st.session_state.setdefault("prior_run_ids", [])
                if current_run_id not in prior_runs:
                    prior_runs.append(current_run_id)
            parent_run_id = (
                current_run_id
                or (
                    st.session_state.get("prior_run_ids", [])[-1]
                    if st.session_state.get("prior_run_ids") else ""
                )
            )
            run_id = uuid.uuid4().hex[:12]
            run_dir = RUNS / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            csv_path = run_dir / "analysis_data.csv"
            raw_path = run_dir / "raw_data.csv"
            st.session_state.raw_df.to_csv(raw_path, index=False)
            df.to_csv(csv_path, index=False)
            run_context = _run_context(context, lane, kwargs)
            _record_interaction(
                "analysis_requested",
                "design",
                {
                    "lane": lane,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "configured_fields": sorted(key for key, value in kwargs.items() if value not in (None, "", [])),
                },
                run_id=run_id,
            )
            initial = {
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "csv_path": str(csv_path),
                "source": st.session_state.source,
                "lane": lane,
                "kwargs": kwargs,
                "context": run_context,
                "repairs": st.session_state.repair_log,
                "data_version": st.session_state.data_version,
                "data_quality": quality_summary(df),
                "preparation": st.session_state.get("preparation_plan", {}),
                "prompt_versions": st.session_state.get("prompt_versions", {}),
                "cohort": st.session_state.get("cohort_spec"),
                "interaction_events": list(st.session_state.get("interaction_events", [])),
                "design_contract": active_contract,
                "design_approval": active_contract.get("approval", {}),
                "preflight": preflight,
                "protocol": protocol_as_dict(lane),
                "events": [{"stage": "intake", "status": "approved", "detail": "context and repairs confirmed"}],
            }
            (run_dir / "context.json").write_text(json.dumps(run_context, indent=2))
            (run_dir / "repairs.json").write_text(json.dumps(st.session_state.repair_log, indent=2))
            (run_dir / "data_version.json").write_text(
                json.dumps(st.session_state.data_version, indent=2, default=str)
            )
            (run_dir / "design_contract.json").write_text(json.dumps(active_contract, indent=2, default=str))
            config = {"configurable": {"thread_id": run_id}}
            try:
                with st.spinner("Estimating, checking sensitivity, and evaluating policy…"):
                    workflow().invoke(initial, config=config)
                completed_state = dict(workflow().get_state(config).values)
                if completed_state.get("estimate"):
                    save_analysis_run({
                        "run_id": run_id,
                        "dataset_id": _contract_dataset_id(),
                        "data_version_id": st.session_state.data_version["version_id"],
                        "contract_hash": active_contract["contract_hash"],
                        "parent_run_id": parent_run_id,
                        "status": "estimated",
                    })
                st.session_state.run_id = run_id
                st.session_state.csv_path = str(csv_path)
                st.session_state.has_visible_result = True
                st.session_state.analysis_invalidated_reason = ""
                st.rerun()
            except Exception as exc:
                st.error(f"Workflow failed: {type(exc).__name__}: {exc}")


with result_tab:
    run_id = st.session_state.get("run_id", "")
    if not run_id:
        if st.session_state.get("analysis_invalidated_reason"):
            prior = st.session_state.get("prior_run_ids", [])
            st.warning(
                st.session_state.analysis_invalidated_reason
                + " The previous run remains immutable"
                + (f" as `{prior[-1]}`" if prior else "")
                + "; re-approve preparation, rerun preflight, and freeze a matching contract."
            )
        else:
            st.info("Run an analysis to see results.")
    else:
        snap = _snapshot(run_id)
        state = dict(snap.values)
        if state.get("error"):
            st.error(state["error"])
        if state.get("estimate"):
            estimate = state["estimate"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Estimate", f"{estimate['value']:.4g}")
            c2.metric("Estimand", estimate["estimand"].upper())
            c3.metric("Rows used", f"{estimate['n']:,}")
            c4.metric("Claim strength", state.get("strength", "unknown").upper())
            st.subheader(state.get("headline", "Result"))
            if estimate.get("ci_low") is not None:
                fig = go.Figure(go.Scatter(
                    x=[estimate["value"]], y=[estimate["estimand"]], mode="markers",
                    marker=dict(size=12, color="#2456a6"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[estimate["ci_high"] - estimate["value"]],
                        arrayminus=[estimate["value"] - estimate["ci_low"]],
                    ),
                ))
                fig.add_vline(x=1 if estimate["estimand"] == "hazard_ratio" else 0, line_dash="dash", line_color="#8b929b")
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=20), yaxis_title="")
                st.plotly_chart(fig, width="stretch")

        if state.get("diagnostics"):
            st.subheader("Sensitivity and diagnostics")
            st.dataframe(pd.DataFrame(state["diagnostics"]), width="stretch", hide_index=True)

        if state.get("policy"):
            st.subheader("Policy gate")
            _policy_card(state["policy"])

        waiting_for_review = bool(snap.next and "human_gate" in snap.next)
        if waiting_for_review:
            st.warning("The workflow is paused. A human decision is required before publication.")
            reviewer = st.text_input("Reviewer name or role", value="causal reviewer")
            note = st.text_area("Review note", placeholder="What was checked, and why is publication acceptable or not?")
            approve, reject = st.columns(2)
            if approve.button("Approve publication", type="primary", width="stretch"):
                _record_interaction(
                    "human_publication_decision",
                    "review",
                    {"approved": True, "reviewer": reviewer, "note_length": len(note)},
                    run_id=run_id,
                )
                workflow().invoke(
                    Command(resume={"approved": True, "note": note, "reviewer": reviewer}),
                    config={"configurable": {"thread_id": run_id}},
                )
                st.rerun()
            if reject.button("Reject publication", width="stretch"):
                _record_interaction(
                    "human_publication_decision",
                    "review",
                    {"approved": False, "reviewer": reviewer, "note_length": len(note)},
                    run_id=run_id,
                )
                workflow().invoke(
                    Command(resume={"approved": False, "note": note, "reviewer": reviewer}),
                    config={"configurable": {"thread_id": run_id}},
                )
                st.rerun()


with report_tab:
    run_id = st.session_state.get("run_id", "")
    if not run_id:
        st.info("A report appears after a completed, policy-approved analysis.")
    else:
        snap = _snapshot(run_id)
        state = dict(snap.values)
        if state.get("report"):
            st.markdown(state["report"])
            bundle = build_bundle(state)
            st.download_button(
                "Download executable notebook + audit bundle",
                data=bundle,
                file_name=f"causal_studio_{run_id}.zip",
                mime="application/zip",
                type="primary",
            )
        elif (state.get("policy") or {}).get("decision") == "block":
            st.error("Policy blocked publication. Resolve the findings and rerun.")
        elif state.get("error"):
            st.error(state["error"])
        else:
            st.info("The run is waiting for human policy review in the Results tab.")

        st.subheader("Audit trail")
        if state.get("events"):
            st.dataframe(pd.DataFrame(state["events"]), width="stretch", hide_index=True)
        interaction_events = st.session_state.get("interaction_events") or state.get("interaction_events", [])
        if interaction_events:
            with st.expander("Server interaction events"):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "timestamp": item.get("timestamp"),
                            "stage": item.get("stage"),
                            "kind": item.get("kind"),
                            "event_id": item.get("event_id"),
                            "parent": item.get("parent_event_id"),
                            "payload": json.dumps(item.get("payload", {}), sort_keys=True),
                        }
                        for item in interaction_events
                    ]),
                    width="stretch",
                    hide_index=True,
                )
        with st.expander("Machine-readable run state"):
            st.json(state)
        if state.get("monitoring"):
            st.subheader("Monitoring snapshot")
            st.json(state["monitoring"])
            for alert in state.get("monitoring_alerts", []):
                if alert["severity"] == "critical":
                    st.error(alert["message"])
                else:
                    st.warning(alert["message"])
        preparation = state.get("preparation") or {}
        if preparation.get("trace"):
            with st.expander("Preparation-agent tool calls"):
                st.dataframe(pd.DataFrame(preparation["trace"]), width="stretch", hide_index=True)
