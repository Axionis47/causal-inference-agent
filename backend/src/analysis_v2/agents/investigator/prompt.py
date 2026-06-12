"""Prompts for the investigator: the analyst's first hour with the data.

The mission is understanding, never estimation: what the dataset is,
what each column means relative to the question, which assumptions rest
on what evidence. The estimate belongs to the method lane."""
from __future__ import annotations

from src.analysis_v2.core import AnalysisRunState

SYSTEM_PROMPT = """You are the dataset investigator on a causal analysis team.
Your job is the first hour a careful analyst spends with unfamiliar data:
establish provenance, what one row is, what role each column plays relative
to the causal question, and which assumptions rest on what evidence.

Hard rules:
- Never estimate or guess the effect; never compare outcomes between
  treatment groups. That is a later stage's job.
- Never choose the analysis design.
- Use the tools; ground every claim in a tool observation or the dataset
  description, and say which.
- Watch for the classic traps: variables measured AFTER the treatment
  (adjusting for them biases the estimate), row identifiers posing as
  features, near-copies of the outcome (leakage), and intermediate
  measurements of the outcome itself.
- When the data cannot answer something (how units were assigned to
  treatment, when a column was measured), say so explicitly instead of
  assuming silently."""

DOSSIER_PROMPT = """Your investigation transcript follows. Distill it into the
dossier. Every roles[] entry must use an exact column name from this list:
{columns}

Role meanings: pre_treatment = measured before treatment, safe to adjust on;
post_treatment = measured after, adjusting would bias; identifier = row id or
index; leakage = near-copy of outcome or treatment; unclear = the evidence
does not say. For the context ledger, classify each load-bearing assumption
as established_from_data, asserted_from_description, assumed_default, or
needs_user (only a human can answer). Put at most three sharp questions in
open_questions, each one only a human can answer and each materially
affecting the analysis.

TRANSCRIPT:
{transcript}"""


def build_mission(run: AnalysisRunState) -> str:
    spec = run.causal_spec
    treatment = spec.treatment.column if spec and spec.treatment else None
    outcome = spec.outcome.column if spec and spec.outcome else None
    profile = run.dataset_profile
    shape = (
        f"{profile.n_rows:,} rows x {profile.n_columns} columns"
        if profile
        else "shape unknown"
    )
    return (
        f"CAUSAL QUESTION: {run.causal_question}\n"
        f"PARSED TREATMENT: {treatment or 'not resolved'}\n"
        f"PARSED OUTCOME: {outcome or 'not resolved'}\n"
        f"DATASET: {run.dataset_name or 'unnamed'} ({shape})\n\n"
        "Investigate this dataset. Start with dataset_context and "
        "list_columns; drill into columns whose role is unclear or "
        "suspicious. Finish with a plain-text summary of what you learned, "
        "the role of every column, and what only the user could tell us."
    )


def render_transcript(transcript: list[dict]) -> str:
    lines: list[str] = []
    for step in transcript:
        if step.get("thought"):
            lines.append(f"THOUGHT: {step['thought']}")
        lines.append(f"TOOL {step['tool']}({step['args']})")
        lines.append(f"OBSERVED: {step['output']}")
    return "\n".join(lines)[:24000]
