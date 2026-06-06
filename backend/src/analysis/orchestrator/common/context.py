"""Shared state-summary helpers for orchestrator prompts.

Both orchestrators want the same compact view of an in-flight job:
"what's been done, what's next, what's blocking." These helpers produce
that view as terse strings that fit in a few hundred tokens regardless
of state size.

The point of this module is to keep orchestrator prompts bounded
without dumping full state. Each helper returns a one- to three-line
summary that's enough for a dispatch decision and nothing more. The
full data lives in AnalysisState; if a specialist needs detail, it
already pulls from state via the ContextTools mixin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analysis.agents.base.state import AnalysisState


# Order matters: specialists are listed roughly in pipeline order so
# the resulting summary reads as a progress bar.
_PROGRESS_FIELDS: list[tuple[str, str]] = [
    ("domain_knowledge", "domain"),
    ("data_profile", "profile"),
    ("eda_result", "eda"),
    ("discovered_dag", "discovery"),
    ("refined_dag", "dag_expert"),
    ("confounder_discovery", "confounders"),
    ("ps_diagnostics", "ps_diag"),
    ("data_repairs", "repairs"),
]


def summarize_progress(state: AnalysisState) -> str:
    """Return a short list of which pipeline stages have produced output.

    Looks like:
        "domain, profile, eda, discovery, 3 effects, 2 sensitivity, critique=ITERATE"
    """
    done: list[str] = []
    for field, label in _PROGRESS_FIELDS:
        value = getattr(state, field, None)
        # data_repairs is a list; the rest are objects or dicts that
        # default to None or empty. Treat empty list as "not done."
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, dict) and not value:
            continue
        done.append(label)

    n_effects = len(state.treatment_effects)
    if n_effects:
        done.append(f"{n_effects} effects")
    n_sens = len(state.sensitivity_results)
    if n_sens:
        done.append(f"{n_sens} sensitivity")

    latest = state.get_latest_critique()
    if latest is not None:
        done.append(f"critique={latest.decision.value}")

    if not done:
        return "no stages complete yet"
    return ", ".join(done)


def summarize_recent_dispatches(state: AnalysisState, n: int = 8) -> str:
    """Compact 'who did I just dispatch' line for the orchestrator's prompt.

    The orchestrator's iterate branch can loop on the same parallel pair
    when the critique flags an issue: "Address covariate imbalance" looks
    like a fresh problem on every prompt rebuild because the LLM has no
    memory of what it just dispatched. Surface the last `n` dispatch
    traces so the model sees its own recent behaviour and can break a
    cycle instead of repeating it.

    Returns a single line like
        "recent dispatches: eda_agent×3, confounder_discovery×3, dag_expert×1"
    or an empty string if there are no dispatch traces yet (e.g. on the
    very first decision of a job).
    """
    if not state.agent_traces:
        return ""

    recent = []
    for trace in reversed(state.agent_traces):
        action = (trace.action or "")
        if not (action.startswith("dispatch_to_") or action.startswith("parallel_dispatch_")):
            continue
        recent.append(action)
        if len(recent) >= n:
            break

    if not recent:
        return ""

    counts: dict[str, int] = {}
    for action in recent:
        if action.startswith("parallel_dispatch_"):
            for name in action[len("parallel_dispatch_"):].split("_"):
                if name and name != "agent":
                    counts[name] = counts.get(name, 0) + 1
        else:
            name = action[len("dispatch_to_"):]
            counts[name] = counts.get(name, 0) + 1

    parts = [f"{name}×{n}" for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "recent dispatches: " + ", ".join(parts)


def summarize_dispatch_focus(state: AnalysisState) -> str:
    """Return what the orchestrator should focus on next, in one or two lines.

    The orchestrator's main decision drivers are:
    - Is metadata available but domain_knowledge not yet extracted?
    - Has profiling happened?
    - Did critique say ITERATE? If so, the top issue is the focus.
    - Is the analysis approved and ready for notebook generation?

    Everything else is covered by the LLM reading the structured state
    context dict.
    """
    if state.is_approved():
        return "Critique APPROVED. Dispatch notebook_generator to finalize."

    if state.is_rejected():
        return "Critique REJECTED. Job is being failed; no further dispatch."

    latest = state.get_latest_critique()
    if latest is not None and latest.decision.value == "ITERATE":
        # Show only the first issue; the LLM can fetch more if needed
        first_issue = (latest.issues[0] if latest.issues else "address feedback")
        return (
            f"Critique says ITERATE (iter {state.iteration_count}/{state.max_iterations}). "
            f"Top issue: {first_issue[:120]}. Re-dispatch the relevant specialist."
        )

    if state.raw_metadata is not None and state.domain_knowledge is None:
        return "Metadata available; dispatch domain_knowledge first to seed downstream agents."

    if state.data_profile is None:
        return "No profile yet; dispatch data_profiler."

    if state.eda_result is None and state.discovered_dag is None:
        return "Profile done; dispatch eda_agent and causal_discovery in parallel."

    # Defence in depth: the orchestrator's loop checks the gate predicate
    # before this prompt is ever built, so reaching this branch means the
    # gate was bypassed. Telling the LLM "do nothing" is safer than letting
    # it suggest effect_estimator while the job should be parked.
    from src.analysis.orchestrator.base import should_pause_for_approval

    if should_pause_for_approval(state):
        return "Awaiting human approval — do not dispatch effect_estimator."

    if not state.treatment_effects:
        return "Discovery and EDA done; dispatch effect_estimator."

    if not state.sensitivity_results:
        return "Effects estimated; dispatch sensitivity_analyst, then request_critique."

    return "Pipeline near complete; request_critique if not done, else finalize_analysis."
