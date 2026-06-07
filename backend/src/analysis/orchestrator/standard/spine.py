"""The deterministic spine: the fixed order the standard orchestrator runs.

This replaces the LLM router's per-turn choice with an explicit, dependency
ordered list of stages. Each agent keeps its own ReAct loop; only the order is
fixed here, so the spine is a "reasonable DAG" rendered as a sequence with one
parallel group. The order respects every agent's declared needs (you cannot run
EDA before the profiler, or estimation before the DAG).

A Stage names one or more agents (run in parallel when there is more than one)
and may carry a skip predicate. Skipping is how an optional agent is left out
when the data does not call for it. Two rules keep skipping safe alongside the
readiness gate (which halts the run on a refused brief):

  - An optional agent is skipped when its own preflight would refuse, so it is
    never dispatched only to refuse and halt the run (e.g. domain_knowledge with
    no metadata).
  - Required agents have no skip predicate. In normal flow their needs are met
    by the time they run, so a refusal there is a real error and should halt.

`produces` lists the state attributes that exist once a stage has run. The
runner uses it to skip already-completed stages when a job resumes after the
data-review gate, so resume does not re-run the profiler.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.analysis.agents.base.state import AnalysisState
from src.analysis.agents.domain_knowledge.brief import preflight as _domain_preflight
from src.analysis.agents.ps_diagnostics.brief import preflight as _ps_preflight
from src.domain.briefs import Flag

# Estimation methods for which propensity-score overlap diagnostics are
# meaningful. ps_diagnostics is skipped when none of the estimated effects used
# one of these, since overlap/balance checks do not apply to e.g. plain OLS.
_PS_METHOD_MARKERS: tuple[str, ...] = (
    "psm",
    "ipw",
    "aipw",
    "matching",
    "propensity",
    "doubly",
)


@dataclass(frozen=True)
class Stage:
    """One step of the spine: the agents to run and when to skip them."""

    name: str
    agents: tuple[str, ...]
    produces: tuple[str, ...]
    skip_when: Callable[[AnalysisState], bool] | None = None

    @property
    def parallel(self) -> bool:
        return len(self.agents) > 1

    def already_done(self, state: AnalysisState) -> bool:
        """True once every output this stage produces is present (resume case)."""
        return all(bool(getattr(state, attr, None)) for attr in self.produces)


def _has_flag(state: AnalysisState, agent: str, flag: Flag) -> bool:
    brief = state.agent_briefs.get(agent)
    return bool(brief and flag in brief.flags)


def _skip_domain_knowledge(state: AnalysisState) -> bool:
    """Skip when there is no metadata to read (domain_knowledge would refuse)."""
    return _domain_preflight(state) is not None


def _skip_data_repair(state: AnalysisState) -> bool:
    """Run repair only when the profiler flagged high missingness."""
    return not _has_flag(state, "data_profiler", Flag.HIGH_MISSINGNESS)


def _skip_confounder_discovery(state: AnalysisState) -> bool:
    """Run confounder search only when dag_expert produced no adjustment set."""
    return not _has_flag(state, "dag_expert", Flag.NO_ADJUSTMENT_SET)


def _skip_ps_diagnostics(state: AnalysisState) -> bool:
    """Skip when ps_diagnostics would refuse, or no PS-based method was used."""
    if _ps_preflight(state) is not None:
        return True
    methods = " ".join((te.method or "").lower() for te in state.treatment_effects)
    return not any(marker in methods for marker in _PS_METHOD_MARKERS)


# The spine. The data-review gate fires after `profile`; the runner owns that,
# not a Stage. Later human gates (DAG after `refine_dag`, results after
# `sensitivity`) will plug in the same way.
SPINE: tuple[Stage, ...] = (
    Stage(
        name="domain_knowledge",
        agents=("domain_knowledge",),
        produces=("domain_knowledge",),
        skip_when=_skip_domain_knowledge,
    ),
    Stage(
        name="profile",
        agents=("data_profiler",),
        produces=("data_profile",),
    ),
    Stage(
        name="repair",
        agents=("data_repair",),
        produces=("data_repairs",),
        skip_when=_skip_data_repair,
    ),
    Stage(
        name="explore",
        agents=("eda_agent", "causal_discovery"),
        produces=("eda_result", "discovered_dag"),
    ),
    Stage(
        name="refine_dag",
        agents=("dag_expert",),
        produces=("refined_dag",),
    ),
    Stage(
        name="confounders",
        agents=("confounder_discovery",),
        produces=("confounder_discovery",),
        skip_when=_skip_confounder_discovery,
    ),
    Stage(
        name="estimate",
        agents=("effect_estimator",),
        produces=("treatment_effects",),
    ),
    Stage(
        name="ps_diagnostics",
        agents=("ps_diagnostics",),
        produces=("ps_diagnostics",),
        skip_when=_skip_ps_diagnostics,
    ),
    Stage(
        name="sensitivity",
        agents=("sensitivity_analyst",),
        produces=("sensitivity_results",),
    ),
)
