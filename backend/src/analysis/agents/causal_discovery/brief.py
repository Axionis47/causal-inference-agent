"""Contract surface for the causal_discovery agent.

The discovery agent runs one or more graph-learning algorithms (PC, GES,
LiNGAM, NOTEARS) over the dataset and writes the chosen graph into
`state.discovered_dag`. Unlike eda / domain_knowledge, the agent has a
hard-coded fallback: on exception or empty results it emits a
"Simple DAG (fallback)" so `state.discovered_dag` is almost always
present after execute. The brief encodes that distinction with flags
so the orchestrator can tell a real discovery from a fallback.

Flags this agent owns (the rest of the DAG-related flags belong to
dag_expert, which fuses user/LLM/data DAGs and computes adjustment
sets):

- LOW_STABILITY -- chosen DAG is empty, has 0-1 edges, or came from the
  simple-DAG fallback (discovery_method contains "fallback").
- CYCLE_DETECTED -- the directed edges form a cycle. The discovery
  engines aim to return DAGs but legacy fallbacks (NOTEARS/LiNGAM via
  causallearn) can in practice emit cycles when the underlying matrix
  is not acyclic.

DAG_CONFLICT, NO_ADJUSTMENT_SET, and COLLIDER_SUSPECTED are intentionally
left to dag_expert so we do not have two detectors disagreeing.
"""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState
from src.analysis.agents.causal_discovery.output import CausalDAG
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "causal_discovery"

LOW_STABILITY_EDGE_MIN = 2


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "What directed graph do the data-driven discovery algorithms "
        "propose for this dataset?"
    ),
    needs=(
        "dataframe_path",
        "data_profile",
    ),
    delivers=(
        "discovered_dag",
        "agent_briefs[causal_discovery]",
    ),
    refuses_when=(
        "dataframe_path is missing (profiler has not loaded data)",
        "data_profile is missing (profiler has not run)",
    ),
    success_criteria=(
        Criterion(
            id="discovery.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['causal_discovery']"
            ),
        ),
        Criterion(
            id="discovery.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "dataframe_path or data_profile is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="discovery.flag.low_stability",
            description=(
                "raises LOW_STABILITY when the chosen DAG has fewer than "
                "2 edges or came from the simple-DAG fallback"
            ),
            raises_flag=Flag.LOW_STABILITY,
        ),
        Criterion(
            id="discovery.flag.cycle_detected",
            description=(
                "raises CYCLE_DETECTED when the directed edges form a cycle"
            ),
            raises_flag=Flag.CYCLE_DETECTED,
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse when discovery's required state slots are absent.

    No treatment / outcome check: the agent has a deterministic
    confounders-only fallback that still produces a usable DAG when
    only the profile is known.
    """
    if state.dataframe_path is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="dataframe_path missing; profiler has not loaded data",
            issues=["state.dataframe_path is None"],
        )
    if state.data_profile is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="data_profile missing; profiler has not run",
            issues=["state.data_profile is None"],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from the typed CausalDAG written to state.

    `state.discovered_dag` is None only on the early data-load failure
    path; in that case the brief is `status="failed"`. Otherwise the
    brief is "done" and the flags describe the quality of the DAG that
    landed.
    """
    dag = state.discovered_dag
    if dag is None:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="discovery produced no DAG (load or loop failure)",
            artifact_keys=[],
        )

    flags: list[Flag] = []
    issues: list[str] = []

    n_edges = len(dag.edges)
    n_nodes = len(dag.nodes)
    is_fallback = "fallback" in dag.discovery_method.lower()

    if is_fallback or n_edges < LOW_STABILITY_EDGE_MIN:
        flags.append(Flag.LOW_STABILITY)
        if is_fallback:
            issues.append(
                f"discovery_method is '{dag.discovery_method}'; "
                "no algorithm produced a usable graph"
            )
        else:
            issues.append(
                f"chosen DAG has {n_edges} edge(s) across {n_nodes} node(s) "
                f"(< {LOW_STABILITY_EDGE_MIN} edge threshold)"
            )

    if _has_directed_cycle(dag):
        flags.append(Flag.CYCLE_DETECTED)
        issues.append(
            "directed edges form a cycle; downstream identification is unsafe"
        )

    headline = (
        f"{dag.discovery_method}, "
        f"{n_nodes} node(s), {n_edges} edge(s)"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=["discovered_dag"],
    )


def _has_directed_cycle(dag: CausalDAG) -> bool:
    """DFS-based cycle check over the directed edges of the DAG.

    Undirected / bidirected edges are skipped: a cycle here means a
    directed loop, which is what blocks downstream identification.
    """
    adjacency: dict[str, list[str]] = {node: [] for node in dag.nodes}
    for edge in dag.edges:
        if edge.edge_type != "directed":
            continue
        if edge.source not in adjacency:
            adjacency[edge.source] = []
        if edge.target not in adjacency:
            adjacency[edge.target] = []
        adjacency[edge.source].append(edge.target)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in adjacency}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for nxt in adjacency.get(node, ()):
            if color.get(nxt, WHITE) == GRAY:
                return True
            if color.get(nxt, WHITE) == WHITE and visit(nxt):
                return True
        color[node] = BLACK
        return False

    return any(color[node] == WHITE and visit(node) for node in adjacency)
