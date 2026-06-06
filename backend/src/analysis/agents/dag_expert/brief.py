"""Contract surface for the dag_expert agent.

dag_expert fuses three DAG sources (user-supplied edges, the LLM's
domain-informed edges, and causal_discovery's data-driven edges) into
a single refined_dag with variable_roles and adjustment_set
populated. The brief reads from `state.refined_dag` and owns the
DAG-fusion-specific flags.

Flags from the closed enum:
- NO_ADJUSTMENT_SET: refined_dag.adjustment_set is empty/None
- DAG_CONFLICT: forbidden_edges are present (the fusion rejected
  edges from one of the sources)
- COLLIDER_SUSPECTED: variable_roles contains potential_collider
- CYCLE_DETECTED: refined_dag has a directed cycle (downstream
  identification is unsafe)
"""
from __future__ import annotations

from src.analysis.agents.base import AnalysisState, CausalDAG
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "dag_expert"


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "What DAG, variable roles, and adjustment set should downstream "
        "estimation use, given user, domain-knowledge, and data-driven inputs?"
    ),
    needs=("dataset_info", "discovered_dag"),
    delivers=(
        "refined_dag",
        "agent_briefs[dag_expert]",
    ),
    refuses_when=(
        "discovered_dag is missing (causal_discovery has not run)",
    ),
    success_criteria=(
        Criterion(
            id="dag_expert.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['dag_expert']"
            ),
        ),
        Criterion(
            id="dag_expert.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "discovered_dag is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="dag_expert.flag.no_adjustment_set",
            description=(
                "raises NO_ADJUSTMENT_SET when refined_dag.adjustment_set "
                "is empty or None"
            ),
            raises_flag=Flag.NO_ADJUSTMENT_SET,
        ),
        Criterion(
            id="dag_expert.flag.dag_conflict",
            description=(
                "raises DAG_CONFLICT when refined_dag.forbidden_edges "
                "is non-empty"
            ),
            raises_flag=Flag.DAG_CONFLICT,
        ),
        Criterion(
            id="dag_expert.flag.collider_suspected",
            description=(
                "raises COLLIDER_SUSPECTED when variable_roles contains a "
                "potential_collider entry"
            ),
            raises_flag=Flag.COLLIDER_SUSPECTED,
        ),
        Criterion(
            id="dag_expert.flag.cycle_detected",
            description=(
                "raises CYCLE_DETECTED when refined_dag has a directed cycle"
            ),
            raises_flag=Flag.CYCLE_DETECTED,
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse when there is no discovery DAG to refine."""
    if state.discovered_dag is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="discovered_dag missing; causal_discovery has not run",
            issues=["state.discovered_dag is None"],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from state.refined_dag (typed CausalDAG)."""
    dag = state.refined_dag
    if dag is None:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="dag_expert produced no refined_dag",
            artifact_keys=[],
        )

    flags: list[Flag] = []
    issues: list[str] = []

    adjustment_set = dag.adjustment_set or []
    if not adjustment_set:
        flags.append(Flag.NO_ADJUSTMENT_SET)
        issues.append(
            "no backdoor adjustment set; downstream estimation must "
            "fall back to the profiler's potential_confounders"
        )

    if dag.forbidden_edges:
        flags.append(Flag.DAG_CONFLICT)
        issues.append(
            f"{len(dag.forbidden_edges)} edge(s) were rejected during fusion"
        )

    roles = dag.variable_roles or {}
    if any(role == "potential_collider" for role in roles.values()):
        flags.append(Flag.COLLIDER_SUSPECTED)
        colliders = [v for v, r in roles.items() if r == "potential_collider"]
        issues.append(
            f"{len(colliders)} variable(s) marked potential_collider: "
            f"{', '.join(colliders[:3])}"
        )

    if _has_directed_cycle(dag):
        flags.append(Flag.CYCLE_DETECTED)
        issues.append(
            "refined_dag contains a directed cycle; identification is unsafe"
        )

    headline = (
        f"{len(dag.nodes)} node(s), {len(dag.edges)} edge(s), "
        f"adjustment set of {len(adjustment_set)}"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=["refined_dag"],
    )


def _has_directed_cycle(dag: CausalDAG) -> bool:
    """DFS cycle check over directed edges only."""
    adjacency: dict[str, list[str]] = {node: [] for node in dag.nodes}
    for edge in dag.edges:
        if edge.edge_type != "directed":
            continue
        adjacency.setdefault(edge.source, []).append(edge.target)
        adjacency.setdefault(edge.target, [])

    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(adjacency, WHITE)

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
