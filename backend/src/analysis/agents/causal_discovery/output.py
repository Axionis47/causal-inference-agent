"""Typed output for the causal_discovery agent.

CausalDAG is the slot this agent writes (`state.discovered_dag`) and
the slot dag_expert writes (`state.refined_dag`). The model lives here
because causal_discovery is the upstream producer; dag_expert imports
it. CausalPair tracks treatment-outcome pairs across the pipeline.
"""

from pydantic import BaseModel


class CausalPair(BaseModel):
    """A treatment-outcome pair that was analyzed."""

    treatment: str
    outcome: str
    rationale: str = ""
    priority: int = 1  # 1 = primary, 2 = secondary, 3 = exploratory


class CausalEdge(BaseModel):
    """An edge in a causal graph."""

    source: str
    target: str
    edge_type: str = "directed"  # directed, bidirected, undirected
    confidence: float = 1.0


class CausalDAG(BaseModel):
    """A causal directed acyclic graph."""

    nodes: list[str]
    edges: list[CausalEdge]
    discovery_method: str
    treatment_variable: str | None = None
    outcome_variable: str | None = None
    interpretation: str = ""  # LLM-generated interpretation of the graph
    # dag_expert outputs (pullable by downstream agents via context tools)
    forbidden_edges: list[dict[str, str]] | None = None
    variable_roles: dict[str, str] | None = None
    adjustment_set: list[str] | None = None
