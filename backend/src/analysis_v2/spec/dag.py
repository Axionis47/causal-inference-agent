"""CausalDAG: the explicit causal model the investigator builds (S2a).

The single source of truth for identification. Nodes carry whether they are
observed (a latent node is an unmeasured common cause); edges carry the
mechanism that justifies them and where that justification came from. The
adjustment set and the identifiability verdict are DERIVED from the graph by
the backdoor criterion, not authored by hand, so design eligibility and the
method plan read one computed source.

Identification is in-house over networkx (d-separation + the backdoor
criterion). No external library ever computes an estimate; this module only
reasons about graph structure.
"""
from __future__ import annotations

from enum import StrEnum

import networkx as nx
from networkx.algorithms.d_separation import is_d_separator
from pydantic import BaseModel, Field, model_validator

from .causal_spec import Confidence


class EdgeProvenance(StrEnum):
    DATA = "data"  # a dependence/independence test or temporal order in the data
    TEMPORAL = "temporal"  # measured-before ordering fixes the arrow
    DOMAIN = "domain"  # subject-matter knowledge the model supplied
    USER = "user"  # the analyst asserted it


class CausalNode(BaseModel):
    name: str = Field(min_length=1)
    observed: bool = True  # False marks a latent (unmeasured) variable


class CausalEdge(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    mechanism: str = Field(min_length=1, max_length=300)
    provenance: EdgeProvenance = EdgeProvenance.DOMAIN
    confidence: Confidence = Confidence.MEDIUM


class CausalDAG(BaseModel):
    nodes: list[CausalNode] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)
    treatment: str | None = None
    outcome: str | None = None
    estimand: str = "ate"

    @model_validator(mode="after")
    def _ensure_edge_endpoints_are_nodes(self) -> CausalDAG:
        declared = {n.name for n in self.nodes}
        for edge in self.edges:
            for endpoint in (edge.source, edge.target):
                if endpoint not in declared:
                    self.nodes.append(CausalNode(name=endpoint, observed=True))
                    declared.add(endpoint)
        return self

    # -- graph views --------------------------------------------------------

    def _graph(self) -> nx.DiGraph:
        g: nx.DiGraph = nx.DiGraph()
        for node in self.nodes:
            g.add_node(node.name, observed=node.observed)
        for edge in self.edges:
            g.add_edge(edge.source, edge.target)
        return g

    def observed_names(self) -> set[str]:
        return {n.name for n in self.nodes if n.observed}

    def is_acyclic(self) -> bool:
        return bool(nx.is_directed_acyclic_graph(self._graph()))

    # -- identification -----------------------------------------------------

    def adjustment_set(self) -> tuple[set[str] | None, str]:
        """The backdoor adjustment set of observed variables, or (None, reason)
        when the treatment effect is not identifiable by covariate adjustment.

        Candidate set: observed common ancestors of treatment and outcome that
        are not descendants of the treatment (so mediators and colliders are
        excluded). Validity is verified on the proper backdoor graph (the
        treatment's outgoing edges removed): the candidate must d-separate
        treatment from outcome. When it cannot, a backdoor path runs through an
        unobserved variable and adjustment cannot identify the effect.
        """
        if self.treatment is None or self.outcome is None:
            return None, "treatment or outcome is not set on the graph"
        g = self._graph()
        t, y = self.treatment, self.outcome
        if t not in g or y not in g:
            return None, "treatment or outcome is missing from the graph"
        if not nx.is_directed_acyclic_graph(g):
            return None, "the causal graph has a cycle"

        descendants_t = nx.descendants(g, t) | {t}
        candidate = nx.ancestors(g, t) | nx.ancestors(g, y)
        z = {
            n for n in candidate
            if n not in descendants_t and n != y and n in self.observed_names()
        }

        backdoor = g.copy()
        backdoor.remove_edges_from([(t, s) for s in list(g.successors(t))])
        if is_d_separator(backdoor, {t}, {y}, z):
            return z, "identified by backdoor adjustment"
        return (
            None,
            "an open backdoor path runs through an unobserved variable; "
            "the effect is not identifiable by covariate adjustment",
        )

    def is_identifiable(self) -> tuple[bool, str]:
        z, reason = self.adjustment_set()
        return z is not None, reason

    def testable_implications(self) -> list[tuple[str, str, frozenset[str]]]:
        """Conditional independencies the graph implies (the local Markov set):
        each node is independent of its non-descendants given its parents. Each
        is a hypothesis the data can refute; both endpoints must be observed."""
        g = self._graph()
        if not nx.is_directed_acyclic_graph(g):
            return []
        observed = self.observed_names()
        implications: list[tuple[str, str, frozenset[str]]] = []
        for node in g.nodes:
            parents = set(g.predecessors(node))
            non_descendants = (
                set(g.nodes) - nx.descendants(g, node) - {node} - parents
            )
            for other in non_descendants:
                if node < other and {node, other} <= observed and parents <= observed:
                    implications.append((node, other, frozenset(parents)))
        return implications
