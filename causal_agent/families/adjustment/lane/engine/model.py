"""Graph-only identification certificates; no treatment-outcome estimates here."""

from __future__ import annotations

import hashlib
import json

import networkx as nx
import pandas as pd

from causal_agent.families.adjustment.lane.contracts import Estimand, Graph
from causal_agent.families.adjustment.lane.support import TREATED

SEED = 7


def build_model(table: pd.DataFrame, graph: Graph, outcome: str):
    from dowhy import CausalModel

    g = nx.relabel_nodes(graph.to_networkx(), {graph.treatment: TREATED})
    return CausalModel(data=table, treatment=TREATED, outcome=outcome, graph=g)


def fingerprint(graph: Graph) -> str:
    raw = json.dumps({"nodes": sorted(graph.nodes), "edges": sorted((e.src, e.dst) for e in graph.edges)}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def observed_set(ide, road: str) -> list[str] | None:
    if getattr(ide, "no_directed_path", False):
        return []
    if not (getattr(ide, "estimands", {}) or {}).get(road):
        return None
    if road == "backdoor":
        return sorted(ide.get_backdoor_variables() or [])
    if road == "general_adjustment":
        return sorted(ide.get_general_adjustment_variables() or [])
    if road == "frontdoor":
        return sorted(ide.get_frontdoor_variables() or [])
    return sorted(ide.get_instrumental_variables() or [])


def identify_candidates(
    graph: Graph, table: pd.DataFrame, methods: list[str], *, mediator: str | None = None, instruments: list[str] = (), controls: list[str] = ()
) -> tuple[list[Estimand], list[str]]:
    """All executable identification certificates and diagnostic exclusion reasons."""
    g = graph.to_networkx()
    if not nx.is_directed_acyclic_graph(g):
        return [], ["invalid.graph: cycle"]
    if not nx.has_path(g, graph.treatment, graph.outcome):
        return [Estimand(kind="zero", structural_zero=True, assumptions=["no directed causal path in the declared graph"])], []
    model = build_model(table, graph, graph.outcome)
    certificates: list[Estimand] = []
    errors: list[str] = []
    seen: set[tuple] = set()
    for method in methods:
        try:
            ide = model.identify_effect(method_name=method, proceed_when_unidentifiable=True)
        except Exception as ex:
            errors.append(f"{method}: {type(ex).__name__}: {ex}")
            continue
        for road in ("backdoor", "general_adjustment", "frontdoor", "iv"):
            chosen = observed_set(ide, road)
            if chosen is None:
                continue
            chosen = [graph.treatment if v == TREATED else v for v in chosen]
            stamp = road, tuple(chosen)
            if stamp in seen:
                continue
            seen.add(stamp)
            record = Estimand(kind=road, identification_method=method, symbolic=str(ide.estimands[road]))
            if road in ("backdoor", "general_adjustment"):
                record.adjustment_set = chosen
            if road == "iv":
                record.instruments = chosen
            if road == "frontdoor":
                record.frontdoor_set = chosen
                record.first_stage_set = sorted((ide.mediation_first_stage_confounders or {}).get("backdoor") or [])
                record.second_stage_set = sorted((ide.mediation_second_stage_confounders or {}).get("backdoor") or [])
            certificates.append(record)
    if instruments and conditional_iv_valid(graph, list(instruments), list(controls)):
        stamp = "iv", tuple(sorted(instruments)), tuple(sorted(controls))
        if stamp not in seen:
            certificates.append(
                Estimand(
                    kind="iv",
                    instruments=sorted(instruments),
                    adjustment_set=sorted(controls),
                    identification_method="conditional_iv_dseparation",
                    assumptions=["conditional exogeneity", "exclusion", "homogeneous additive effect"],
                )
            )
    if mediator and mediator in g:
        for target in ("nie", "nde"):
            ide = model.identify_effect(estimand_type=f"nonparametric-{target}", proceed_when_unidentifiable=True)
            selected = ide.get_mediator_variables() or []
            first = (ide.mediation_first_stage_confounders or {}).get("backdoor")
            second = (ide.mediation_second_stage_confounders or {}).get("backdoor")
            total = (ide.backdoor_variables or {}).get("backdoor")
            if selected == [mediator] and first is not None and second is not None and total is not None:
                certificates.append(
                    Estimand(
                        kind="mediation",
                        target=target,
                        mediator=mediator,
                        first_stage_set=sorted(first),
                        second_stage_set=sorted(second),
                        total_set=sorted(total),
                        identification_method="minimal-adjustment",
                        symbolic=str(ide.estimands["mediation"]),
                    )
                )
            else:
                errors.append(f"mediation.{target}: specified singleton mediator or stage/total adjustment not identified")
    if not certificates:
        errors.append("unidentified: no implemented identification certificate survives; hidden causes remain in the graph")
    return certificates, errors


def conditional_iv_valid(graph: Graph, instruments: list[str], controls: list[str]) -> bool:
    """Observed baseline controls must block every non-treatment IV-to-outcome path."""
    g = graph.to_networkx()
    if not instruments or any(v not in g for v in instruments + controls):
        return False
    if set(instruments) & set(controls):
        return False
    if any(c in nx.descendants(g, graph.treatment) or any(c in nx.descendants(g, z) for z in instruments) for c in controls):
        return False
    if any(not nx.has_path(g, z, graph.treatment) for z in instruments):
        return False
    cut = g.copy()
    cut.remove_edges_from(list(cut.out_edges(graph.treatment)))
    return nx.is_d_separator(cut, set(instruments), {graph.outcome}, set(controls))


def id_diagnostic(graph: Graph, observed: set[str]) -> dict:
    """The installed ID API does not encode latent-node observability; never certify it."""
    if set(graph.nodes) - observed:
        return {"status": "unsupported", "reason": "ID API ignores observed_nodes for this explicit-hidden-node representation"}
    from dowhy.causal_identifier.id_identifier import identify_effect_id

    expression = identify_effect_id(graph.to_networkx(), [graph.treatment], [graph.outcome])
    variables = set()

    def visit(term):
        if isinstance(term, dict):
            variables.update(term.get("outcome_vars", []))
            variables.update(term.get("condition_vars", []))
        elif term is not None:
            variables.update(term.get_val("sum"))
            for child in term.get_val("prod"):
                visit(child)

    visit(expression)
    if variables - observed:
        return {"status": "unsupported", "reason": "symbolic expression refers to unavailable variables", "variables": sorted(variables)}
    return {
        "status": "diagnostic",
        "expression": str(expression),
        "variables": sorted(variables),
        "cites": [f"col:{v}" for v in sorted(variables)],
        "reason": "symbolic expression only; no arbitrary-expression estimator is implemented",
    }


def valid_model_adjustment(graph: Graph, columns: list[str]) -> bool:
    g = graph.to_networkx()
    if set(columns) & nx.descendants(g, graph.treatment):
        return False
    cut = g.copy()
    cut.remove_edges_from(list(cut.out_edges(graph.treatment)))
    return nx.is_d_separator(cut, {graph.treatment}, {graph.outcome}, set(columns))


def stratification_support(table: pd.DataFrame, graph: Graph, certificate: Estimand, params: dict) -> dict:
    """Native propensity fit and stratum membership, with actual outcomes removed."""
    from dowhy.causal_estimators.propensity_score_stratification_estimator import PropensityScoreStratificationEstimator

    safe = table.copy(deep=True)
    safe[graph.outcome] = 0.0
    model = build_model(safe, graph, graph.outcome)
    ide = model.identify_effect(method_name=certificate.identification_method, proceed_when_unidentifiable=True)
    if observed_set(ide, "backdoor") != sorted(certificate.adjustment_set):
        raise ValueError("stratification support certificate differs from candidate")
    estimator = PropensityScoreStratificationEstimator(ide, **params)
    estimator.fit(safe[[TREATED, *certificate.adjustment_set]])
    estimator.estimate_propensity_score_column(safe)
    threshold = estimator.clipping_threshold
    strata = 0.5 * len(safe) / threshold if estimator.num_strata == "auto" else estimator.num_strata
    while True:
        labels = ((safe[estimator.propensity_score_column].rank(ascending=True) / len(safe)) * strata).round(0)
        counts = safe[TREATED].groupby(labels).agg(["sum", "count"])
        retained = counts[(counts["sum"] > threshold) & ((counts["count"] - counts["sum"]) > threshold)]
        if estimator.num_strata != "auto" or (not retained.empty and len(retained) >= 0.5 * strata):
            return {"input_rows": len(safe), "retained_rows": int(retained["count"].sum()), "strata": float(strata)}
        strata = int(strata / 2)
        if strata < 2:
            raise ValueError("stratification cannot form two supported strata")
