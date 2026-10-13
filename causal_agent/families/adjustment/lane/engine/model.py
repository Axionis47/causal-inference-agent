"""The causal model and what the engine identifies on a graph: the part of the engine the reasoning may call. DoWhy always sees
a binary treatment column named `treated`; a factor the person says exists and the file does not hold is the node `unobserved`."""

from __future__ import annotations

import pandas as pd

from causal_agent.common.contracts import Contrast
from causal_agent.families.adjustment.lane.contracts import Estimand, Graph

TREATED = "treated"  # the 0/1 column every contrast table carries
HIDDEN = "unobserved"  # the node that stands for a factor the person says exists and the file does not hold
SEED = 7  # the draws of every falsification and sensitivity run, set once per run (see falsifications in the catalogue)


def contrast_table(table: pd.DataFrame, treatment: str, contrast: Contrast) -> pd.DataFrame:
    """Rows at the two levels only, with a 0/1 `treated` column. DoWhy always sees a binary treatment."""
    col = table[treatment].astype(str)
    sub = table[col.isin([str(contrast.control), str(contrast.treated)])].copy()
    flag = (col.loc[sub.index] == str(contrast.treated)).astype(int)
    if treatment != TREATED:  # a file whose treatment column is itself named "treated" keeps its place
        sub = sub.drop(columns=[treatment])
    sub[TREATED] = flag
    return sub


def build_model(table: pd.DataFrame, graph: Graph, outcome: str):
    from dowhy import CausalModel

    g = graph.to_networkx()
    g = _relabel(g, graph.treatment, TREATED)
    return CausalModel(data=table, treatment=TREATED, outcome=outcome, graph=g)


def identify(model) -> Estimand:
    """Every road DoWhy finds on the graph. The design takes the first open one in the order backdoor, frontdoor, iv; the
    estimator pick may take another, and the frozen design records which."""
    ide = model.identify_effect(proceed_when_unidentifiable=True)
    ests = getattr(ide, "estimands", {}) or {}
    roads = [k for k in ("backdoor", "frontdoor", "iv") if ests.get(k) is not None]
    if not roads:
        return Estimand(kind="none")
    return Estimand(
        kind=roads[0],
        roads=roads,
        adjustment_set=list(ide.get_backdoor_variables()) if "backdoor" in roads else [],
        instruments=[str(v) for v in (ide.get_instrumental_variables() or [])] if "iv" in roads else [],
        frontdoor_set=[str(v) for v in (ide.get_frontdoor_variables() or [])] if "frontdoor" in roads else [],
    )


def road_set(ide, road: str) -> set[str] | None:
    """The set DoWhy's identification gives the road an estimator runs on, or None when the road is not open on this graph."""
    if (getattr(ide, "estimands", {}) or {}).get(road) is None:
        return None
    if road == "backdoor":
        return {str(v) for v in ide.get_backdoor_variables()}
    if road == "iv":
        return {str(v) for v in (ide.get_instrumental_variables() or [])}
    return {str(v) for v in (ide.get_frontdoor_variables() or [])}


def frozen_set(frozen: Estimand, road: str) -> set[str]:
    return set({"backdoor": frozen.adjustment_set, "iv": frozen.instruments, "frontdoor": frozen.frontdoor_set}.get(road, []))


def _relabel(g, old: str, new: str):
    import networkx as nx

    return nx.relabel_nodes(g, {old: new}) if old in g else g
