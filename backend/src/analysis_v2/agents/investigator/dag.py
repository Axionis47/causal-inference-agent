"""Derive a CausalDAG from the dossier role table.

The deterministic projection of what the investigator already established: the
treatment effect, pre-treatment confounders as common causes, mediators on the
treatment->outcome path, and an instrument feeding the treatment. Its backdoor
adjustment set equals the reconciled confounder list (the fold design_detection
performs today), so reading identification off the DAG is a clean swap rather
than a behaviour change. This is also the fallback when no richer model is
synthesised; a latent-confounder model, when one is built, supersedes it.
"""
from __future__ import annotations

from src.analysis_v2.spec import (
    BANNED_ADJUSTMENT_ROLES,
    CausalDAG,
    CausalEdge,
    CausalNode,
    CausalSpec,
    DatasetDossier,
    EdgeProvenance,
    RoleLabel,
)


def dag_from_dossier(dossier: DatasetDossier, spec: CausalSpec | None) -> CausalDAG:
    treatment = spec.treatment.column if spec and spec.treatment else None
    outcome = spec.outcome.column if spec and spec.outcome else None
    protected = {c for c in (treatment, outcome) if c}
    banned = {r.column for r in dossier.roles if r.role in BANNED_ADJUSTMENT_ROLES}

    # Confounders: intake's candidates and the dossier's pre-treatment columns,
    # minus anything a banned role rules out, minus the treatment/outcome. This
    # mirrors design_detection's fold so the backdoor set matches.
    confounders: list[str] = []
    for column in (spec.candidate_confounders if spec else []):
        if column not in banned and column not in protected and column not in confounders:
            confounders.append(column)
    for role in dossier.roles:
        if (
            role.role == RoleLabel.PRE_TREATMENT
            and role.column not in protected
            and role.column not in confounders
        ):
            confounders.append(role.column)

    nodes: dict[str, CausalNode] = {}

    def node(name: str | None) -> None:
        if name and name not in nodes:
            nodes[name] = CausalNode(name=name, observed=True)

    edges: list[CausalEdge] = []
    node(treatment)
    node(outcome)
    if treatment and outcome:
        edges.append(CausalEdge(
            source=treatment, target=outcome,
            mechanism="the treatment effect under study",
            provenance=EdgeProvenance.USER,
        ))
    for column in confounders:
        node(column)
        if treatment:
            edges.append(CausalEdge(
                source=column, target=treatment,
                mechanism="pre-treatment common cause of treatment and outcome",
                provenance=EdgeProvenance.TEMPORAL,
            ))
        if outcome:
            edges.append(CausalEdge(
                source=column, target=outcome,
                mechanism="pre-treatment common cause of treatment and outcome",
                provenance=EdgeProvenance.TEMPORAL,
            ))
    for role in dossier.roles:
        if role.role == RoleLabel.MEDIATOR and treatment and outcome and role.column not in protected:
            node(role.column)
            edges.append(CausalEdge(
                source=treatment, target=role.column,
                mechanism="the treatment changes the mediator",
                provenance=EdgeProvenance.DOMAIN,
            ))
            edges.append(CausalEdge(
                source=role.column, target=outcome,
                mechanism="the mediator changes the outcome",
                provenance=EdgeProvenance.DOMAIN,
            ))

    instrument = next(
        (r.column for r in dossier.roles if r.role == RoleLabel.INSTRUMENT), None
    )
    if instrument is None and spec and spec.instrument and spec.instrument.column:
        instrument = spec.instrument.column
    if instrument and treatment and instrument not in protected:
        node(instrument)
        edges.append(CausalEdge(
            source=instrument, target=treatment,
            mechanism="instrument shifts the treatment, excluded from the outcome",
            provenance=EdgeProvenance.DOMAIN,
        ))

    return CausalDAG(
        nodes=list(nodes.values()), edges=edges, treatment=treatment, outcome=outcome
    )
