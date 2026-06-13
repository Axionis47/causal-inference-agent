"""Assemble the causal DAG at S3, after the spec is refined.

The treatment and outcome are only resolved by design_detection's refinement
(sole-candidate promotion, the survival duration->outcome rule), so the DAG is
built here, not at the investigator: it needs the resolved roles. It is the
deterministic projection of the dossier role table, the treatment effect plus
pre-treatment confounders as common causes and mediators on the path, and its
backdoor adjustment set equals the confounder fold design_detection used to do
inline. Instruments are deliberately omitted: they belong to the IV lane, not
the adjustment set (an instrument is an ancestor of the treatment and would
otherwise be adjusted on). A richer latent-confounder model can supersede it.
"""
from __future__ import annotations

from src.analysis_v2.spec import (
    BANNED_ADJUSTMENT_ROLES,
    CausalDAG,
    CausalEdge,
    CausalNode,
    CausalSpec,
    Confidence,
    DatasetDossier,
    EdgeProvenance,
    RoleLabel,
)


def dag_from_dossier(spec: CausalSpec, dossier: DatasetDossier | None) -> CausalDAG:
    treatment = spec.treatment.column if spec.treatment else None
    outcome = spec.outcome.column if spec.outcome else None
    protected = {c for c in (treatment, outcome) if c}
    roles = dossier.roles if dossier else []
    banned = {r.column for r in roles if r.role in BANNED_ADJUSTMENT_ROLES}

    # Confounders mirror design_detection's fold: intake's candidates plus the
    # dossier's pre-treatment columns, minus banned roles and the treatment and
    # outcome themselves. So the backdoor set matches what the fold produced.
    confounders: list[str] = []
    for column in spec.candidate_confounders:
        if column not in banned and column not in protected and column not in confounders:
            confounders.append(column)
    for role in roles:
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
    for role in roles:
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

    # Suspected unmeasured common causes enter as latent nodes (observed=False),
    # opening a backdoor path that observed adjustment cannot block, so the
    # effect becomes "not identified" rather than silently treated as identified.
    latents = dossier.suspected_latent_confounders if dossier else []
    for latent in latents:
        if latent in nodes or latent in protected:
            continue
        nodes[latent] = CausalNode(name=latent, observed=False)
        if treatment:
            edges.append(CausalEdge(
                source=latent, target=treatment,
                mechanism="suspected unmeasured common cause of treatment and outcome",
                provenance=EdgeProvenance.DOMAIN, confidence=Confidence.LOW,
            ))
        if outcome:
            edges.append(CausalEdge(
                source=latent, target=outcome,
                mechanism="suspected unmeasured common cause of treatment and outcome",
                provenance=EdgeProvenance.DOMAIN, confidence=Confidence.LOW,
            ))

    return CausalDAG(
        nodes=list(nodes.values()), edges=edges, treatment=treatment, outcome=outcome
    )


def apply_dag_adjustment(spec: CausalSpec, dossier: DatasetDossier | None) -> CausalDAG:
    """Build the DAG from the refined spec and dossier, then set the spec's
    candidate_confounders to its backdoor adjustment set. The single place the
    adjustment set is decided; both design_detection and the resume re-derive
    call it so they cannot diverge. Returns the DAG for the run-state slot."""
    dag = dag_from_dossier(spec, dossier)
    spec.candidate_confounders = sorted(dag.adjustment_set())
    return dag
