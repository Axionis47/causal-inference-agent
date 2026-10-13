"""Illustrative causal motifs are validated by the installed graph identifiers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from causal_agent.families.adjustment.lane.contracts import Edge, Graph
from causal_agent.families.adjustment.lane.engine.model import identify_candidates, valid_model_adjustment
from causal_agent.families.adjustment.lane.knowledge import load_motifs


@pytest.mark.parametrize("motif", load_motifs(), ids=lambda motif: motif.id)
def test_each_declared_motif_has_the_recorded_graph_certificates(motif):
    g = Graph(
        treatment=motif.treatment,
        outcome=motif.outcome,
        nodes=motif.observed + motif.latent,
        edges=[Edge(src=src, dst=dst) for src, dst in motif.edges],
    )
    # Identification reads the graph and observed column presence, never associations.
    rng = np.random.default_rng(33)
    table = pd.DataFrame({"treated" if column == motif.treatment else column: rng.normal(size=64) for column in motif.observed})
    methods = yaml.safe_load((Path(__file__).parents[1] / "lane/knowledge/rivals.yaml").read_text())["backdoor_methods"]
    assert methods[0] == "default" and methods.count("default") == 1
    certificates, errors = identify_candidates(g, table, methods, mediator=motif.mediator)
    assert not any(error.startswith("default:") for error in errors)
    if motif.stop == "unidentified_effect":
        assert not certificates
    for expected in motif.expected:
        candidates = [certificate for certificate in certificates if certificate.kind == expected.road and certificate.target == expected.target]
        assert candidates, (motif.id, expected, certificates, errors)
        if expected.variables is not None:
            attr = {"backdoor": "adjustment_set", "general_adjustment": "adjustment_set", "iv": "instruments", "frontdoor": "frontdoor_set"}[expected.road]
            assert any(set(getattr(certificate, attr)) == set(expected.variables) for certificate in candidates), (motif.id, expected, certificates)
    for forbidden in motif.forbidden_adjustments:
        assert not valid_model_adjustment(g, forbidden), (motif.id, forbidden)
    if motif.modifiers:
        assert valid_model_adjustment(g, motif.modifiers)
    signatures = [
        (certificate.kind, certificate.target, tuple(certificate.adjustment_set), tuple(certificate.instruments), tuple(certificate.frontdoor_set))
        for certificate in certificates
    ]
    assert len(signatures) == len(set(signatures)), "default and alternative policies must deduplicate equivalent road/set certificates"


def test_requested_ten_motifs_are_distinct_accounts_not_universal_dag_classes():
    motifs = load_motifs()
    assert {motif.id for motif in motifs if motif.category == "base"} == {
        "randomized_treatment",
        "exogenous_assignment",
        "observed_common_cause",
        "measured_confounding_network",
        "outcome_predictor",
        "collider_selection_trap",
        "mediator_with_adjustment",
        "hidden_confounding_iv",
        "hidden_confounding_frontdoor",
        "multiple_identification_routes",
    }
    assert {motif.id for motif in motifs if motif.category == "variant"} == {"natural_mediation_variant", "baseline_modifier_variant"}
    assert all(motif.account and motif.evidence and motif.assumptions for motif in motifs)
