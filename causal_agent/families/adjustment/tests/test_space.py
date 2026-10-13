"""The endpoint catalogue, numerical capabilities and ladder agree."""

from causal_agent.families.adjustment.lane.graph import build
from causal_agent.families.adjustment.lane.knowledge import SPEC, load_endpoint_catalogue, load_estimators
from causal_agent.lane.endpoints import verify_catalogue


def test_each_endpoint_has_an_executable_declared_trajectory():
    catalogue = load_endpoint_catalogue()
    assert len([e for e in catalogue.endpoints if e.executable]) == 12
    assert verify_catalogue(catalogue, SPEC, {e.name for e in load_estimators()}) == []
    actual = build().compile(name="test-adjustment").get_graph()
    assert set(SPEC.names) <= set(actual.nodes)
    for endpoint in catalogue.endpoints:
        if endpoint.executable:
            assert "freeze_design" in endpoint.trajectory and "analyse" in endpoint.trajectory


def test_diagnostics_do_not_claim_arbitrary_expression_estimation():
    endpoint = load_endpoint_catalogue().endpoint("identification.id_expression")
    assert not endpoint.executable and not endpoint.estimators
    assert endpoint.configuration["observed_only"] is True
