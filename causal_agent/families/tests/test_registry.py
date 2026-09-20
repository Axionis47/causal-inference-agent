"""The registry is what the desk knows about a family: every declared family is there, the built ones bring a block, probes,
figures and a lane, and the core packages reach them only through it."""

from __future__ import annotations

from causal_agent.common.contracts import DESIGNS, AdjustmentDesign, DidDesign, RdDesign
from causal_agent.families import registry as R
from causal_agent.viz import graph as G

BUILT = {"adjustment": AdjustmentDesign, "diff_in_diff": DidDesign, "discontinuity": RdDesign}


def test_every_family_in_the_knowledge_is_registered_and_the_built_ones_are_whole():
    assert set(R.REGISTRY) >= {"adjustment", "diff_in_diff", "discontinuity", "synthetic_control", "interrupted_series", "instrument", "root_cause"}
    for name, cls in BUILT.items():
        f = R.family(name)
        assert f.built and f.design_cls is cls and f.design_block is not None and f.probes is not None and f.previz and f.lane is not None
        assert DESIGNS[name] is cls
    for name in ("synthetic_control", "interrupted_series", "instrument", "root_cause"):
        f = R.family(name)
        assert not f.built and f.design_cls is None and f.design_block is None and f.lane is not None


def test_the_needs_grid_lists_only_families_with_claims_and_the_figures_are_registered_with_the_viz_tool():
    needs = R.needs()
    assert "root_cause" not in needs and needs["discontinuity"].fits.get("assignment.kind") == ["cutoff_rule"]
    assert [f.decl.name for f in G.declared("discontinuity")] == ["discontinuity.density", "discontinuity.outcome_by_bin"]
    assert G.declared("root_cause") == []


def test_the_lanes_are_compiled_once_and_a_declared_family_gets_a_stub():
    lanes = R.lanes()
    assert lanes is R.lanes()
    assert "freeze_design" in lanes["adjustment"].get_graph().nodes and "shape_table" in lanes["diff_in_diff"].get_graph().nodes
    assert list(lanes["synthetic_control"].get_graph().nodes) == ["__start__", "run", "__end__"]
