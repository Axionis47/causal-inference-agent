"""Tests for the pure helpers: covariate_universe, build_canonical_dag,
patterns_for_domain, fuse_edges, adjustment_set_from_dag."""

import types

from src.analysis.agents.base import CausalDAG, CausalEdge
from src.analysis.agents.dag_expert.helpers import (
    adjustment_set_from_dag,
    build_canonical_dag,
    covariate_universe,
    fuse_edges,
    patterns_for_domain,
)

_LALONDE = ["treat", "re78", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
_LALONDE_COVARIATES = {"age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"}


def _state(treatment, outcome, feature_names=None, feature_types=None):
    names = list(feature_names if feature_names is not None else [])
    # Default every column to numeric so type filtering is a no-op unless a test
    # supplies its own types.
    ftypes = dict(feature_types if feature_types is not None else {n: "numeric" for n in names})
    profile = types.SimpleNamespace(feature_names=names, feature_types=ftypes)
    return types.SimpleNamespace(
        job_id="t",
        treatment_variable=treatment,
        outcome_variable=outcome,
        data_profile=profile,
    )


class TestCovariateUniverse:
    def test_includes_every_pretreatment_covariate(self):
        # The whole point of the fix: all 8 LaLonde confounders survive, not 3.
        st = _state("treat", "re78", feature_names=_LALONDE)
        assert set(covariate_universe(st)) == _LALONDE_COVARIATES

    def test_excludes_treatment_and_outcome(self):
        st = _state("treat", "re78", feature_names=["treat", "re78", "age"])
        universe = covariate_universe(st)
        assert "treat" not in universe and "re78" not in universe
        assert universe == ["age"]

    def test_excludes_row_identifier_columns(self):
        names = ["Unnamed: 0", "customer_id", "id", "age", "treat", "re78"]
        st = _state("treat", "re78", feature_names=names)
        assert covariate_universe(st) == ["age"]

    def test_excludes_datetime_and_text_types(self):
        names = ["treat", "re78", "age", "signup_date", "notes"]
        ftypes = {
            "treat": "binary", "re78": "numeric", "age": "numeric",
            "signup_date": "datetime", "notes": "text",
        }
        st = _state("treat", "re78", feature_names=names, feature_types=ftypes)
        assert covariate_universe(st) == ["age"]

    def test_empty_when_no_profile(self):
        st = types.SimpleNamespace(
            job_id="t", treatment_variable="treat", outcome_variable="re78", data_profile=None
        )
        assert covariate_universe(st) == []


class TestBuildCanonicalDag:
    def test_adjustment_set_is_the_full_covariate_universe(self):
        dag = build_canonical_dag(_state("treat", "re78", feature_names=_LALONDE))
        assert set(dag.adjustment_set) == _LALONDE_COVARIATES

    def test_draws_canonical_edges_each_covariate_to_t_and_y(self):
        dag = build_canonical_dag(_state("treat", "re78", feature_names=["treat", "re78", "age", "educ"]))
        pairs = {(e.source, e.target) for e in dag.edges}
        assert ("age", "treat") in pairs and ("age", "re78") in pairs
        assert ("educ", "treat") in pairs and ("educ", "re78") in pairs
        assert ("treat", "re78") in pairs

    def test_never_includes_treatment_or_outcome_in_adjustment_set(self):
        dag = build_canonical_dag(_state("treat", "re78", feature_names=["treat", "re78", "age"]))
        assert "treat" not in dag.adjustment_set
        assert "re78" not in dag.adjustment_set

    def test_caller_supplied_exclusions_are_removed(self):
        # The Step 2 seam: a covariate the LLM justifies as a mediator/collider/
        # instrument is dropped from the set; everything else stays in.
        names = ["treat", "re78", "age", "post_treatment_earnings"]
        dag = build_canonical_dag(
            _state("treat", "re78", feature_names=names),
            exclusions=("post_treatment_earnings",),
        )
        assert set(dag.adjustment_set) == {"age"}
        assert "post_treatment_earnings" not in dag.adjustment_set


class TestPatternsForDomain:
    def test_healthcare(self):
        assert any("Risk factors" in p for p in patterns_for_domain("healthcare"))

    def test_economics(self):
        assert any("Earnings" in p for p in patterns_for_domain("economics"))

    def test_unknown_falls_back_to_default(self):
        out = patterns_for_domain(None)
        assert any("Pre-treatment" in p for p in out)


class TestFuseEdges:
    def test_domain_priority_keeps_domain_direction(self):
        """Domain says a->b, data says b->a, with domain_priority we keep a->b."""
        dag, conflicts, sources = fuse_edges(
            domain_edges=[{"source": "a", "target": "b", "confidence": "high"}],
            data_edges=[{"source": "b", "target": "a", "confidence": 0.8, "edge_type": "directed"}],
            forbidden_edges=[],
            conflict_resolution="domain_priority",
            treatment_var=None,
            outcome_var=None,
            variable_roles={},
        )
        assert len(conflicts) == 1
        assert any(e.source == "a" and e.target == "b" for e in dag.edges)
        assert not any(e.source == "b" and e.target == "a" for e in dag.edges)

    def test_consensus_only_skips_domain_keeps_data(self):
        """consensus_only drops the domain copy of a conflict but still adds the data edge.

        Documented quirk of the existing monolith: the domain loop continues when
        reverse_key is in data_edge_set, but the data loop unconditionally adds
        the data edge since the reverse pair was never processed. Net effect is
        that data wins, not that the conflict is dropped.
        """
        dag, conflicts, _ = fuse_edges(
            domain_edges=[{"source": "a", "target": "b", "confidence": "high"}],
            data_edges=[{"source": "b", "target": "a", "confidence": 0.8, "edge_type": "directed"}],
            forbidden_edges=[],
            conflict_resolution="consensus_only",
            treatment_var=None,
            outcome_var=None,
            variable_roles={},
        )
        assert len(conflicts) == 1
        assert any(e.source == "b" and e.target == "a" for e in dag.edges)
        assert not any(e.source == "a" and e.target == "b" for e in dag.edges)

    def test_forbidden_filters_both_domain_and_data(self):
        dag, _, _ = fuse_edges(
            domain_edges=[{"source": "outcome", "target": "treat", "confidence": "low"}],
            data_edges=[{"source": "outcome", "target": "treat", "confidence": 0.6, "edge_type": "directed"}],
            forbidden_edges=[("outcome", "treat", "reverse causation")],
            conflict_resolution="domain_priority",
            treatment_var="treat",
            outcome_var="outcome",
            variable_roles={},
        )
        assert dag.edges == []


class TestAdjustmentSetFromDag:
    def test_simple_backdoor(self):
        """Classic confounder pattern: age -> {treat, outcome}, treat -> outcome."""
        dag = CausalDAG(
            nodes=["age", "treat", "outcome"],
            edges=[
                CausalEdge(source="age", target="treat", edge_type="directed"),
                CausalEdge(source="age", target="outcome", edge_type="directed"),
                CausalEdge(source="treat", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        result = adjustment_set_from_dag(dag, "treat", "outcome")
        assert result["adjustment_set"] == ["age"]
        assert result["confounders"] == ["age"]
        assert result["mediators"] == []

    def test_mediator_excluded_from_adjustment(self):
        """treat -> mediator -> outcome: never adjust for the mediator."""
        dag = CausalDAG(
            nodes=["treat", "mediator", "outcome"],
            edges=[
                CausalEdge(source="treat", target="mediator", edge_type="directed"),
                CausalEdge(source="mediator", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        result = adjustment_set_from_dag(dag, "treat", "outcome")
        assert result["adjustment_set"] == []
        assert "mediator" in result["mediators"]
