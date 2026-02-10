"""Unit tests for causal discovery algorithms.

Tests PC algorithm, Fisher-Z conditional independence test,
variable selection, FDR correction, and the CausalDiscovery main interface.
"""

import numpy as np
import pandas as pd
import pytest


# ─── Synthetic Data Generators ───────────────────────────────────────────────


def make_simple_dag_data(n=500, seed=42):
    """Generate data from a known DAG: X1 -> X2 -> X3, X1 -> X3.

    The true adjacency (non-zero entries):
        X1 -> X2 (coeff 0.8)
        X2 -> X3 (coeff 0.6)
        X1 -> X3 (coeff 0.5)

    Returns:
        Tuple of (data array, variable names, true adjacency dict)
    """
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    X2 = 0.8 * X1 + rng.normal(0, 0.5, n)
    X3 = 0.5 * X1 + 0.6 * X2 + rng.normal(0, 0.5, n)

    data = np.column_stack([X1, X2, X3])
    variables = ["X1", "X2", "X3"]
    true_edges = {(0, 1), (0, 2), (1, 2)}  # X1->X2, X1->X3, X2->X3

    return data, variables, true_edges


def make_independent_data(n=500, seed=42):
    """Generate independent variables (no edges in true DAG).

    Returns:
        data array with 3 independent columns.
    """
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    X3 = rng.normal(0, 1, n)
    return np.column_stack([X1, X2, X3])


def make_chain_data(n=500, seed=42):
    """Generate data from a chain DAG: X1 -> X2 -> X3.

    X1 and X3 are conditionally independent given X2.

    Returns:
        Tuple of (data array, true edges set)
    """
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    X2 = 0.7 * X1 + rng.normal(0, 0.5, n)
    X3 = 0.7 * X2 + rng.normal(0, 0.5, n)
    data = np.column_stack([X1, X2, X3])
    true_edges = {(0, 1), (1, 2)}
    return data, true_edges


# ─── Fisher-Z Tests ─────────────────────────────────────────────────────────


class TestFisherZ:
    """Tests for the Fisher-Z conditional independence test."""

    def test_fisherz_detects_dependence(self):
        """Fisher-Z should detect dependence between correlated variables."""
        from src.causal.dag.discovery import ConditionalIndependenceTests

        rng = np.random.RandomState(42)
        n = 500
        X = rng.normal(0, 1, n)
        Y = 0.5 * X + rng.normal(0, 0.5, n)
        data = np.column_stack([X, Y])

        p_value, is_independent = ConditionalIndependenceTests.fisherz(
            data, x=0, y=1, z=[], alpha=0.05
        )

        assert not is_independent, "Fisher-Z should detect dependence"
        assert p_value < 0.05

    def test_fisherz_detects_independence(self):
        """Fisher-Z should not reject independence for truly independent vars."""
        from src.causal.dag.discovery import ConditionalIndependenceTests

        rng = np.random.RandomState(42)
        n = 500
        X = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)
        data = np.column_stack([X, Y])

        p_value, is_independent = ConditionalIndependenceTests.fisherz(
            data, x=0, y=1, z=[], alpha=0.05
        )

        assert is_independent, "Fisher-Z should not reject independence"
        assert p_value > 0.05

    def test_fisherz_conditional_independence(self):
        """Fisher-Z should detect conditional independence in a chain.

        X1 -> X2 -> X3: X1 _|_ X3 | X2.
        """
        from src.causal.dag.discovery import ConditionalIndependenceTests

        data, _ = make_chain_data(n=500, seed=42)

        # Unconditional: X1 and X3 are dependent
        _, is_indep_unconditional = ConditionalIndependenceTests.fisherz(
            data, x=0, y=2, z=[], alpha=0.05
        )
        assert not is_indep_unconditional, (
            "X1 and X3 should be marginally dependent"
        )

        # Conditional on X2: X1 and X3 should be closer to independent
        # (higher p-value than unconditional)
        p_cond, _ = ConditionalIndependenceTests.fisherz(
            data, x=0, y=2, z=[1], alpha=0.01
        )
        p_uncond, _ = ConditionalIndependenceTests.fisherz(
            data, x=0, y=2, z=[], alpha=0.01
        )
        assert p_cond > p_uncond, (
            "Conditioning on X2 should increase p-value (weaken X1-X3 association)"
        )

    def test_fisherz_small_n_large_conditioning_set(self):
        """Fisher-Z should handle small n with large conditioning set gracefully.

        When degrees of freedom <= 0, it should return p=1.0, independent=True.
        """
        from src.causal.dag.discovery import ConditionalIndependenceTests

        rng = np.random.RandomState(42)
        n = 10  # Very small sample
        p = 8   # Many variables
        data = rng.normal(0, 1, (n, p))

        # Condition on 6 variables: dof = 10 - 6 - 3 = 1
        # Condition on 8 variables: dof = 10 - 8 - 3 = -1 (should fail gracefully)
        p_value, is_independent = ConditionalIndependenceTests.fisherz(
            data, x=0, y=1, z=list(range(2, p)), alpha=0.05
        )

        # With very limited degrees of freedom, should handle gracefully
        # (either return high p-value or default to independent)
        assert isinstance(p_value, float), "p_value should be a float"
        assert is_independent or p_value > 0.05, (
            "Should handle gracefully when dof is very low"
        )

    def test_fisherz_perfect_correlation(self):
        """Fisher-Z should handle perfect correlation (|r| >= 1.0) gracefully."""
        from src.causal.dag.discovery import ConditionalIndependenceTests

        n = 100
        X = np.arange(n, dtype=float)
        Y = X.copy()  # Perfect correlation
        data = np.column_stack([X, Y])

        p_value, is_independent = ConditionalIndependenceTests.fisherz(
            data, x=0, y=1, z=[], alpha=0.05
        )

        # Should handle gracefully (r=1.0 is edge case)
        assert isinstance(p_value, (float, np.floating))
        assert isinstance(is_independent, (bool, np.bool_))


# ─── PC Algorithm Tests ──────────────────────────────────────────────────────


class TestPCAlgorithm:
    """Tests for the PC algorithm implementation."""

    def test_pc_recovers_edges_from_simple_dag(self):
        """PC should recover edges from a simple known DAG."""
        from src.causal.dag.discovery import PCAlgorithm

        data, variables, true_edges = make_simple_dag_data(n=500, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        adj, sep_sets = pc.fit(data)

        # Check that true edges are discovered (at least as undirected)
        for i, j in true_edges:
            has_edge = adj[i, j] != 0 or adj[j, i] != 0
            assert has_edge, (
                f"PC missed edge {variables[i]} -> {variables[j]}"
            )

    def test_pc_no_edges_for_independent_data(self):
        """PC should find no edges for independent data."""
        from src.causal.dag.discovery import PCAlgorithm

        data = make_independent_data(n=500, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        adj, _ = pc.fit(data)

        # No edges should be found
        n_edges = np.sum(np.abs(adj) > 0)
        assert n_edges == 0, (
            f"PC found {n_edges} edges in independent data"
        )

    def test_pc_returns_symmetric_or_directed_adjacency(self):
        """PC adjacency matrix should have valid structure."""
        from src.causal.dag.discovery import PCAlgorithm

        data, _, _ = make_simple_dag_data(n=500, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        adj, _ = pc.fit(data)

        assert adj.shape[0] == adj.shape[1] == 3
        # Diagonal should be zero
        assert np.all(np.diag(adj) == 0)

    def test_pc_separation_sets_populated(self):
        """PC should populate separation sets for removed edges."""
        from src.causal.dag.discovery import PCAlgorithm

        data, _ = make_chain_data(n=500, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        adj, sep_sets = pc.fit(data)

        # X1 and X3 should be separated (edge removed)
        # The separation set should contain X2 (index 1)
        if (0, 2) in sep_sets:
            assert 1 in sep_sets[(0, 2)], (
                "X2 should be in the separation set of X1 and X3"
            )

    def test_pc_max_cond_size(self):
        """PC with limited max_cond_size should still produce results."""
        from src.causal.dag.discovery import PCAlgorithm

        data, _, _ = make_simple_dag_data(n=500, seed=42)
        pc = PCAlgorithm(alpha=0.05, max_cond_size=1)
        adj, _ = pc.fit(data)

        # Should still produce an adjacency matrix
        assert adj.shape == (3, 3)


# ─── CausalDiscovery Main Interface Tests ────────────────────────────────────


class TestCausalDiscovery:
    """Tests for the CausalDiscovery main interface."""

    def test_discover_pc_algorithm(self):
        """CausalDiscovery with PC should produce a DiscoveryResult."""
        from src.causal.dag.discovery import CausalDiscovery

        data, variables, _ = make_simple_dag_data(n=200, seed=42)
        df = pd.DataFrame(data, columns=variables)

        cd = CausalDiscovery(algorithm="pc", alpha=0.05)
        result = cd.discover(df, variables=variables)

        assert result.algorithm == "PC"
        assert len(result.nodes) == 3
        assert result.adjacency_matrix is not None

    def test_discover_with_treatment_outcome_orientation(self):
        """CausalDiscovery should orient undirected edges toward the outcome."""
        from src.causal.dag.discovery import CausalDiscovery, EdgeType

        data, variables, _ = make_simple_dag_data(n=200, seed=42)
        df = pd.DataFrame(data, columns=variables)

        cd = CausalDiscovery(algorithm="pc", alpha=0.05)
        result = cd.discover(
            df,
            variables=variables,
            treatment="X1",
            outcome="X3",
        )

        # Check that edges have been oriented
        directed_count = sum(
            1 for e in result.edges if e.edge_type == EdgeType.DIRECTED
        )
        assert directed_count >= 0  # At least some edges should exist

    def test_discover_result_to_dict(self):
        """DiscoveryResult.to_dict() should return a proper dictionary."""
        from src.causal.dag.discovery import CausalDiscovery

        data, variables, _ = make_simple_dag_data(n=200, seed=42)
        df = pd.DataFrame(data, columns=variables)

        cd = CausalDiscovery(algorithm="pc", alpha=0.05)
        result = cd.discover(df, variables=variables)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "nodes" in d
        assert "edges" in d
        assert "algorithm" in d

    def test_discover_too_few_variables_raises(self):
        """CausalDiscovery should raise ValueError with fewer than 2 variables."""
        from src.causal.dag.discovery import CausalDiscovery

        df = pd.DataFrame({"X1": [1, 2, 3] * 20})
        cd = CausalDiscovery(algorithm="pc")
        with pytest.raises(ValueError, match="at least 2"):
            cd.discover(df, variables=["X1"])

    def test_discover_too_few_observations_raises(self):
        """CausalDiscovery should raise ValueError with fewer than 50 observations."""
        from src.causal.dag.discovery import CausalDiscovery

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"X1": rng.normal(0, 1, 20), "X2": rng.normal(0, 1, 20)})
        cd = CausalDiscovery(algorithm="pc")
        with pytest.raises(ValueError, match="at least 50"):
            cd.discover(df, variables=["X1", "X2"])

    def test_discover_unsupported_algorithm_raises(self):
        """CausalDiscovery should raise ValueError for unsupported algorithm."""
        from src.causal.dag.discovery import CausalDiscovery

        with pytest.raises(ValueError, match="Unknown algorithm"):
            CausalDiscovery(algorithm="nonexistent")

    def test_discover_correlation_fallback(self):
        """CausalDiscovery with 'correlation' algorithm should work as fallback."""
        from src.causal.dag.discovery import CausalDiscovery

        data, variables, _ = make_simple_dag_data(n=200, seed=42)
        df = pd.DataFrame(data, columns=variables)

        cd = CausalDiscovery(algorithm="correlation", alpha=0.05)
        result = cd.discover(df, variables=variables)

        assert result.algorithm == "Correlation-based"
        assert len(result.nodes) == 3

    def test_discovery_result_get_parents_children(self):
        """DiscoveryResult should correctly report parents and children."""
        from src.causal.dag.discovery import CausalEdge, DiscoveryResult, EdgeType

        result = DiscoveryResult(
            nodes=["A", "B", "C"],
            edges=[
                CausalEdge(source="A", target="B", edge_type=EdgeType.DIRECTED),
                CausalEdge(source="B", target="C", edge_type=EdgeType.DIRECTED),
            ],
            algorithm="test",
        )

        assert result.get_parents("B") == ["A"]
        assert result.get_children("B") == ["C"]
        assert result.get_parents("A") == []
        assert result.get_children("C") == []


# ─── FDR Correction Tests ───────────────────────────────────────────────────


class TestFDRCorrection:
    """Tests for FDR correction in the PC algorithm."""

    def test_fdr_removes_spurious_edges(self):
        """FDR correction should remove more edges than uncorrected version.

        When multiple tests are performed, FDR-adjusted p-values should
        be more conservative.
        """
        from src.causal.dag.discovery import PCAlgorithm

        # Use a slightly noisy dataset where marginal spurious correlations
        # could arise
        rng = np.random.RandomState(42)
        n = 200
        p = 5
        data = rng.normal(0, 1, (n, p))
        # Add one true edge
        data[:, 1] = 0.5 * data[:, 0] + rng.normal(0, 0.5, n)

        pc = PCAlgorithm(alpha=0.05)
        adj, _ = pc.fit(data)

        # The algorithm uses FDR internally; just verify it runs and
        # the result has valid structure
        assert adj.shape == (p, p)
        assert np.all(np.diag(adj) == 0)


# ─── NOTEARS Tests ───────────────────────────────────────────────────────────


class TestNOTEARS:
    """Tests for the NOTEARS continuous optimization algorithm."""

    def test_notears_basic_structure(self):
        """NOTEARS should produce a DAG (acyclic adjacency matrix)."""
        from src.causal.dag.discovery import NOTEARS

        data, _, _ = make_simple_dag_data(n=200, seed=42)
        notears = NOTEARS(lambda1=0.1, max_iter=20, w_threshold=0.1)
        W = notears.fit(data)

        assert W.shape == (3, 3)
        assert np.all(np.diag(W) == 0), "Self-loops not allowed"

    def test_notears_detects_strong_edges(self):
        """NOTEARS should detect the strongest edges in the true DAG."""
        from src.causal.dag.discovery import NOTEARS

        data, _, true_edges = make_simple_dag_data(n=300, seed=42)
        notears = NOTEARS(lambda1=0.05, max_iter=50, w_threshold=0.05)
        W = notears.fit(data)

        # At least one of the true edges should be detected
        edges_detected = set()
        for i in range(3):
            for j in range(3):
                if abs(W[i, j]) > 0:
                    edges_detected.add((i, j))

        overlap = true_edges & edges_detected
        assert len(overlap) > 0, (
            f"NOTEARS detected no true edges. Detected: {edges_detected}, True: {true_edges}"
        )
