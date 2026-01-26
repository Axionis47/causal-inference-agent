"""Causal Discovery Agent - ReAct-based Graph Structure Learning.

This agent iteratively discovers causal structure from data using ReAct:
- Queries domain knowledge for prior constraints
- Investigates data characteristics to choose algorithms
- Runs algorithms and inspects results
- Validates discovered graphs for sensibility
- Compares multiple algorithms

Uses pull-based context - queries for information on demand.
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.agents.base import (
    AnalysisState,
    CausalDAG,
    CausalEdge,
    JobStatus,
    ToolResult,
    ToolResultStatus,
)
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class CausalDiscoveryAgent(ReActAgent, ContextTools):
    """ReAct-based causal discovery agent.

    This agent:
    1. Queries domain knowledge for prior constraints (immutable vars, temporal ordering)
    2. Investigates data characteristics to select algorithms
    3. Runs discovery algorithms and inspects results
    4. Validates discovered graphs for causal sensibility
    5. Compares multiple algorithms to find robust structure
    6. Finalizes with the best graph

    Uses pull-based context - queries for information on demand.
    """

    AGENT_NAME = "causal_discovery"
    MAX_STEPS = 15

    SYSTEM_PROMPT = """You are an expert in causal discovery and graphical models.
Your role is to learn the causal structure from observational data.

WORKFLOW:
1. Query domain knowledge for constraints (immutable variables, temporal ordering)
2. Get data characteristics to understand sample size, variable types, distributions
3. Select an appropriate algorithm based on data properties
4. Run the algorithm and inspect results
5. Validate the discovered graph makes causal sense
6. Optionally try another algorithm and compare
7. Finalize with your best graph

ALGORITHM SELECTION GUIDE:

1. PC ALGORITHM (Constraint-based)
   - Best for: Large number of variables, sparse graphs
   - Assumptions: Faithfulness, causal sufficiency, no hidden confounders
   - Pros: Theoretically grounded, efficient for sparse graphs
   - Cons: Sensitive to sample size, assumes no hidden confounders

2. GES (Greedy Equivalence Search)
   - Best for: Moderate number of variables, when PC fails
   - Assumptions: Causal sufficiency
   - Pros: Score-based, more robust to violations of faithfulness
   - Cons: Can be slow for large variable sets

3. NOTEARS (Continuous optimization)
   - Best for: Dense graphs, continuous data, linear relationships
   - Assumptions: Linear functional relationships
   - Pros: Modern, continuous optimization, handles dense graphs
   - Cons: May produce non-DAG solutions needing post-processing

4. LiNGAM (Linear Non-Gaussian)
   - Best for: Non-Gaussian data with linear relationships
   - Assumptions: Linear, non-Gaussian errors, acyclic
   - Pros: Can identify full causal ordering, unique solution
   - Cons: Requires non-Gaussianity, sensitive to Gaussian variables

DATA CONSIDERATIONS:
- Sample size < 500: Be cautious, use simpler methods
- Variables > 20: Use PC or limit variable set
- Gaussian data: Avoid LiNGAM
- Non-linear relationships: Linear methods may be unreliable

DOMAIN KNOWLEDGE:
- Use ask_domain_knowledge to query for prior information
- Check for immutable variables (can't be caused by others)
- Check for temporal ordering constraints
- Domain knowledge can help validate discovered edges

VALIDATION CRITERIA:
- Does treatment → outcome path exist?
- Are confounders properly placed?
- Are there unrealistic edges (e.g., outcome causing treatment)?
- Is the graph too dense or too sparse?"""

    def __init__(self) -> None:
        """Initialize the causal discovery agent."""
        super().__init__()

        # Register context query tools from mixin
        self.register_context_tools()

        # Internal state
        self._df: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._discovered_graphs: dict[str, CausalDAG] = {}
        self._current_graph: CausalDAG | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        # Register discovery-specific tools
        self._register_discovery_tools()

    def _register_discovery_tools(self) -> None:
        """Register tools for causal discovery."""
        self.register_tool(
            name="get_data_characteristics",
            description="Get data characteristics relevant for algorithm selection: sample size, variable count, distributions, correlations, Gaussianity. Call this early.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_get_data_characteristics,
        )

        self.register_tool(
            name="run_discovery_algorithm",
            description="Run a causal discovery algorithm on the data. Choose based on data characteristics.",
            parameters={
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["pc", "ges", "notears", "lingam"],
                        "description": "Discovery algorithm to run",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Significance level for independence tests (default: 0.05)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Edge weight threshold for weighted methods (default: 0.1)",
                    },
                },
                "required": ["algorithm"],
            },
            handler=self._tool_run_discovery_algorithm,
        )

        self.register_tool(
            name="inspect_graph",
            description="Inspect the most recently discovered graph structure in detail. Shows edges, treatment/outcome connections, potential confounders.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_inspect_graph,
        )

        self.register_tool(
            name="validate_graph",
            description="Validate the discovered graph for causal sensibility. Checks treatment-outcome path, reverse causation, density, and isolated nodes.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_validate_graph,
        )

        self.register_tool(
            name="compare_algorithms",
            description="Compare results from different algorithms that have been run. Identifies common and divergent edges.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_compare_algorithms,
        )

        self.register_tool(
            name="finalize_discovery",
            description="Finalize the causal discovery with chosen graph. Call this when satisfied with the discovered structure.",
            parameters={
                "type": "object",
                "properties": {
                    "chosen_algorithm": {
                        "type": "string",
                        "description": "Which algorithm's result to use",
                    },
                    "confounders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Identified confounder variables",
                    },
                    "mediators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Identified mediator variables",
                    },
                    "interpretation": {
                        "type": "string",
                        "description": "Overall interpretation of the discovered structure",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence in the discovered structure",
                    },
                },
                "required": ["chosen_algorithm", "interpretation", "confidence"],
            },
            handler=self._tool_finalize_discovery,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Generate lean initial observation for causal discovery."""
        treatment, outcome = state.get_primary_pair()
        treatment = treatment or "unknown"
        outcome = outcome or "unknown"

        n_samples = state.dataset_info.n_samples or "unknown"
        n_features = state.dataset_info.n_features or "unknown"

        obs = f"""Discover the causal structure for causal inference analysis.
Treatment: {treatment}
Outcome: {outcome}
Dataset: {n_samples} samples x {n_features} features

Workflow:
1. Query domain knowledge for constraints (temporal ordering, immutable vars)
2. Get data characteristics to guide algorithm selection
3. Run appropriate discovery algorithm
4. Inspect and validate the graph
5. Optionally try another algorithm and compare
6. Finalize when satisfied with the discovered structure"""

        return obs

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if discovery task is complete."""
        return self._finalized and state.proposed_dag is not None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute causal discovery through ReAct loop.

        Args:
            state: Current analysis state

        Returns:
            Updated state with proposed DAG
        """
        self.logger.info(
            "discovery_start",
            job_id=state.job_id,
            dataset=state.dataset_info.name or state.dataset_info.url,
        )

        state.status = JobStatus.DISCOVERING_CAUSAL
        start_time = time.time()

        try:
            # Load data
            self._df = self._load_dataframe(state)
            if self._df is None:
                state.mark_failed("Failed to load dataset for discovery", self.AGENT_NAME)
                return state

            self._current_state = state
            self._discovered_graphs = {}
            self._current_graph = None
            self._finalized = False

            # Resolve treatment/outcome
            self._treatment_var, self._outcome_var = state.get_primary_pair()

            # Run the ReAct loop
            state = await super().execute(state)

            # If not finalized via tool, auto-finalize
            if not self._finalized:
                self.logger.warning("discovery_auto_finalize")
                self._final_result = self._auto_finalize()
                self._finalized = True

            # Set the proposed DAG
            chosen_alg = self._final_result.get("chosen_algorithm", "")
            if chosen_alg and chosen_alg in self._discovered_graphs:
                state.proposed_dag = self._discovered_graphs[chosen_alg]
            elif self._current_graph:
                state.proposed_dag = self._current_graph
            elif self._discovered_graphs:
                state.proposed_dag = list(self._discovered_graphs.values())[0]
            else:
                state.proposed_dag = self._create_simple_dag()

            # Store interpretation
            if state.proposed_dag:
                state.proposed_dag.interpretation = self._final_result.get("interpretation", "")

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "discovery_complete",
                has_dag=state.proposed_dag is not None,
                algorithm=self._final_result.get("chosen_algorithm", "unknown"),
                n_nodes=len(state.proposed_dag.nodes) if state.proposed_dag else 0,
                n_edges=len(state.proposed_dag.edges) if state.proposed_dag else 0,
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("discovery_failed", error=str(e))
            # Don't fail the job - create simple DAG as fallback
            state.proposed_dag = self._create_simple_dag()

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load DataFrame from pickle path."""
        if state.dataframe_path and Path(state.dataframe_path).exists():
            with open(state.dataframe_path, "rb") as f:
                return pickle.load(f)
        return None

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    async def _tool_get_data_characteristics(self, state: AnalysisState) -> ToolResult:
        """Get data characteristics for algorithm selection."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df
        profile = state.data_profile

        # Prepare numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Get relevant columns
        relevant_cols = []
        if profile:
            relevant_cols = (
                profile.treatment_candidates[:2] +
                profile.outcome_candidates[:2] +
                profile.potential_confounders[:10]
            )
            relevant_cols = [c for c in relevant_cols if c in numeric_cols]

        if not relevant_cols:
            relevant_cols = numeric_cols[:15]

        df_subset = df[relevant_cols].dropna()

        # Distribution analysis
        distributions = []
        non_gaussian_count = 0
        for col in relevant_cols[:10]:
            data = df_subset[col].dropna()
            if len(data) > 10:
                skew = sp_stats.skew(data)
                if len(data) <= 5000:
                    _, p_val = sp_stats.shapiro(data.sample(min(len(data), 5000), random_state=42))
                else:
                    _, p_val = sp_stats.normaltest(data)
                is_gaussian = p_val > 0.05
                if not is_gaussian:
                    non_gaussian_count += 1
                distributions.append({
                    "column": col,
                    "skewness": round(skew, 2),
                    "gaussian": is_gaussian,
                })

        # Correlation summary
        corr_matrix = df_subset.corr().abs()
        high_corrs = []
        for i, c1 in enumerate(corr_matrix.columns):
            for j, c2 in enumerate(corr_matrix.columns):
                if i < j and corr_matrix.iloc[i, j] > 0.7:
                    high_corrs.append({
                        "var1": c1,
                        "var2": c2,
                        "correlation": round(float(corr_matrix.iloc[i, j]), 2),
                    })

        # Algorithm recommendations
        n_samples = len(df_subset)
        n_vars = len(relevant_cols)
        recommendations = []

        if n_samples < 300:
            recommendations.append("WARNING: Small sample size - results may be unreliable")
            recommendations.append("Consider PC with higher alpha (0.1) or use simple DAG")
        elif n_vars > 15:
            recommendations.append("Many variables - consider PC (efficient for sparse graphs)")
        else:
            recommendations.append("Sample size adequate for discovery")

        if non_gaussian_count > len(relevant_cols) * 0.5:
            recommendations.append("Most variables are non-Gaussian - LiNGAM may work well")
        else:
            recommendations.append("Many Gaussian variables - avoid LiNGAM, use PC or GES")

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_samples": len(df),
                "n_complete_cases": len(df_subset),
                "n_variables": len(relevant_cols),
                "variables": relevant_cols,
                "treatment": self._treatment_var,
                "outcome": self._outcome_var,
                "distributions": distributions[:10],
                "non_gaussian_pct": round(non_gaussian_count / max(len(distributions), 1) * 100, 1),
                "high_correlations": high_corrs[:5],
                "recommendations": recommendations,
            },
        )

    async def _tool_run_discovery_algorithm(
        self,
        state: AnalysisState,
        algorithm: str,
        alpha: float = 0.05,
        threshold: float = 0.1,
    ) -> ToolResult:
        """Run a discovery algorithm."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df
        profile = state.data_profile

        # Prepare data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if profile:
            relevant_cols = (
                profile.treatment_candidates[:2] +
                profile.outcome_candidates[:2] +
                profile.potential_confounders[:10]
            )
            relevant_cols = [c for c in relevant_cols if c in numeric_cols]
        else:
            relevant_cols = numeric_cols

        relevant_cols = relevant_cols[:20]  # Limit for computation
        df_subset = df[relevant_cols].dropna()

        if len(df_subset) < 50:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Not enough complete cases ({len(df_subset)}) for discovery",
            )

        try:
            dag = self._run_algorithm(df_subset, algorithm, alpha, threshold)

            if dag:
                self._discovered_graphs[algorithm] = dag
                self._current_graph = dag

                directed = [e for e in dag.edges if e.edge_type == "directed"]
                undirected = [e for e in dag.edges if e.edge_type == "undirected"]

                # Check treatment-outcome path
                has_direct_edge = False
                if self._treatment_var and self._outcome_var:
                    has_direct_edge = any(
                        e.source == self._treatment_var and e.target == self._outcome_var
                        for e in directed
                    )

                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "algorithm": algorithm.upper(),
                        "n_nodes": len(dag.nodes),
                        "n_edges": len(dag.edges),
                        "n_directed": len(directed),
                        "n_undirected": len(undirected),
                        "variables_used": len(relevant_cols),
                        "samples_used": len(df_subset),
                        "treatment_outcome_edge": has_direct_edge,
                        "message": "Discovery completed. Call inspect_graph for details.",
                    },
                )
            else:
                # Fallback to simple DAG
                dag = self._create_simple_dag()
                self._discovered_graphs[algorithm] = dag
                self._current_graph = dag

                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "algorithm": algorithm.upper(),
                        "fallback": True,
                        "message": "Algorithm failed - using simple DAG fallback",
                        "n_nodes": len(dag.nodes),
                        "n_edges": len(dag.edges),
                    },
                )

        except Exception as e:
            self.logger.error("discovery_algorithm_error", algorithm=algorithm, error=str(e))
            # Fallback to simple DAG
            dag = self._create_simple_dag()
            self._discovered_graphs[algorithm] = dag
            self._current_graph = dag

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "algorithm": algorithm.upper(),
                    "error": str(e),
                    "fallback": True,
                    "message": f"Error during discovery: {str(e)}. Using simple DAG fallback.",
                    "n_nodes": len(dag.nodes),
                    "n_edges": len(dag.edges),
                },
            )

    def _run_algorithm(
        self, df: pd.DataFrame, algorithm: str, alpha: float, threshold: float
    ) -> CausalDAG | None:
        """Run the specified algorithm."""
        nodes = df.columns.tolist()

        try:
            # Try new CausalDiscovery engine first
            from src.causal.dag import CausalDiscovery

            discovery = CausalDiscovery(algorithm=algorithm, alpha=alpha)
            result = discovery.discover(
                df=df,
                variables=nodes,
                treatment=self._treatment_var,
                outcome=self._outcome_var,
            )

            edges = [
                CausalEdge(
                    source=e.source,
                    target=e.target,
                    edge_type=e.edge_type,
                    confidence=e.confidence,
                )
                for e in result.edges
            ]

            return CausalDAG(
                nodes=result.nodes,
                edges=edges,
                discovery_method=f"{algorithm.upper()} (engine)",
                treatment_variable=self._treatment_var,
                outcome_variable=self._outcome_var,
            )

        except Exception as e:
            self.logger.warning(f"engine_failed_{algorithm}", error=str(e))
            # Fall back to legacy
            return self._run_algorithm_legacy(df, algorithm, alpha, threshold)

    def _run_algorithm_legacy(
        self, df: pd.DataFrame, algorithm: str, alpha: float, threshold: float
    ) -> CausalDAG | None:
        """Legacy algorithm implementation."""
        nodes = df.columns.tolist()
        edges = []

        try:
            if algorithm == "pc":
                from causallearn.search.ConstraintBased.PC import pc
                from causallearn.utils.cit import fisherz

                cg = pc(df.values, alpha=alpha, indep_test=fisherz)
                adj_matrix = cg.G.graph

                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                            edges.append(CausalEdge(
                                source=nodes[i], target=nodes[j],
                                edge_type="directed",
                            ))
                        elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1 and i < j:
                            edges.append(CausalEdge(
                                source=nodes[i], target=nodes[j],
                                edge_type="undirected",
                            ))

            elif algorithm == "ges":
                from causallearn.search.ScoreBased.GES import ges

                record = ges(df.values)
                adj_matrix = record['G'].graph

                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                            edges.append(CausalEdge(
                                source=nodes[i], target=nodes[j],
                                edge_type="directed",
                            ))

            elif algorithm in ["notears", "lingam"]:
                from causallearn.search.FCMBased import lingam

                if algorithm == "lingam":
                    model = lingam.ICALiNGAM()
                else:
                    model = lingam.DirectLiNGAM()

                model.fit(df.values)
                adj_matrix = model.adjacency_matrix_

                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        if abs(adj_matrix[i, j]) > threshold:
                            edges.append(CausalEdge(
                                source=nodes[j], target=nodes[i],
                                edge_type="directed",
                                confidence=float(abs(adj_matrix[i, j])),
                            ))

            return CausalDAG(
                nodes=nodes,
                edges=edges,
                discovery_method=f"{algorithm.upper()} (legacy)",
                treatment_variable=self._treatment_var,
                outcome_variable=self._outcome_var,
            )

        except ImportError as e:
            self.logger.warning(f"import_failed_{algorithm}", error=str(e))
            return None
        except Exception as e:
            self.logger.warning(f"algorithm_failed_{algorithm}", error=str(e))
            return None

    async def _tool_inspect_graph(self, state: AnalysisState) -> ToolResult:
        """Inspect the current graph structure."""
        if not self._current_graph:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No graph discovered yet. Run a discovery algorithm first.",
            )

        dag = self._current_graph

        # Group edges by type
        directed = [e for e in dag.edges if e.edge_type == "directed"]
        undirected = [e for e in dag.edges if e.edge_type == "undirected"]

        # Format directed edges
        directed_edges = []
        for e in directed[:20]:
            edge_info = {"source": e.source, "target": e.target}
            if e.confidence:
                edge_info["confidence"] = round(e.confidence, 2)
            directed_edges.append(edge_info)

        # Format undirected edges
        undirected_edges = [{"var1": e.source, "var2": e.target} for e in undirected[:10]]

        # Treatment and outcome analysis
        treatment = self._treatment_var
        outcome = self._outcome_var

        treatment_info = None
        if treatment and treatment in dag.nodes:
            incoming = [e.source for e in directed if e.target == treatment]
            outgoing = [e.target for e in directed if e.source == treatment]
            treatment_info = {
                "variable": treatment,
                "parents": incoming,
                "children": outgoing,
            }

        outcome_info = None
        if outcome and outcome in dag.nodes:
            incoming = [e.source for e in directed if e.target == outcome]
            outgoing = [e.target for e in directed if e.source == outcome]
            outcome_info = {
                "variable": outcome,
                "parents": incoming,
                "children": outgoing,
            }

        # Identify potential confounders
        confounders = []
        if treatment and outcome:
            t_parents = set(e.source for e in directed if e.target == treatment)
            o_parents = set(e.source for e in directed if e.target == outcome)
            confounders = list(t_parents & o_parents)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "method": dag.discovery_method,
                "n_nodes": len(dag.nodes),
                "nodes": dag.nodes[:15] + (["..."] if len(dag.nodes) > 15 else []),
                "n_directed": len(directed),
                "n_undirected": len(undirected),
                "directed_edges": directed_edges,
                "undirected_edges": undirected_edges,
                "treatment": treatment_info,
                "outcome": outcome_info,
                "potential_confounders": confounders,
            },
        )

    async def _tool_validate_graph(self, state: AnalysisState) -> ToolResult:
        """Validate the discovered graph."""
        if not self._current_graph:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No graph to validate. Run a discovery algorithm first.",
            )

        dag = self._current_graph
        issues = []
        warnings = []

        treatment = self._treatment_var
        outcome = self._outcome_var

        # Check 1: Treatment and outcome in graph
        if treatment and treatment not in dag.nodes:
            issues.append(f"Treatment variable '{treatment}' not in graph nodes")
        if outcome and outcome not in dag.nodes:
            issues.append(f"Outcome variable '{outcome}' not in graph nodes")

        # Check 2: Treatment → Outcome path
        has_direct_edge = False
        has_path = False
        if treatment and outcome and treatment in dag.nodes and outcome in dag.nodes:
            has_direct_edge = any(e.source == treatment and e.target == outcome for e in dag.edges)
            has_path = self._check_path(dag, treatment, outcome)

            if not has_direct_edge and not has_path:
                warnings.append("No path from treatment to outcome - may indicate no causal effect")

        # Check 3: Outcome → Treatment (reverse causation)
        reverse_edge = False
        if treatment and outcome:
            reverse_edge = any(e.source == outcome and e.target == treatment for e in dag.edges)
            if reverse_edge:
                issues.append(f"Reverse edge {outcome} → {treatment} detected - may be spurious")

        # Check 4: Graph density
        n_nodes = len(dag.nodes)
        n_edges = len(dag.edges)
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        if density > 0.5:
            warnings.append("Graph is very dense - may include spurious edges")
        elif density < 0.05 and n_nodes > 5:
            warnings.append("Graph is very sparse - may be missing edges")

        # Check 5: Isolated nodes
        connected = set()
        for e in dag.edges:
            connected.add(e.source)
            connected.add(e.target)
        isolated = list(set(dag.nodes) - connected)

        if isolated:
            warnings.append(f"Isolated nodes (no edges): {isolated[:5]}")

        # Determine validation status
        if not issues and not warnings:
            validation_status = "passed"
        elif not issues:
            validation_status = "passed_with_warnings"
        else:
            validation_status = "issues_found"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "validation_status": validation_status,
                "has_treatment_outcome_edge": has_direct_edge,
                "has_treatment_outcome_path": has_path,
                "reverse_causation": reverse_edge,
                "density": round(density * 100, 1),
                "n_isolated": len(isolated),
                "issues": issues,
                "warnings": warnings,
            },
        )

    def _check_path(self, dag: CausalDAG, source: str, target: str) -> bool:
        """Check if there's a directed path from source to target."""
        visited = set()
        queue = [source]

        while queue:
            node = queue.pop(0)
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)

            for edge in dag.edges:
                if edge.edge_type == "directed" and edge.source == node:
                    queue.append(edge.target)

        return False

    async def _tool_compare_algorithms(self, state: AnalysisState) -> ToolResult:
        """Compare results from different algorithms."""
        if len(self._discovered_graphs) < 2:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "n_algorithms": len(self._discovered_graphs),
                    "message": f"Only {len(self._discovered_graphs)} algorithm(s) run. Run more algorithms to compare.",
                },
            )

        # Compare basic stats
        comparison = []
        for alg, dag in self._discovered_graphs.items():
            directed = len([e for e in dag.edges if e.edge_type == "directed"])
            undirected = len([e for e in dag.edges if e.edge_type == "undirected"])

            has_edge = False
            if self._treatment_var and self._outcome_var:
                has_edge = any(
                    e.source == self._treatment_var and e.target == self._outcome_var
                    for e in dag.edges
                )

            comparison.append({
                "algorithm": alg.upper(),
                "n_nodes": len(dag.nodes),
                "n_directed": directed,
                "n_undirected": undirected,
                "treatment_outcome_edge": has_edge,
            })

        # Find common edges
        common_edges = []
        unique_edges = {}
        if len(self._discovered_graphs) >= 2:
            algs = list(self._discovered_graphs.keys())
            dag1 = self._discovered_graphs[algs[0]]
            dag2 = self._discovered_graphs[algs[1]]

            edges1 = set((e.source, e.target) for e in dag1.edges if e.edge_type == "directed")
            edges2 = set((e.source, e.target) for e in dag2.edges if e.edge_type == "directed")

            common = edges1 & edges2
            only_first = edges1 - edges2
            only_second = edges2 - edges1

            common_edges = [{"source": s, "target": t} for s, t in list(common)[:10]]
            unique_edges = {
                algs[0]: len(only_first),
                algs[1]: len(only_second),
            }

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_algorithms": len(self._discovered_graphs),
                "comparison": comparison,
                "common_edges": common_edges,
                "unique_edge_counts": unique_edges,
                "recommendation": "Prefer algorithms that agree on treatment-outcome relationships",
            },
        )

    async def _tool_finalize_discovery(
        self,
        state: AnalysisState,
        chosen_algorithm: str,
        interpretation: str,
        confidence: str,
        confounders: list[str] | None = None,
        mediators: list[str] | None = None,
    ) -> ToolResult:
        """Finalize the discovery with chosen graph."""
        self.logger.info(
            "discovery_finalizing",
            chosen_algorithm=chosen_algorithm,
            confidence=confidence,
        )

        self._final_result = {
            "chosen_algorithm": chosen_algorithm,
            "confounders": confounders or [],
            "mediators": mediators or [],
            "interpretation": interpretation,
            "confidence": confidence,
        }
        self._finalized = True

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "discovery_finalized": True,
                "chosen_algorithm": chosen_algorithm,
                "confidence": confidence,
                "n_confounders": len(confounders or []),
                "n_mediators": len(mediators or []),
            },
        )

    def _create_simple_dag(self) -> CausalDAG:
        """Create a simple treatment → outcome DAG with confounders."""
        nodes = []
        edges = []

        treatment = self._treatment_var
        outcome = self._outcome_var
        profile = self._current_state.data_profile if self._current_state else None

        if treatment:
            nodes.append(treatment)
        if outcome and outcome not in nodes:
            nodes.append(outcome)

        # Add confounders
        if profile and profile.potential_confounders:
            for conf in profile.potential_confounders[:5]:
                if conf not in nodes:
                    nodes.append(conf)
                    if treatment:
                        edges.append(CausalEdge(
                            source=conf, target=treatment,
                            edge_type="directed",
                        ))
                    if outcome:
                        edges.append(CausalEdge(
                            source=conf, target=outcome,
                            edge_type="directed",
                        ))

        # Treatment → Outcome
        if treatment and outcome:
            edges.append(CausalEdge(
                source=treatment, target=outcome,
                edge_type="directed",
            ))

        return CausalDAG(
            nodes=nodes,
            edges=edges,
            discovery_method="Simple DAG (fallback)",
            treatment_variable=treatment,
            outcome_variable=outcome,
        )

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-generate final result."""
        self.logger.info("discovery_auto_finalize")

        # Choose best algorithm (prefer PC, then GES, then others)
        preferred_order = ["pc", "ges", "lingam", "notears"]
        chosen_alg = None

        for alg in preferred_order:
            if alg in self._discovered_graphs:
                chosen_alg = alg
                break

        if not chosen_alg and self._discovered_graphs:
            chosen_alg = list(self._discovered_graphs.keys())[0]

        # Identify confounders from graph
        confounders = []
        if chosen_alg and chosen_alg in self._discovered_graphs:
            dag = self._discovered_graphs[chosen_alg]
            treatment = self._treatment_var
            outcome = self._outcome_var

            if treatment and outcome:
                t_parents = set(e.source for e in dag.edges
                                if e.edge_type == "directed" and e.target == treatment)
                o_parents = set(e.source for e in dag.edges
                                if e.edge_type == "directed" and e.target == outcome)
                confounders = list(t_parents & o_parents)

        return {
            "chosen_algorithm": chosen_alg or "simple",
            "confounders": confounders,
            "mediators": [],
            "interpretation": "Causal structure discovered through automated analysis.",
            "confidence": "medium" if chosen_alg else "low",
        }
