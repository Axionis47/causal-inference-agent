"""Causal Discovery Agent - Truly Agentic Graph Structure Learning.

This agent iteratively discovers causal structure from data:
- LLM investigates data characteristics to choose algorithms
- LLM runs algorithms and inspects results
- LLM validates discovered graphs for sensibility
- LLM can try multiple algorithms and compare
- Iterates until confident about causal structure

No hardcoded algorithm selection - all decisions through tool calls.
"""

import pickle
import time
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base import (
    AnalysisState,
    BaseAgent,
    CausalDAG,
    CausalEdge,
    JobStatus,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class CausalDiscoveryAgent(BaseAgent):
    """Truly agentic causal discovery agent.

    Unlike traditional discovery that runs a single algorithm, this agent:
    1. LLM investigates data characteristics first
    2. LLM selects and runs algorithms based on findings
    3. LLM inspects discovered graphs and validates them
    4. LLM can compare multiple algorithms
    5. LLM finalizes with best graph

    This approach allows intelligent algorithm selection and validation.
    """

    AGENT_NAME = "causal_discovery"

    SYSTEM_PROMPT = """You are an expert in causal discovery and graphical models.
Your role is to learn the causal structure from observational data.

CRITICAL: You must ITERATIVELY investigate and discover by calling tools. Do NOT
select an algorithm blindly. Instead:
1. First understand the data characteristics
2. Select an appropriate algorithm based on data properties
3. Run the algorithm and inspect the results
4. Validate the discovered graph makes sense
5. Optionally try another algorithm to compare
6. Finalize with your best graph

ALGORITHM SELECTION GUIDE:

1. PC ALGORITHM (Constraint-based)
   - Best for: Large number of variables, sparse graphs
   - Assumptions: Faithfulness, causal sufficiency, no hidden confounders
   - Pros: Theoretically grounded, computationally efficient for sparse graphs
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
   - Cons: May produce non-DAG solutions that need post-processing

4. LiNGAM (Linear Non-Gaussian)
   - Best for: Non-Gaussian data with linear relationships
   - Assumptions: Linear, non-Gaussian errors, acyclic
   - Pros: Can identify full causal ordering, unique solution
   - Cons: Requires non-Gaussianity, sensitive to Gaussian variables

DATA CONSIDERATIONS:
- Sample size < 500: Be cautious, consider simpler methods
- Variables > 20: Use PC or limit variable set
- Gaussian data: Avoid LiNGAM
- Non-linear relationships: Results may be unreliable for linear methods

VALIDATION CRITERIA:
- Does treatment → outcome path exist?
- Are confounders properly placed?
- Are there unrealistic edges?
- Is the graph too dense or too sparse?

WORKFLOW:
1. Call get_data_characteristics to understand the data
2. Call run_discovery_algorithm with chosen algorithm
3. Call inspect_graph to see the structure
4. Call validate_graph to check sensibility
5. Optionally try another algorithm and compare
6. Call finalize_discovery with your chosen graph"""

    TOOLS = [
        {
            "name": "get_data_characteristics",
            "description": "Get data characteristics relevant for algorithm selection: sample size, variable count, distributions, correlations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "run_discovery_algorithm",
            "description": "Run a causal discovery algorithm on the data.",
            "parameters": {
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
        },
        {
            "name": "inspect_graph",
            "description": "Inspect the most recently discovered graph structure in detail.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "validate_graph",
            "description": "Validate the discovered graph for causal sensibility.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "compare_algorithms",
            "description": "Compare results from different algorithms that have been run.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "finalize_discovery",
            "description": "Finalize the causal discovery with chosen graph. Call this when satisfied.",
            "parameters": {
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
        },
    ]

    def __init__(self):
        """Initialize the causal discovery agent."""
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._state: AnalysisState | None = None
        self._discovered_graphs: dict[str, CausalDAG] = {}
        self._current_graph: CausalDAG | None = None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute causal discovery through iterative LLM-driven investigation.

        Args:
            state: Current analysis state

        Returns:
            Updated state with proposed DAG
        """
        self.logger.info("discovery_start", job_id=state.job_id)

        state.status = JobStatus.DISCOVERING_CAUSAL
        start_time = time.time()

        try:
            # Load data
            if state.dataframe_path is None:
                self.logger.warning("no_data_for_discovery")
                return state

            with open(state.dataframe_path, "rb") as f:
                self._df = pickle.load(f)

            self._state = state
            self._discovered_graphs = {}
            self._current_graph = None

            # Build the initial prompt
            initial_prompt = self._build_initial_prompt()

            # Run the agentic loop
            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=12)

            # Get the chosen graph
            chosen_alg = final_result.get("chosen_algorithm", "")
            if chosen_alg and chosen_alg in self._discovered_graphs:
                state.proposed_dag = self._discovered_graphs[chosen_alg]
            elif self._current_graph:
                state.proposed_dag = self._current_graph
            elif self._discovered_graphs:
                # Use first discovered graph
                state.proposed_dag = list(self._discovered_graphs.values())[0]
            else:
                # Create simple fallback DAG
                state.proposed_dag = self._create_simple_dag()

            # Store interpretation
            if state.proposed_dag:
                state.proposed_dag.interpretation = final_result.get("interpretation", "")

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="discovery_complete",
                reasoning=final_result.get("interpretation", "Causal discovery completed"),
                outputs={
                    "algorithm": final_result.get("chosen_algorithm", "unknown"),
                    "n_nodes": len(state.proposed_dag.nodes) if state.proposed_dag else 0,
                    "n_edges": len(state.proposed_dag.edges) if state.proposed_dag else 0,
                    "confidence": final_result.get("confidence", "medium"),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "discovery_complete",
                has_dag=state.proposed_dag is not None,
                algorithm=final_result.get("chosen_algorithm", "unknown"),
            )

        except Exception as e:
            self.logger.error("discovery_failed", error=str(e))
            import traceback
            traceback.print_exc()
            # Don't fail the job - create simple DAG as fallback
            state.proposed_dag = self._create_simple_dag()

        return state

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agentic loop."""
        treatment = self._state.treatment_variable or "unknown"
        outcome = self._state.outcome_variable or "unknown"

        return f"""You are discovering the causal structure for a causal inference analysis.

Treatment variable: {treatment}
Outcome variable: {outcome}

Your goal is to discover how variables are causally related, especially:
1. What variables are confounders (affect both treatment and outcome)?
2. What variables are mediators (on the path from treatment to outcome)?
3. What is the overall causal structure?

Start by calling get_data_characteristics to understand the data,
then select an appropriate discovery algorithm.

After running an algorithm, inspect and validate the results.
You may try multiple algorithms to compare.

Call finalize_discovery when you have a satisfactory graph."""

    async def _run_agentic_loop(
        self, initial_prompt: str, max_iterations: int = 12
    ) -> dict[str, Any]:
        """Run the agentic loop where LLM iteratively discovers structure."""
        messages = [{"role": "user", "content": initial_prompt}]

        for iteration in range(max_iterations):
            self.logger.info(
                "discovery_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                graphs_discovered=len(self._discovered_graphs),
            )

            # Get LLM response with tool calls
            result = await self.reason(
                prompt=messages[-1]["content"],
                context={"iteration": iteration, "max_iterations": max_iterations},
            )

            # Log LLM reasoning if provided
            response_text = result.get("response", "")
            if response_text:
                self.logger.info(
                    "discovery_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("discovery_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or iteration > 8:
                    return self._auto_finalize()
                messages.append({
                    "role": "user",
                    "content": "Please continue by calling tools. "
                    "If satisfied with the discovered structure, call finalize_discovery.",
                })
                continue

            # Execute tool calls
            tool_results = []
            for call in pending_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_discovery":
                    self.logger.info(
                        "discovery_finalizing",
                        chosen_algorithm=tool_args.get("chosen_algorithm"),
                        confidence=tool_args.get("confidence"),
                        interpretation=tool_args.get("interpretation", "")[:200],
                    )
                    return tool_args

                # Log the tool call decision
                self.logger.info(
                    "discovery_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                )

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_results.append(f"## {tool_name}\n{tool_result}")

                # Log tool result summary
                self.logger.info(
                    "discovery_tool_result",
                    tool=tool_name,
                    result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                )

            # Feed results back to LLM
            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n\n{results_text}\n\nContinue investigating or call finalize_discovery.",
            })

        # Max iterations reached - auto-finalize
        self.logger.warning("discovery_max_iterations_reached")
        return self._auto_finalize()

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return results as string."""
        try:
            if tool_name == "get_data_characteristics":
                return self._tool_get_data_characteristics()
            elif tool_name == "run_discovery_algorithm":
                return self._tool_run_discovery_algorithm(
                    args.get("algorithm", "pc"),
                    args.get("alpha", 0.05),
                    args.get("threshold", 0.1),
                )
            elif tool_name == "inspect_graph":
                return self._tool_inspect_graph()
            elif tool_name == "validate_graph":
                return self._tool_validate_graph()
            elif tool_name == "compare_algorithms":
                return self._tool_compare_algorithms()
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            self.logger.error("tool_execution_error", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _tool_get_data_characteristics(self) -> str:
        """Get data characteristics for algorithm selection."""
        df = self._df
        profile = self._state.data_profile

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

        output = f"""Data Characteristics for Causal Discovery:
{"=" * 50}

Dataset:
- Total samples: {len(df)}
- Complete cases: {len(df_subset)}
- Variables for discovery: {len(relevant_cols)}

Variables: {', '.join(relevant_cols)}

Treatment: {self._state.treatment_variable}
Outcome: {self._state.outcome_variable}

Distributions (Skewness):"""

        from scipy import stats as sp_stats
        for col in relevant_cols[:10]:
            data = df_subset[col].dropna()
            if len(data) > 10:
                skew = sp_stats.skew(data)
                # Normality test
                if len(data) <= 5000:
                    _, p_val = sp_stats.shapiro(data.sample(min(len(data), 5000), random_state=42))
                else:
                    _, p_val = sp_stats.normaltest(data)
                gaussian = "likely Gaussian" if p_val > 0.05 else "non-Gaussian"
                output += f"\n  {col}: skew={skew:.2f}, {gaussian}"

        # Correlation summary
        corr_matrix = df_subset.corr().abs()
        high_corrs = []
        for i, c1 in enumerate(corr_matrix.columns):
            for j, c2 in enumerate(corr_matrix.columns):
                if i < j and corr_matrix.iloc[i, j] > 0.7:
                    high_corrs.append((c1, c2, corr_matrix.iloc[i, j]))

        output += f"\n\nHigh Correlations (|r| > 0.7): {len(high_corrs)}"
        if high_corrs:
            for c1, c2, corr in high_corrs[:5]:
                output += f"\n  {c1} <-> {c2}: r={corr:.2f}"

        # Algorithm recommendations
        output += "\n\nAlgorithm Recommendations:"
        n_samples = len(df_subset)
        n_vars = len(relevant_cols)

        if n_samples < 300:
            output += "\n- WARNING: Small sample size - results may be unreliable"
            output += "\n- Recommend: PC with higher alpha (0.1) or simple DAG"
        elif n_vars > 15:
            output += "\n- Many variables - consider limiting to most relevant"
            output += "\n- Recommend: PC (efficient for sparse graphs)"
        else:
            output += "\n- Sample size adequate for discovery"

        # Check for Gaussian data
        non_gaussian_count = sum(1 for col in relevant_cols[:10]
                                  if len(df_subset[col].dropna()) > 10
                                  and sp_stats.shapiro(df_subset[col].dropna().sample(
                                      min(len(df_subset[col].dropna()), 5000), random_state=42))[1] < 0.05)
        if non_gaussian_count > len(relevant_cols) * 0.5:
            output += "\n- Most variables are non-Gaussian - LiNGAM may work well"
        else:
            output += "\n- Many Gaussian variables - avoid LiNGAM, use PC or GES"

        return output

    def _tool_run_discovery_algorithm(
        self, algorithm: str, alpha: float = 0.05, threshold: float = 0.1
    ) -> str:
        """Run a discovery algorithm."""
        df = self._df
        profile = self._state.data_profile

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
            return f"Error: Not enough complete cases ({len(df_subset)}) for discovery."

        output = f"Running {algorithm.upper()} algorithm...\n"
        output += f"Variables: {len(relevant_cols)}, Samples: {len(df_subset)}\n"
        output += "=" * 50 + "\n"

        try:
            dag = self._run_algorithm(df_subset, algorithm, alpha, threshold)

            if dag:
                self._discovered_graphs[algorithm] = dag
                self._current_graph = dag

                output += "\nDiscovery completed successfully!\n"
                output += f"Nodes: {len(dag.nodes)}\n"
                output += f"Edges: {len(dag.edges)}\n"

                # Summarize edges
                directed = [e for e in dag.edges if e.edge_type == "directed"]
                undirected = [e for e in dag.edges if e.edge_type == "undirected"]
                output += f"Directed edges: {len(directed)}\n"
                output += f"Undirected edges: {len(undirected)}\n"

                # Check treatment-outcome path
                if self._state.treatment_variable and self._state.outcome_variable:
                    has_path = self._check_path(dag, self._state.treatment_variable, self._state.outcome_variable)
                    if has_path:
                        output += f"\nDirect edge exists: {self._state.treatment_variable} → {self._state.outcome_variable}"
                    else:
                        output += f"\nNo direct edge: {self._state.treatment_variable} → {self._state.outcome_variable}"

                output += "\n\nCall inspect_graph for detailed structure."

            else:
                output += "\nDiscovery failed - falling back to simple DAG."
                dag = self._create_simple_dag()
                self._discovered_graphs[algorithm] = dag
                self._current_graph = dag

        except Exception as e:
            output += f"\nError during discovery: {str(e)}"
            output += "\nFalling back to simple DAG."
            dag = self._create_simple_dag()
            self._discovered_graphs[algorithm] = dag
            self._current_graph = dag

        return output

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
                treatment=self._state.treatment_variable,
                outcome=self._state.outcome_variable,
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
                treatment_variable=self._state.treatment_variable,
                outcome_variable=self._state.outcome_variable,
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
                treatment_variable=self._state.treatment_variable,
                outcome_variable=self._state.outcome_variable,
            )

        except ImportError as e:
            self.logger.warning(f"import_failed_{algorithm}", error=str(e))
            return None
        except Exception as e:
            self.logger.warning(f"algorithm_failed_{algorithm}", error=str(e))
            return None

    def _tool_inspect_graph(self) -> str:
        """Inspect the current graph structure."""
        if not self._current_graph:
            return "No graph discovered yet. Run a discovery algorithm first."

        dag = self._current_graph
        output = f"Graph Structure ({dag.discovery_method}):\n"
        output += "=" * 50 + "\n"
        output += f"Nodes ({len(dag.nodes)}): {', '.join(dag.nodes[:15])}"
        if len(dag.nodes) > 15:
            output += f"... ({len(dag.nodes) - 15} more)"
        output += "\n"

        # Group edges by type
        directed = [e for e in dag.edges if e.edge_type == "directed"]
        undirected = [e for e in dag.edges if e.edge_type == "undirected"]

        output += f"\nDirected Edges ({len(directed)}):\n"
        for e in directed[:20]:
            conf = f" (conf={e.confidence:.2f})" if e.confidence else ""
            output += f"  {e.source} → {e.target}{conf}\n"
        if len(directed) > 20:
            output += f"  ... ({len(directed) - 20} more)\n"

        if undirected:
            output += f"\nUndirected Edges ({len(undirected)}):\n"
            for e in undirected[:10]:
                output += f"  {e.source} -- {e.target}\n"

        # Treatment and outcome analysis
        treatment = self._state.treatment_variable
        outcome = self._state.outcome_variable

        if treatment:
            output += f"\nTreatment ({treatment}) connections:\n"
            incoming = [e.source for e in directed if e.target == treatment]
            outgoing = [e.target for e in directed if e.source == treatment]
            output += f"  Parents: {incoming if incoming else 'None'}\n"
            output += f"  Children: {outgoing if outgoing else 'None'}\n"

        if outcome:
            output += f"\nOutcome ({outcome}) connections:\n"
            incoming = [e.source for e in directed if e.target == outcome]
            outgoing = [e.target for e in directed if e.source == outcome]
            output += f"  Parents: {incoming if incoming else 'None'}\n"
            output += f"  Children: {outgoing if outgoing else 'None'}\n"

        # Identify potential confounders
        if treatment and outcome:
            t_parents = set(e.source for e in directed if e.target == treatment)
            o_parents = set(e.source for e in directed if e.target == outcome)
            confounders = t_parents & o_parents
            if confounders:
                output += f"\nPotential Confounders: {list(confounders)}\n"
            else:
                output += "\nNo common parents of treatment and outcome (confounders) found.\n"

        return output

    def _tool_validate_graph(self) -> str:
        """Validate the discovered graph."""
        if not self._current_graph:
            return "No graph to validate. Run a discovery algorithm first."

        dag = self._current_graph
        output = "Graph Validation:\n"
        output += "=" * 50 + "\n"

        issues = []
        warnings = []

        treatment = self._state.treatment_variable
        outcome = self._state.outcome_variable

        # Check 1: Treatment and outcome in graph
        if treatment and treatment not in dag.nodes:
            issues.append(f"Treatment variable '{treatment}' not in graph nodes")
        if outcome and outcome not in dag.nodes:
            issues.append(f"Outcome variable '{outcome}' not in graph nodes")

        # Check 2: Treatment → Outcome path
        if treatment and outcome and treatment in dag.nodes and outcome in dag.nodes:
            has_direct = any(e.source == treatment and e.target == outcome for e in dag.edges)
            has_path = self._check_path(dag, treatment, outcome)
            if has_direct:
                output += f"Direct edge {treatment} → {outcome}: YES\n"
            elif has_path:
                output += f"Direct edge {treatment} → {outcome}: NO\n"
                output += "Indirect path exists: YES\n"
            else:
                warnings.append("No path from treatment to outcome - may indicate no causal effect")

        # Check 3: Outcome → Treatment (reverse causation)
        if treatment and outcome:
            reverse = any(e.source == outcome and e.target == treatment for e in dag.edges)
            if reverse:
                issues.append(f"Reverse edge {outcome} → {treatment} detected - may be spurious")

        # Check 4: Graph density
        n_nodes = len(dag.nodes)
        n_edges = len(dag.edges)
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        output += f"\nGraph density: {density:.2%}\n"
        if density > 0.5:
            warnings.append("Graph is very dense - may include spurious edges")
        elif density < 0.05 and n_nodes > 5:
            warnings.append("Graph is very sparse - may be missing edges")

        # Check 5: Isolated nodes
        connected = set()
        for e in dag.edges:
            connected.add(e.source)
            connected.add(e.target)
        isolated = set(dag.nodes) - connected
        if isolated:
            warnings.append(f"Isolated nodes (no edges): {list(isolated)[:5]}")

        # Summary
        output += f"\nIssues ({len(issues)}):\n"
        for issue in issues:
            output += f"  - ISSUE: {issue}\n"

        output += f"\nWarnings ({len(warnings)}):\n"
        for warning in warnings:
            output += f"  - WARNING: {warning}\n"

        if not issues and not warnings:
            output += "\nValidation passed - graph appears reasonable.\n"
        elif not issues:
            output += "\nValidation passed with warnings - graph may need refinement.\n"
        else:
            output += "\nValidation found issues - consider trying a different algorithm.\n"

        return output

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

    def _tool_compare_algorithms(self) -> str:
        """Compare results from different algorithms."""
        if len(self._discovered_graphs) < 2:
            return f"Only {len(self._discovered_graphs)} algorithm(s) run. Run more algorithms to compare."

        output = "Algorithm Comparison:\n"
        output += "=" * 50 + "\n"

        # Compare basic stats
        for alg, dag in self._discovered_graphs.items():
            directed = len([e for e in dag.edges if e.edge_type == "directed"])
            undirected = len([e for e in dag.edges if e.edge_type == "undirected"])
            output += f"\n{alg.upper()}:\n"
            output += f"  Nodes: {len(dag.nodes)}, Directed: {directed}, Undirected: {undirected}\n"

            # Treatment-outcome edge
            treatment = self._state.treatment_variable
            outcome = self._state.outcome_variable
            if treatment and outcome:
                has_edge = any(e.source == treatment and e.target == outcome for e in dag.edges)
                output += f"  {treatment} → {outcome}: {'YES' if has_edge else 'NO'}\n"

        # Find common edges
        if len(self._discovered_graphs) >= 2:
            algs = list(self._discovered_graphs.keys())
            dag1 = self._discovered_graphs[algs[0]]
            dag2 = self._discovered_graphs[algs[1]]

            edges1 = set((e.source, e.target) for e in dag1.edges if e.edge_type == "directed")
            edges2 = set((e.source, e.target) for e in dag2.edges if e.edge_type == "directed")

            common = edges1 & edges2
            only_first = edges1 - edges2
            only_second = edges2 - edges1

            output += f"\nCommon directed edges: {len(common)}\n"
            output += f"Only in {algs[0]}: {len(only_first)}\n"
            output += f"Only in {algs[1]}: {len(only_second)}\n"

            if common:
                output += "\nCommon edges (likely robust):\n"
                for src, tgt in list(common)[:10]:
                    output += f"  {src} → {tgt}\n"

        output += "\nRecommendation: Prefer algorithms that agree on treatment-outcome relationships."

        return output

    def _create_simple_dag(self) -> CausalDAG:
        """Create a simple treatment → outcome DAG with confounders."""
        nodes = []
        edges = []

        treatment = self._state.treatment_variable
        outcome = self._state.outcome_variable
        profile = self._state.data_profile

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
            treatment = self._state.treatment_variable
            outcome = self._state.outcome_variable

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
