"""Confounder Discovery Agent - Autonomously identifies confounders using causal reasoning.

This agent uses TRUE agentic tool-use: the LLM decides what evidence to compute,
calls tools to gather that evidence, and iteratively reasons about confounders.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import AnalysisState, CausalDAG, JobStatus, ToolResult, ToolResultStatus
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.agents.registry import register_agent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


def _classify_variable_role(
    dag: CausalDAG | None,
    variable: str,
    treatment: str,
    outcome: str,
    domain_knowledge: dict | None = None,
) -> str:
    """Classify a variable's causal role relative to treatment and outcome.

    Uses DAG topology if available, otherwise falls back to domain knowledge
    temporal ordering.

    Returns: "confounder", "potential_mediator", "potential_collider", or "unknown".
    """
    # Strategy 1: Use DAG topology if available
    if dag and dag.edges:
        import networkx as nx

        G = nx.DiGraph()
        for edge in dag.edges:
            if edge.edge_type == "directed":
                G.add_edge(edge.source, edge.target)

        if variable in G.nodes and treatment in G.nodes and outcome in G.nodes:
            is_ancestor_of_t = variable in nx.ancestors(G, treatment)
            is_ancestor_of_y = variable in nx.ancestors(G, outcome)
            is_descendant_of_t = variable in nx.descendants(G, treatment)
            is_descendant_of_y = variable in nx.descendants(G, outcome)

            if is_ancestor_of_t and is_ancestor_of_y:
                return "confounder"
            if is_descendant_of_t and is_ancestor_of_y:
                return "potential_mediator"
            if is_descendant_of_t and is_descendant_of_y:
                return "potential_collider"
            # Ancestor of T only, or ancestor of Y only — still safe to adjust for
            if is_ancestor_of_t or is_ancestor_of_y:
                return "confounder"
            return "unknown"

    # Strategy 2: Use domain knowledge temporal ordering
    if domain_knowledge:
        temporal = domain_knowledge.get("temporal_understanding", {})
        pre_treatment = temporal.get("pre_treatment_vars", [])
        post_treatment = temporal.get("post_treatment_vars", [])

        if variable in post_treatment:
            return "potential_mediator"
        if variable in pre_treatment:
            return "confounder"

    return "unknown"


@register_agent("confounder_discovery")
class ConfounderDiscoveryAgent(ReActAgent, ContextTools):
    """Agent that discovers confounders through iterative tool-based reasoning.

    This agent is TRULY AGENTIC - the LLM:
    1. Decides what statistical tests to run
    2. Calls tools to compute evidence
    3. Iterates based on results
    4. Makes final decisions based on gathered evidence

    Uses ReAct pattern with pull-based context tools.
    """

    AGENT_NAME = "confounder_discovery"
    MAX_STEPS = 15
    WRITES_STATE_FIELDS = ["confounder_discovery"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.DISCOVERING_CAUSAL
    PROGRESS_WEIGHT = 0.06

    SYSTEM_PROMPT = """You are an expert causal inference researcher investigating confounders.

Your task is to identify variables that confound the treatment-outcome relationship.
A CONFOUNDER is a variable that:
1. Affects the TREATMENT (causes or is associated with treatment assignment)
2. Affects the OUTCOME (causes or is associated with the outcome)
3. Is NOT caused by the treatment (not a mediator or collider)

You have tools to gather statistical evidence. USE THEM to investigate each variable:
- compute_correlation: Check association between any two variables
- compute_partial_correlation: Check if association remains after controlling for another variable
- test_confounder_criteria: Test if a variable meets confounder criteria
- finalize_confounders: Submit your final list of confounders

INVESTIGATION STRATEGY:
1. First, get the list of candidate variables
2. For each candidate, compute its correlation with BOTH treatment AND outcome
3. If a variable correlates with both, it's a potential confounder - investigate further
4. Use partial correlations to distinguish confounders from mediators
5. Rank confounders by the strength of their confounding effect

Be THOROUGH - missing a true confounder leads to biased causal estimates.
Call tools to gather evidence, don't just guess.

CONTEXT TOOLS (pull upstream results if needed):
- ask_domain_knowledge: Query domain knowledge for immutable variables, temporal ordering
- get_eda_finding: Query EDA results (e.g. "covariate balance", "outliers")"""

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._investigation_log: list[dict] = []
        self._finalized: bool = False

        # Register context tools from mixin
        self.register_context_tools()
        # Register discovery-specific tools
        self._register_discovery_tools()

    def _register_discovery_tools(self) -> None:
        """Register confounder discovery tools."""
        self.register_tool(
            name="get_candidate_variables",
            description="Get the list of candidate variables to investigate as potential confounders",
            handler=self._tool_get_candidates,
            parameters={},
        )

        self.register_tool(
            name="compute_correlation",
            description="Compute Pearson correlation between two variables. Use this to check if a variable is associated with treatment or outcome.",
            handler=self._tool_compute_correlation,
            parameters={
                "var1": {"type": "string", "description": "First variable name"},
                "var2": {"type": "string", "description": "Second variable name"},
            },
        )

        self.register_tool(
            name="compute_partial_correlation",
            description="Compute partial correlation between var1 and var2, controlling for control_var. Helps distinguish confounders from mediators.",
            handler=self._tool_compute_partial_correlation,
            parameters={
                "var1": {"type": "string", "description": "First variable"},
                "var2": {"type": "string", "description": "Second variable"},
                "control_var": {"type": "string", "description": "Variable to control for"},
            },
        )

        self.register_tool(
            name="test_confounder_criteria",
            description="Test if a variable meets the statistical criteria for being a confounder (associated with both T and Y)",
            handler=self._tool_test_confounder_criteria,
            parameters={
                "variable": {"type": "string", "description": "Variable to test"},
            },
        )

        self.register_tool(
            name="finalize_confounders",
            description="Submit your final list of identified confounders after investigation",
            handler=self._tool_finalize_confounders,
            parameters={
                "confounders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of confirmed confounders, ranked by importance",
                },
                "excluded": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variables investigated but excluded (mediators, colliders, noise)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of your confounder identification process",
                },
            },
        )

    def _resolve_treatment_outcome(self) -> tuple[str | None, str | None]:
        """Get treatment and outcome variables using state helper.

        Returns:
            Tuple of (treatment_var, outcome_var)
        """
        if self._treatment_var and self._outcome_var:
            return self._treatment_var, self._outcome_var

        if self._current_state:
            t, o = self._current_state.get_primary_pair()
            self._treatment_var = t
            self._outcome_var = o
            return t, o

        return None, None

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Build initial observation for confounder discovery."""
        candidates = self._get_candidates()
        treatment = self._treatment_var or "unknown"
        outcome = self._outcome_var or "unknown"

        # Check if DAG is empty or missing
        dag_available = (
            state.proposed_dag is not None
            and hasattr(state.proposed_dag, "edges")
            and len(state.proposed_dag.edges) > 0
        )
        dag_note = ""
        if not dag_available:
            dag_note = (
                "\n\nIMPORTANT: No causal graph available (DAG is empty or missing). "
                "Use statistical methods to identify confounders. Focus on correlations "
                "with both treatment and outcome variables."
            )

        return f"""CONFOUNDER DISCOVERY TASK
Treatment variable: {treatment}
Outcome variable: {outcome}
Candidate variables to investigate: {candidates}{dag_note}

Your goal is to identify which candidates are confounders (affect both treatment and outcome).
Use the tools to systematically investigate each candidate:
1. Start by getting the candidate list
2. For each candidate, test if it meets confounder criteria
3. Use correlations and partial correlations to verify
4. Call finalize_confounders when done with your ranked list."""

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if confounder discovery is complete."""
        return self._finalized

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute confounder discovery through iterative tool-based reasoning."""
        self.logger.info("confounder_discovery_start", job_id=state.job_id)
        start_time = time.time()

        try:
            # Load data
            self._df = self._load_dataframe(state)
            if self._df is None:
                self.logger.warning("no_dataframe_for_confounder_discovery")
                return state

            # Store state reference for helper methods
            self._current_state = state
            self._investigation_log = []
            self._finalized = False

            # Use helper to resolve treatment/outcome (falls back to analyzed_pairs)
            self._treatment_var, self._outcome_var = self._resolve_treatment_outcome()

            if not self._treatment_var or not self._outcome_var:
                self.logger.warning("no_treatment_or_outcome_specified")
                return state

            # Run ReAct loop - LLM calls tools iteratively
            await super().execute(state)

            # Auto-finalize if LLM didn't call finalize_confounders
            if not self._finalized:
                self.logger.info("using_fallback_confounder_identification")
                state = self._fallback_confounder_identification(state)

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="confounder_discovery_complete",
                reasoning="Identified confounders through agentic tool-based investigation",
                outputs=state.confounder_discovery if state.confounder_discovery else {},
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

        except Exception as e:
            self.logger.error("confounder_discovery_failed", error=str(e))
            state = self._fallback_confounder_identification(state)

        return state

    # -------------------------------------------------------------------------
    # Tool Handlers (async, return ToolResult)
    # -------------------------------------------------------------------------

    async def _tool_get_candidates(self, state: AnalysisState, **kwargs) -> ToolResult:
        """Get candidate variables for confounder investigation."""
        if kwargs:
            logger.debug("tool_ignored_kwargs", tool="get_candidate_variables", extra_keys=list(kwargs.keys()))
        candidates = self._get_candidates()
        self._investigation_log.append({
            "tool": "get_candidate_variables",
            "args": {},
            "result": candidates,
        })
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "candidates": candidates,
                "treatment": self._treatment_var,
                "outcome": self._outcome_var,
            },
        )

    async def _tool_compute_correlation(
        self, state: AnalysisState, var1: str = "", var2: str = "", **kwargs
    ) -> ToolResult:
        """Compute Pearson correlation between two variables."""
        if kwargs:
            logger.debug("tool_ignored_kwargs", tool="compute_correlation", extra_keys=list(kwargs.keys()))
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No data loaded",
            )

        if var1 not in self._df.columns or var2 not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Variable not found. Available: {list(self._df.columns)}",
            )

        corr, pval = stats.pearsonr(self._df[var1], self._df[var2])
        significance = "significant" if pval < 0.05 else "not significant"
        strength = "strong" if abs(corr) > 0.3 else ("moderate" if abs(corr) > 0.1 else "weak")

        result = {
            "var1": var1,
            "var2": var2,
            "correlation": float(corr),
            "p_value": float(pval),
            "strength": strength,
            "significance": significance,
        }

        self._investigation_log.append({
            "tool": "compute_correlation",
            "args": {"var1": var1, "var2": var2},
            "result": result,
        })

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_compute_partial_correlation(
        self, state: AnalysisState, var1: str = "", var2: str = "", control_var: str = "", **kwargs
    ) -> ToolResult:
        """Compute partial correlation controlling for a third variable."""
        if kwargs:
            logger.debug("tool_ignored_kwargs", tool="compute_partial_correlation", extra_keys=list(kwargs.keys()))
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No data loaded",
            )

        for v in [var1, var2, control_var]:
            if v not in self._df.columns:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    error=f"Variable '{v}' not found",
                )

        from sklearn.linear_model import LinearRegression

        X = self._df[[var1, var2, control_var]].dropna()
        v1_vals = X[var1].values
        v2_vals = X[var2].values
        ctrl = X[control_var].values.reshape(-1, 1)

        # Residualize
        v1_resid = v1_vals - LinearRegression().fit(ctrl, v1_vals).predict(ctrl)
        v2_resid = v2_vals - LinearRegression().fit(ctrl, v2_vals).predict(ctrl)

        partial_corr, pval = stats.pearsonr(v1_resid, v2_resid)

        result = {
            "var1": var1,
            "var2": var2,
            "control_var": control_var,
            "partial_correlation": float(partial_corr),
            "p_value": float(pval),
        }

        self._investigation_log.append({
            "tool": "compute_partial_correlation",
            "args": {"var1": var1, "var2": var2, "control_var": control_var},
            "result": result,
        })

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_test_confounder_criteria(
        self, state: AnalysisState, variable: str = "", **kwargs
    ) -> ToolResult:
        """Test if a variable meets confounder criteria."""
        if kwargs:
            logger.debug("tool_ignored_kwargs", tool="test_confounder_criteria", extra_keys=list(kwargs.keys()))
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No data loaded",
            )

        if variable not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Variable '{variable}' not found",
            )

        # Test association with treatment
        corr_t, pval_t = stats.pearsonr(self._df[variable], self._df[self._treatment_var])
        # Test association with outcome
        corr_y, pval_y = stats.pearsonr(self._df[variable], self._df[self._outcome_var])

        # Load configurable thresholds
        from src.config.settings import get_settings
        _settings = get_settings()
        _corr_thresh = _settings.confounder_correlation_threshold
        _prog_thresh = _settings.confounder_prognostic_threshold
        _bal_thresh = _settings.balance_pvalue_threshold

        # Criterion 1: Classic confounder (low threshold for RCT compatibility)
        affects_treatment = abs(corr_t) > _corr_thresh
        affects_outcome = abs(corr_y) > _corr_thresh
        is_classic = affects_treatment and affects_outcome

        # Criterion 2: Prognostic variable (strongly predicts Y)
        is_prognostic = abs(corr_y) > _prog_thresh

        # Criterion 3: Imbalanced between treatment groups (t-test)
        try:
            treated_mask = self._df[self._treatment_var] == 1
            if treated_mask.sum() == 0 or (~treated_mask).sum() == 0:
                treated_mask = self._df[self._treatment_var] > self._df[self._treatment_var].median()
            treated_vals = self._df.loc[treated_mask, variable].dropna()
            control_vals = self._df.loc[~treated_mask, variable].dropna()
            if len(treated_vals) > 2 and len(control_vals) > 2:
                _, balance_p = stats.ttest_ind(treated_vals, control_vals, equal_var=False)
                is_imbalanced = balance_p < _bal_thresh
            else:
                is_imbalanced = False
                balance_p = 1.0
        except Exception:
            is_imbalanced = False
            balance_p = 1.0

        statistically_associated = is_classic or is_prognostic or is_imbalanced
        confounder_strength = abs(corr_y)  # rank by outcome prediction strength

        # Classify causal role using DAG or domain knowledge
        causal_role = _classify_variable_role(
            dag=state.proposed_dag,
            variable=variable,
            treatment=self._treatment_var,
            outcome=self._outcome_var,
            domain_knowledge=state.domain_knowledge if hasattr(state, "domain_knowledge") else None,
        )

        # A variable is a safe confounder only if it meets any criterion
        # AND not classified as a mediator or collider
        is_confounder = (
            statistically_associated
            and causal_role not in ("potential_mediator", "potential_collider")
        )

        reasons = []
        if is_classic:
            reasons.append(f"correlated with T ({abs(corr_t):.3f}) and Y ({abs(corr_y):.3f})")
        if is_prognostic:
            reasons.append(f"prognostic variable (corr_Y={abs(corr_y):.3f})")
        if is_imbalanced:
            reasons.append(f"imbalanced across groups (p={balance_p:.3f})")

        result = {
            "variable": variable,
            "treatment_var": self._treatment_var,
            "outcome_var": self._outcome_var,
            "corr_with_treatment": float(corr_t),
            "pval_treatment": float(pval_t),
            "affects_treatment": affects_treatment,
            "corr_with_outcome": float(corr_y),
            "pval_outcome": float(pval_y),
            "affects_outcome": affects_outcome,
            "is_prognostic": is_prognostic,
            "is_imbalanced": is_imbalanced,
            "balance_p_value": float(balance_p),
            "reasons": reasons,
            "causal_role": causal_role,
            "is_confounder": is_confounder,
            "confounder_strength": float(confounder_strength) if is_confounder else 0.0,
        }

        self._investigation_log.append({
            "tool": "test_confounder_criteria",
            "args": {"variable": variable},
            "result": result,
        })

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _statistical_confounder_scan(
        self, state: AnalysisState, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> list[dict]:
        """Fallback: identify confounders via multiple statistical criteria.

        Uses three criteria (any one sufficient):
        1. CLASSIC CONFOUNDER: correlated with both T and Y (threshold: 0.03)
        2. PROGNOSTIC VARIABLE: strongly predicts Y (|corr| > 0.1), even if weakly related to T
           (adjusting for prognostic variables reduces variance and improves precision)
        3. BALANCE INDICATOR: different means between treatment groups (t-test p < 0.1)
        """
        confounders = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude oracle/ground-truth columns
        oracle_patterns = ['y0_true', 'y1_true', 'y0', 'y1', 'cate_true', 'cate', 'ite',
                           'propensity_true', 'propensity_score', 'true_propensity', 'counterfactual']
        candidate_cols = [
            c for c in numeric_cols
            if c not in (treatment_col, outcome_col)
            and not any(p in c.lower() for p in oracle_patterns)
        ]

        treatment_vals = df[treatment_col].values.astype(float)
        outcome_vals = df[outcome_col].values.astype(float)

        # Split treatment groups for balance check
        treated_mask = treatment_vals == 1
        if treated_mask.sum() == 0 or (~treated_mask).sum() == 0:
            # Can't do balance check with no variation in treatment
            treated_mask = treatment_vals > np.median(treatment_vals)

        # Load configurable thresholds
        from src.config.settings import get_settings
        _settings = get_settings()
        _corr_thresh = _settings.confounder_correlation_threshold
        _prog_thresh = _settings.confounder_prognostic_threshold
        _bal_thresh = _settings.balance_pvalue_threshold

        for col in candidate_cols:
            try:
                col_data = df[col].values.astype(float)
                valid = ~(np.isnan(col_data) | np.isnan(treatment_vals) | np.isnan(outcome_vals))
                if valid.sum() < 20:
                    continue

                c = col_data[valid]
                t = treatment_vals[valid]
                y = outcome_vals[valid]

                corr_t = abs(np.corrcoef(c, t)[0, 1]) if not np.isnan(np.corrcoef(c, t)[0, 1]) else 0
                corr_y = abs(np.corrcoef(c, y)[0, 1]) if not np.isnan(np.corrcoef(c, y)[0, 1]) else 0

                # Criterion 1: Classic confounder (lower threshold)
                is_classic = corr_t > _corr_thresh and corr_y > _corr_thresh

                # Criterion 2: Prognostic variable (strongly predicts Y)
                is_prognostic = corr_y > _prog_thresh

                # Criterion 3: Imbalanced between treatment groups
                try:
                    treated_vals_col = c[treated_mask[valid]]
                    control_vals_col = c[~treated_mask[valid]]
                    if len(treated_vals_col) > 2 and len(control_vals_col) > 2:
                        _, balance_p = stats.ttest_ind(treated_vals_col, control_vals_col, equal_var=False)
                        is_imbalanced = balance_p < _bal_thresh
                    else:
                        is_imbalanced = False
                        balance_p = 1.0
                except Exception:
                    is_imbalanced = False
                    balance_p = 1.0

                if is_classic or is_prognostic or is_imbalanced:
                    reasons = []
                    if is_classic:
                        reasons.append(f"correlated with T ({corr_t:.3f}) and Y ({corr_y:.3f})")
                    if is_prognostic:
                        reasons.append(f"prognostic variable (corr_Y={corr_y:.3f})")
                    if is_imbalanced:
                        reasons.append(f"imbalanced across groups (p={balance_p:.3f})")

                    confounders.append({
                        "variable": col,
                        "corr_with_treatment": round(float(corr_t), 3),
                        "corr_with_outcome": round(float(corr_y), 3),
                        "balance_p_value": round(float(balance_p), 3),
                        "strength": round(float(corr_y), 4),  # rank by outcome prediction strength
                        "reasons": reasons,
                        "source": "statistical_fallback",
                    })

                    self.logger.info(
                        "statistical_scan_confounder_found",
                        variable=col,
                        corr_t=round(float(corr_t), 3),
                        corr_y=round(float(corr_y), 3),
                        balance_p=round(float(balance_p), 3),
                        reasons=reasons,
                    )
            except Exception:
                continue

        # Sort by outcome prediction strength (most prognostic first)
        confounders.sort(key=lambda x: x["strength"], reverse=True)

        self.logger.info(
            "statistical_scan_complete",
            n_candidates=len(candidate_cols),
            n_confounders=len(confounders),
        )
        return confounders

    async def _tool_finalize_confounders(
        self,
        state: AnalysisState,
        confounders: list[str] | None = None,
        reasoning: str = "",
        excluded: list[str] | None = None,
        **kwargs,
    ) -> ToolResult:
        """Finalize the list of identified confounders."""
        if kwargs:
            logger.debug("tool_ignored_kwargs", tool="finalize_confounders", extra_keys=list(kwargs.keys()))
        confounders = confounders or []
        excluded = excluded or []

        # --- Statistical fallback when 0 confounders found ---
        fallback_details = None
        if len(confounders) == 0 and self._df is not None and self._treatment_var and self._outcome_var:
            self.logger.info("finalize_zero_confounders_triggering_statistical_fallback")

            scan_results = await self._statistical_confounder_scan(
                state, self._df, self._treatment_var, self._outcome_var
            )

            if scan_results:
                confounders = [r["variable"] for r in scan_results]
                fallback_details = scan_results
                reasoning = (
                    f"Statistical fallback: DAG-based confounder identification produced no results. "
                    f"Correlation scan found {len(confounders)} variable(s) associated with both "
                    f"treatment ({self._treatment_var}) and outcome ({self._outcome_var})."
                )

                state.push_decision(
                    agent="confounder_discovery",
                    decision_type="statistical_fallback",
                    choice=f"Found {len(confounders)} confounders via correlation scan",
                    reason="DAG-based confounder identification produced no results, falling back to statistical correlation with T and Y",
                )

                self.logger.info(
                    "statistical_fallback_confounders_found",
                    n_confounders=len(confounders),
                    top_confounders=confounders[:5],
                )

        self.logger.info(
            "confounder_finalizing",
            n_confounders=len(confounders),
            top_confounders=confounders[:5] if confounders else [],
        )

        # Update state
        if state.data_profile:
            state.data_profile.potential_confounders = confounders

        state.confounder_discovery = {
            "ranked_confounders": confounders,
            "excluded_variables": excluded,
            "adjustment_strategy": reasoning,
            "investigation_log": self._investigation_log,
        }

        # Include fallback details so downstream agents know the source
        if fallback_details:
            state.confounder_discovery["statistical_fallback_details"] = fallback_details

        self._finalized = True

        state.push_decision(
            agent="confounder_discovery",
            decision_type="confounders_selected",
            choice=", ".join(confounders) if confounders else "(none)",
            reason=f"Selected {len(confounders)} confounder(s) based on statistical criteria (correlation with both {self._treatment_var} and {self._outcome_var}) and domain knowledge; excluded {len(excluded)} variable(s) as mediators/colliders/noise",
        )

        self.logger.info(
            "confounder_discovery_complete",
            n_confounders=len(confounders),
            n_excluded=len(excluded),
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "confounders": confounders,
                "excluded": excluded,
                "reasoning": reasoning,
                **({"fallback_scan": fallback_details} if fallback_details else {}),
            },
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_candidates(self) -> list[str]:
        """Get candidate variables for confounder investigation."""
        if self._df is None:
            return []

        # All numeric columns except treatment and outcome
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [
            c for c in numeric_cols
            if c != self._treatment_var and c != self._outcome_var
        ]
        return candidates

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load dataframe from state."""
        if state.dataframe_path:
            try:
                if state.dataframe_path.endswith(".csv"):
                    return pd.read_csv(state.dataframe_path)
                else:
                    return pd.read_parquet(state.dataframe_path)
            except Exception as e:
                self.logger.error("load_failed", error=str(e))
        return None

    def _fallback_confounder_identification(self, state: AnalysisState) -> AnalysisState:
        """Fallback when agentic loop fails - use statistical heuristics.

        Uses three criteria (any one sufficient):
        1. Classic confounder: correlated with both T and Y (threshold: 0.03)
        2. Prognostic variable: strongly predicts Y (|corr| > 0.1)
        3. Balance indicator: different means between treatment groups (t-test p < 0.1)
        """
        self.logger.info("using_statistical_fallback")

        if self._df is None:
            return state

        candidates = self._get_candidates()
        confounders = []
        excluded = []

        treatment_vals = self._df[self._treatment_var].values.astype(float)

        # Split treatment groups for balance check
        treated_mask = treatment_vals == 1
        if treated_mask.sum() == 0 or (~treated_mask).sum() == 0:
            treated_mask = treatment_vals > np.median(treatment_vals)

        # Load configurable thresholds
        from src.config.settings import get_settings
        _settings = get_settings()
        _corr_thresh = _settings.confounder_correlation_threshold
        _prog_thresh = _settings.confounder_prognostic_threshold
        _bal_thresh = _settings.balance_pvalue_threshold

        for col in candidates:
            try:
                corr_t = abs(stats.pearsonr(self._df[col], self._df[self._treatment_var])[0])
                corr_y = abs(stats.pearsonr(self._df[col], self._df[self._outcome_var])[0])

                # Criterion 1: Classic confounder
                is_classic = corr_t > _corr_thresh and corr_y > _corr_thresh

                # Criterion 2: Prognostic variable
                is_prognostic = corr_y > _prog_thresh

                # Criterion 3: Imbalanced between treatment groups
                try:
                    treated_vals_col = self._df.loc[treated_mask, col].dropna()
                    control_vals_col = self._df.loc[~treated_mask, col].dropna()
                    if len(treated_vals_col) > 2 and len(control_vals_col) > 2:
                        _, balance_p = stats.ttest_ind(treated_vals_col, control_vals_col, equal_var=False)
                        is_imbalanced = balance_p < _bal_thresh
                    else:
                        is_imbalanced = False
                except Exception:
                    is_imbalanced = False

                if is_classic or is_prognostic or is_imbalanced:
                    # Check causal role -- exclude mediators and colliders
                    role = _classify_variable_role(
                        dag=state.proposed_dag,
                        variable=col,
                        treatment=self._treatment_var,
                        outcome=self._outcome_var,
                        domain_knowledge=state.domain_knowledge if hasattr(state, "domain_knowledge") else None,
                    )
                    if role in ("potential_mediator", "potential_collider"):
                        excluded.append(col)
                        logger.info(
                            "fallback_excluded_variable",
                            variable=col,
                            causal_role=role,
                        )
                    else:
                        reasons = []
                        if is_classic:
                            reasons.append(f"correlated with T ({corr_t:.3f}) and Y ({corr_y:.3f})")
                        if is_prognostic:
                            reasons.append(f"prognostic (corr_Y={corr_y:.3f})")
                        if is_imbalanced:
                            reasons.append("imbalanced across groups")
                        confounders.append((col, corr_y, reasons))
                        logger.info(
                            "fallback_confounder_found",
                            variable=col,
                            corr_y=round(corr_y, 3),
                            reasons=reasons,
                        )
            except Exception:
                logger.debug("correlation_computation_skipped", column=col, exc_info=True)

        # Sort by outcome prediction strength
        confounders.sort(key=lambda x: x[1], reverse=True)
        ranked = [c for c, _, _ in confounders]

        if state.data_profile:
            state.data_profile.potential_confounders = ranked

        state.confounder_discovery = {
            "ranked_confounders": ranked,
            "excluded_variables": excluded,
            "adjustment_strategy": (
                "Statistical fallback - three criteria (classic confounder, prognostic variable, "
                "balance indicator), filtered by causal role"
            ),
            "investigation_log": [],
        }

        self.logger.info(
            "fallback_confounder_identification_complete",
            n_confounders=len(ranked),
            n_excluded=len(excluded),
            top_confounders=ranked[:5],
        )

        return state
