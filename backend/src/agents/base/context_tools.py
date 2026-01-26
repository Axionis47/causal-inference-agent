"""
Shared context query tools for all agents.

Agents pull what they need via tools - no context dumps.
This module provides a mixin class that any ReAct agent can use
to query domain knowledge, data profiles, EDA results, etc.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .react_agent import ToolResult, ToolResultStatus

if TYPE_CHECKING:
    from .state import AnalysisState


class ContextTools:
    """
    Mixin providing context query tools for ReAct agents.

    Usage:
        class MyAgent(ReActAgent, ContextTools):
            def __init__(self):
                super().__init__()
                self.register_context_tools()

    This provides tools for agents to pull context on demand:
    - ask_domain_knowledge: Query domain knowledge findings
    - get_column_info: Get info about a specific column
    - get_dataset_summary: Get basic dataset stats
    - get_eda_finding: Query EDA results by topic
    - get_previous_finding: Get findings from previous agents
    """

    def register_context_tools(self) -> None:
        """Register all context query tools with the agent."""

        # Domain Knowledge queries
        self.register_tool(
            name="ask_domain_knowledge",
            description=(
                "Ask a question about domain knowledge. Use this to get hints about "
                "treatment, outcome, confounders, temporal ordering, or causal constraints. "
                "Examples: 'What is the likely treatment?', 'Is age immutable?', "
                "'What is the temporal ordering?', 'Are there any forbidden edges?'"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your question about the domain or causal structure"
                    }
                },
                "required": ["question"]
            },
            handler=self._ask_domain_knowledge,
        )

        # Data Profile queries
        self.register_tool(
            name="get_column_info",
            description=(
                "Get information about a specific column: dtype, basic stats, "
                "unique values, missing count. Use this to investigate individual variables."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to investigate"
                    }
                },
                "required": ["column"]
            },
            handler=self._get_column_info,
        )

        self.register_tool(
            name="get_dataset_summary",
            description=(
                "Get basic dataset summary: row count, column count, column names, "
                "data types overview. Use this for initial orientation."
            ),
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._get_dataset_summary,
        )

        self.register_tool(
            name="list_columns",
            description="Get list of all column names in the dataset.",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._list_columns,
        )

        # EDA queries
        self.register_tool(
            name="get_eda_finding",
            description=(
                "Query EDA results for a specific topic. "
                "Topics: 'outliers', 'correlations', 'balance', 'missing', "
                "'multicollinearity', 'quality_score', 'distributions'"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": [
                            "outliers",
                            "correlations",
                            "balance",
                            "missing",
                            "multicollinearity",
                            "quality_score",
                            "distributions"
                        ],
                        "description": "The EDA topic to query"
                    }
                },
                "required": ["topic"]
            },
            handler=self._get_eda_finding,
        )

        # Previous agent findings
        self.register_tool(
            name="get_previous_finding",
            description=(
                "Get key findings from a previous agent in the pipeline. "
                "Use this to build on earlier analysis without re-doing work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": [
                            "domain_knowledge",
                            "data_profiler",
                            "eda_agent",
                            "causal_discovery",
                            "effect_estimator"
                        ],
                        "description": "Which agent's findings to retrieve"
                    }
                },
                "required": ["agent"]
            },
            handler=self._get_previous_finding,
        )

        # Treatment/Outcome info
        self.register_tool(
            name="get_treatment_outcome",
            description=(
                "Get the current treatment and outcome variables. "
                "Returns what has been identified so far."
            ),
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._get_treatment_outcome,
        )

        # Confounder analysis (on-demand)
        self.register_tool(
            name="get_confounder_analysis",
            description=(
                "Get confounder analysis results including ranked confounders, "
                "their relationships with treatment and outcome, and recommended "
                "adjustment strategy. Call this before running methods that require "
                "covariate adjustment."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top confounders to return (default 10)"
                    }
                },
                "required": []
            },
            handler=self._get_confounder_analysis,
        )

        # Profile for specific variables (on-demand)
        self.register_tool(
            name="get_profile_for_variables",
            description=(
                "Get profile statistics for specific variables only. "
                "More efficient than loading full profile when you only need "
                "stats for a few columns (e.g., treatment, outcome, key confounders)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of variable names to get stats for"
                    }
                },
                "required": ["variables"]
            },
            handler=self._get_profile_for_variables,
        )

        # Dataset context (semantic understanding)
        self.register_tool(
            name="get_dataset_context",
            description=(
                "Get semantic context about the dataset including domain, "
                "variable descriptions from metadata, and study design hints. "
                "Helps understand WHAT variables mean, not just their statistics."
            ),
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._get_dataset_context,
        )

        # Variable semantics analysis
        self.register_tool(
            name="analyze_variable_semantics",
            description=(
                "Analyze what a variable likely represents semantically, "
                "using column name, description from metadata, and domain context. "
                "Helps distinguish demographics (age, race) from outcomes (income, score)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Variable name to analyze"
                    }
                },
                "required": ["variable"]
            },
            handler=self._analyze_variable_semantics,
        )

        # DAG-based adjustment set (uses backdoor criterion)
        self.register_tool(
            name="get_dag_adjustment_set",
            description=(
                "Get the proper adjustment set from the causal DAG using the "
                "backdoor criterion. Returns confounders to adjust for, mediators "
                "to avoid, and colliders that must NOT be adjusted. "
                "This is the recommended way to select covariates for effect estimation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "treatment": {
                        "type": "string",
                        "description": "Treatment variable (optional, uses state default)"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "Outcome variable (optional, uses state default)"
                    }
                },
                "required": []
            },
            handler=self._get_dag_adjustment_set,
        )

    async def _ask_domain_knowledge(
        self,
        state: AnalysisState,
        question: str
    ) -> ToolResult:
        """
        Search domain knowledge for relevant information.

        Searches through hypotheses, uncertainties, and other findings
        to answer the agent's question.
        """
        dk = state.domain_knowledge

        if not dk:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "found": False,
                    "message": "No domain knowledge available yet. Use your judgment or investigate the data directly."
                }
            )

        question_lower = question.lower()
        relevant_findings = []

        # Search hypotheses
        hypotheses = dk.get("hypotheses", [])
        for h in hypotheses:
            claim = h.get("claim", "").lower()
            # Check if hypothesis is relevant to the question
            if self._is_relevant(question_lower, claim):
                relevant_findings.append({
                    "type": "hypothesis",
                    "claim": h.get("claim"),
                    "confidence": h.get("confidence"),
                    "evidence": h.get("evidence")
                })

        # Search uncertainties
        uncertainties = dk.get("uncertainties", [])
        for u in uncertainties:
            issue = u.get("issue", "").lower()
            if self._is_relevant(question_lower, issue):
                relevant_findings.append({
                    "type": "uncertainty",
                    "issue": u.get("issue"),
                    "impact": u.get("impact")
                })

        # Check for specific question types
        if "treatment" in question_lower:
            treatment_hyps = [h for h in hypotheses if "treatment" in h.get("claim", "").lower()]
            if treatment_hyps:
                relevant_findings = [{"type": "hypothesis", **h} for h in treatment_hyps] + relevant_findings

        if "outcome" in question_lower:
            outcome_hyps = [h for h in hypotheses if "outcome" in h.get("claim", "").lower()]
            if outcome_hyps:
                relevant_findings = [{"type": "hypothesis", **h} for h in outcome_hyps] + relevant_findings

        if "temporal" in question_lower or "order" in question_lower:
            if "temporal_understanding" in dk:
                relevant_findings.insert(0, {
                    "type": "temporal_ordering",
                    "understanding": dk["temporal_understanding"]
                })

        if "immutable" in question_lower or "forbidden" in question_lower:
            if "immutable_vars" in dk:
                relevant_findings.insert(0, {
                    "type": "immutable_variables",
                    "variables": dk["immutable_vars"]
                })

        if "confounder" in question_lower:
            confounder_hyps = [h for h in hypotheses if "confounder" in h.get("claim", "").lower()]
            if confounder_hyps:
                relevant_findings = [{"type": "hypothesis", **h} for h in confounder_hyps] + relevant_findings

        # Deduplicate
        seen = set()
        unique_findings = []
        for f in relevant_findings:
            key = str(f)
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)

        if unique_findings:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "found": True,
                    "findings": unique_findings[:5],  # Limit to top 5
                    "total_found": len(unique_findings)
                }
            )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "found": False,
                "message": f"No specific information found for: '{question}'. Domain knowledge exists but doesn't address this question directly."
            }
        )

    def _is_relevant(self, question: str, text: str) -> bool:
        """Check if text is relevant to the question using keyword matching."""
        # Extract key terms from question
        stop_words = {"what", "is", "the", "are", "there", "any", "how", "does", "do", "can", "which", "a", "an"}
        question_terms = set(question.lower().split()) - stop_words
        text_terms = set(text.lower().split())

        # Check for overlap
        overlap = question_terms & text_terms
        return len(overlap) >= 1

    async def _get_column_info(
        self,
        state: AnalysisState,
        column: str
    ) -> ToolResult:
        """Get detailed information about a specific column."""
        profile = state.data_profile

        if not profile:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Data profile not available yet. Run data profiler first."
            )

        if column not in profile.feature_names:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Column '{column}' not found. Available: {profile.feature_names[:10]}..."
            )

        info = {
            "name": column,
            "dtype": profile.feature_types.get(column, "unknown"),
            "missing_count": profile.missing_values.get(column, 0),
            "missing_pct": round(profile.missing_values.get(column, 0) / profile.n_samples * 100, 2)
        }

        # Add numeric stats if available
        if column in profile.numeric_stats:
            info["stats"] = profile.numeric_stats[column]

        # Add categorical stats if available
        if column in profile.categorical_stats:
            info["value_counts"] = profile.categorical_stats[column]

        # Check if it's a treatment/outcome candidate
        if column in profile.treatment_candidates:
            info["is_treatment_candidate"] = True
        if column in profile.outcome_candidates:
            info["is_outcome_candidate"] = True
        if column in profile.potential_confounders:
            info["is_potential_confounder"] = True

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=info
        )

    async def _get_dataset_summary(
        self,
        state: AnalysisState
    ) -> ToolResult:
        """Get basic dataset summary."""
        profile = state.data_profile

        if not profile:
            # Return what we know from dataset_info
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "profiled": False,
                    "name": state.dataset_info.name,
                    "n_samples": state.dataset_info.n_samples,
                    "n_features": state.dataset_info.n_features,
                    "message": "Full profile not available yet."
                }
            )

        # Count types
        type_counts = {}
        for dtype in profile.feature_types.values():
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "profiled": True,
                "n_samples": profile.n_samples,
                "n_features": profile.n_features,
                "type_counts": type_counts,
                "total_missing": sum(profile.missing_values.values()),
                "treatment_candidates": profile.treatment_candidates,
                "outcome_candidates": profile.outcome_candidates,
                "has_time_dimension": profile.has_time_dimension
            }
        )

    async def _list_columns(
        self,
        state: AnalysisState
    ) -> ToolResult:
        """Get list of all column names."""
        profile = state.data_profile

        if not profile:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Data profile not available yet."
            )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "columns": profile.feature_names,
                "count": len(profile.feature_names)
            }
        )

    async def _get_eda_finding(
        self,
        state: AnalysisState,
        topic: str
    ) -> ToolResult:
        """Get EDA findings for a specific topic."""
        eda = state.eda_result

        if not eda:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="EDA results not available yet. Run EDA agent first."
            )

        topic_handlers = {
            "outliers": self._get_outlier_findings,
            "correlations": self._get_correlation_findings,
            "balance": self._get_balance_findings,
            "missing": self._get_missing_findings,
            "multicollinearity": self._get_multicollinearity_findings,
            "quality_score": self._get_quality_findings,
            "distributions": self._get_distribution_findings,
        }

        handler = topic_handlers.get(topic)
        if handler:
            return handler(eda)

        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Unknown topic: {topic}"
        )

    def _get_outlier_findings(self, eda) -> ToolResult:
        """Extract outlier findings from EDA."""
        if not eda.outliers:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"has_outliers": False, "message": "No outlier analysis available or no outliers found."}
            )

        # Summarize - don't dump everything
        outlier_summary = []
        for col, info in eda.outliers.items():
            if isinstance(info, dict) and info.get("iqr_outliers", 0) > 0:
                outlier_summary.append({
                    "column": col,
                    "count": info.get("iqr_outliers", 0),
                    "pct": info.get("iqr_pct", 0)
                })

        outlier_summary.sort(key=lambda x: x["count"], reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "has_outliers": len(outlier_summary) > 0,
                "columns_with_outliers": len(outlier_summary),
                "top_outlier_columns": outlier_summary[:5]
            }
        )

    def _get_correlation_findings(self, eda) -> ToolResult:
        """Extract correlation findings from EDA."""
        if not eda.high_correlations:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"has_high_correlations": False, "message": "No high correlations found."}
            )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "has_high_correlations": True,
                "count": len(eda.high_correlations),
                "top_correlations": eda.high_correlations[:5]
            }
        )

    def _get_balance_findings(self, eda) -> ToolResult:
        """Extract covariate balance findings from EDA."""
        if not eda.covariate_balance:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"checked": False, "message": "Covariate balance not checked yet."}
            )

        # Find imbalanced covariates
        imbalanced = []
        for cov, info in eda.covariate_balance.items():
            if isinstance(info, dict):
                smd = info.get("smd", 0)
                if smd >= 0.1:
                    imbalanced.append({"covariate": cov, "smd": smd})

        imbalanced.sort(key=lambda x: x["smd"], reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "checked": True,
                "total_covariates": len(eda.covariate_balance),
                "imbalanced_count": len(imbalanced),
                "imbalanced_covariates": imbalanced[:5],
                "summary": eda.balance_summary
            }
        )

    def _get_missing_findings(self, eda) -> ToolResult:
        """Extract missing data findings."""
        issues = [i for i in eda.data_quality_issues if "missing" in i.lower()]

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "missing_issues": issues,
                "quality_score": eda.data_quality_score
            }
        )

    def _get_multicollinearity_findings(self, eda) -> ToolResult:
        """Extract multicollinearity findings from EDA."""
        if not eda.vif_scores:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"checked": False, "message": "VIF not calculated yet."}
            )

        # Find problematic VIFs
        high_vif = [(col, vif) for col, vif in eda.vif_scores.items() if vif > 5]
        high_vif.sort(key=lambda x: x[1], reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "checked": True,
                "has_multicollinearity": len(high_vif) > 0,
                "high_vif_columns": [{"column": c, "vif": v} for c, v in high_vif[:5]],
                "warnings": eda.multicollinearity_warnings
            }
        )

    def _get_quality_findings(self, eda) -> ToolResult:
        """Extract data quality score and issues."""
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "quality_score": eda.data_quality_score,
                "issues": eda.data_quality_issues,
                "summary": eda.summary_table.get("causal_readiness", "unknown")
            }
        )

    def _get_distribution_findings(self, eda) -> ToolResult:
        """Extract distribution findings."""
        if not eda.distribution_stats:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"checked": False, "message": "Distribution analysis not available."}
            )

        # Summarize key findings
        skewed = []
        for col, stats in eda.distribution_stats.items():
            if isinstance(stats, dict):
                skewness = stats.get("skewness", 0)
                if abs(skewness) > 1:
                    skewed.append({"column": col, "skewness": skewness})

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "checked": True,
                "columns_analyzed": len(eda.distribution_stats),
                "highly_skewed": skewed[:5]
            }
        )

    async def _get_previous_finding(
        self,
        state: AnalysisState,
        agent: str
    ) -> ToolResult:
        """Get key findings from a previous agent."""

        if agent == "domain_knowledge":
            if not state.domain_knowledge:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"available": False, "message": "Domain knowledge agent hasn't run yet."}
                )

            dk = state.domain_knowledge
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": True,
                    "hypotheses_count": len(dk.get("hypotheses", [])),
                    "uncertainties_count": len(dk.get("uncertainties", [])),
                    "key_hypotheses": dk.get("hypotheses", [])[:3],
                    "key_uncertainties": dk.get("uncertainties", [])[:2]
                }
            )

        elif agent == "data_profiler":
            if not state.data_profile:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"available": False, "message": "Data profiler hasn't run yet."}
                )

            profile = state.data_profile
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": True,
                    "treatment_candidates": profile.treatment_candidates,
                    "outcome_candidates": profile.outcome_candidates,
                    "potential_confounders": profile.potential_confounders[:5],
                    "n_samples": profile.n_samples,
                    "has_time": profile.has_time_dimension
                }
            )

        elif agent == "eda_agent":
            if not state.eda_result:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"available": False, "message": "EDA agent hasn't run yet."}
                )

            eda = state.eda_result
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": True,
                    "quality_score": eda.data_quality_score,
                    "key_issues": eda.data_quality_issues[:3],
                    "balance_summary": eda.balance_summary,
                    "has_multicollinearity": len(eda.multicollinearity_warnings) > 0
                }
            )

        elif agent == "causal_discovery":
            if not state.proposed_dag:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"available": False, "message": "Causal discovery hasn't run yet."}
                )

            dag = state.proposed_dag
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": True,
                    "method": dag.discovery_method,
                    "n_nodes": len(dag.nodes),
                    "n_edges": len(dag.edges),
                    "treatment": dag.treatment_variable,
                    "outcome": dag.outcome_variable,
                    "interpretation": dag.interpretation[:200] if dag.interpretation else None
                }
            )

        elif agent == "effect_estimator":
            if not state.treatment_effects:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"available": False, "message": "Effect estimation hasn't run yet."}
                )

            effects = state.treatment_effects
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": True,
                    "n_estimates": len(effects),
                    "methods_used": list(set(e.method for e in effects)),
                    "estimates": [
                        {
                            "method": e.method,
                            "estimate": e.estimate,
                            "ci": [e.ci_lower, e.ci_upper]
                        }
                        for e in effects[:3]
                    ]
                }
            )

        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Unknown agent: {agent}"
        )

    async def _get_treatment_outcome(
        self,
        state: AnalysisState
    ) -> ToolResult:
        """Get current treatment and outcome variables."""
        treatment, outcome = state.get_primary_pair()

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "treatment": treatment,
                "outcome": outcome,
                "confirmed": treatment is not None and outcome is not None,
                "source": "user_specified" if state.treatment_variable else "inferred"
            }
        )

    async def _get_confounder_analysis(
        self,
        state: AnalysisState,
        top_n: int = 10
    ) -> ToolResult:
        """Get confounder analysis summary."""
        # Check for detailed confounder discovery results
        if state.confounder_discovery:
            cd = state.confounder_discovery
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "source": "confounder_discovery",
                    "ranked_confounders": cd.get("ranked_confounders", [])[:top_n],
                    "adjustment_strategy": cd.get("adjustment_strategy", ""),
                    "total_identified": len(cd.get("ranked_confounders", [])),
                    "excluded_variables": cd.get("excluded_variables", [])[:5]
                }
            )

        # Fall back to profile confounders if discovery not run
        if state.data_profile and state.data_profile.potential_confounders:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "source": "data_profile",
                    "confounders": state.data_profile.potential_confounders[:top_n],
                    "total_identified": len(state.data_profile.potential_confounders),
                    "strategy": "Include as covariates in regression/matching"
                }
            )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "available": False,
                "message": "Confounder analysis not run. Use all pre-treatment covariates."
            }
        )

    async def _get_profile_for_variables(
        self,
        state: AnalysisState,
        variables: list[str]
    ) -> ToolResult:
        """Get profile stats for specific variables only."""
        if not state.data_profile:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Data profile not available"
            )

        profile = state.data_profile
        result: dict = {"variables": {}}

        for var in variables:
            if var not in profile.feature_names:
                result["variables"][var] = {"error": "not found"}
                continue

            var_info: dict = {
                "dtype": profile.feature_types.get(var, "unknown"),
                "missing": profile.missing_values.get(var, 0),
                "missing_pct": round(
                    profile.missing_values.get(var, 0) / profile.n_samples * 100, 2
                )
            }

            # Add numeric stats if available
            if var in profile.numeric_stats:
                var_info["stats"] = profile.numeric_stats[var]

            # Add categorical stats if available (limited to top 5)
            if var in profile.categorical_stats:
                cat_stats = profile.categorical_stats[var]
                var_info["n_categories"] = len(cat_stats)
                var_info["top_categories"] = dict(list(cat_stats.items())[:5])

            result["variables"][var] = var_info

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result
        )

    async def _get_dataset_context(self, state: AnalysisState) -> ToolResult:
        """Get semantic dataset context from metadata."""
        # Check for Kaggle metadata in state
        if not state.raw_metadata and not state.dataset_info.kaggle_description:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": False,
                    "message": "No dataset metadata available. Interpret variables from names only."
                }
            )

        # Build context from raw_metadata or dataset_info
        metadata = state.raw_metadata or {}

        context = {
            "dataset_title": metadata.get("title") or state.dataset_info.name,
            "domain_tags": metadata.get("tags") or state.dataset_info.kaggle_tags,
            "description_summary": None,
            "column_descriptions": metadata.get("column_descriptions") or state.dataset_info.kaggle_column_descriptions,
            "study_type_hints": self._infer_study_type(metadata, state.dataset_info),
            "metadata_quality": metadata.get("metadata_quality") or state.dataset_info.metadata_quality,
            "inferred_domain": state.dataset_info.kaggle_domain
        }

        # Add description summary (truncated)
        desc = metadata.get("description") or state.dataset_info.kaggle_description
        if desc:
            context["description_summary"] = desc[:500] if len(desc) > 500 else desc

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=context
        )

    def _infer_study_type(self, metadata: dict, dataset_info) -> list[str]:
        """Infer study type from metadata."""
        hints = []
        desc = (
            (metadata.get("description", "") or "") + " " +
            (metadata.get("subtitle", "") or "")
        ).lower()
        tags = [t.lower() for t in (metadata.get("tags") or dataset_info.kaggle_tags or [])]

        if any(w in desc for w in ["randomized", "rct", "experiment", "randomly assigned"]):
            hints.append("Likely RCT/Experimental")
        if any(w in desc for w in ["observational", "survey", "registry"]):
            hints.append("Likely Observational")
        if any(w in desc for w in ["panel", "longitudinal", "time series"]):
            hints.append("Panel/Longitudinal data")
        if any(t in tags for t in ["healthcare", "medical", "clinical"]):
            hints.append("Healthcare domain")
        if any(t in tags for t in ["economics", "finance", "income"]):
            hints.append("Economics domain")
        if any(t in tags for t in ["education", "academic", "school"]):
            hints.append("Education domain")

        return hints if hints else ["Study type unclear from metadata"]

    async def _analyze_variable_semantics(
        self,
        state: AnalysisState,
        variable: str
    ) -> ToolResult:
        """Analyze semantic meaning of a variable."""
        var_lower = variable.lower()

        # Start with metadata description if available
        description = None
        if state.raw_metadata:
            col_descs = state.raw_metadata.get("column_descriptions", {})
            description = col_descs.get(variable)
        elif state.dataset_info.kaggle_column_descriptions:
            description = state.dataset_info.kaggle_column_descriptions.get(variable)

        # Build semantic analysis
        analysis: dict = {
            "variable": variable,
            "description": description,
            "likely_role": [],
            "is_likely_immutable": False,
            "temporal_position": "unknown",  # pre, post, unclear
            "causal_constraints": []
        }

        # Demographic/Immutable detection
        immutable_indicators = {
            "age": "Demographic - cannot be caused by treatment",
            "sex": "Demographic - cannot be caused by treatment",
            "gender": "Demographic - cannot be caused by treatment",
            "race": "Demographic - cannot be caused by treatment",
            "black": "Race indicator - immutable demographic",
            "white": "Race indicator - immutable demographic",
            "hisp": "Hispanic indicator - immutable demographic",
            "hispanic": "Hispanic indicator - immutable demographic",
            "asian": "Race indicator - immutable demographic",
            "ethnicity": "Demographic - cannot be caused by treatment",
            "birth": "Birth-related - immutable",
            "dob": "Date of birth - immutable",
            "married": "Marital status - typically pre-treatment",
            "education": "Education level - typically pre-treatment",
            "educ": "Education level - typically pre-treatment",
            "nodegree": "Education indicator - typically pre-treatment",
        }

        for indicator, meaning in immutable_indicators.items():
            if indicator in var_lower:
                analysis["is_likely_immutable"] = True
                analysis["likely_role"].append("confounder")
                analysis["temporal_position"] = "pre"
                analysis["causal_constraints"].append(meaning)
                break

        # Treatment indicators
        treatment_indicators = ["treat", "intervention", "program", "policy", "exposed", "assigned"]
        if any(ind in var_lower for ind in treatment_indicators):
            analysis["likely_role"].append("treatment")

        # Outcome indicators
        outcome_indicators = {
            "outcome": "Explicit outcome",
            "result": "Result variable",
            "earnings": "Economic outcome",
            "income": "Economic outcome",
            "wage": "Economic outcome",
            "salary": "Economic outcome",
            "score": "Measurement outcome",
            "effect": "Effect measurement",
            "response": "Response variable",
        }
        for indicator, meaning in outcome_indicators.items():
            if indicator in var_lower:
                analysis["likely_role"].append("outcome")
                analysis["temporal_position"] = "post"
                break

        # Year-prefixed variables (e.g., re74, re75, re78)
        year_match = re.search(r'(\d{2,4})$', variable)
        if year_match:
            year = year_match.group(1)
            if len(year) == 2:
                year = "19" + year if int(year) > 50 else "20" + year
            analysis["temporal_hint"] = f"Year {year} - check against treatment timing"

        # Use domain knowledge if available
        if state.domain_knowledge:
            dk = state.domain_knowledge
            if variable in dk.get("immutable_vars", []):
                analysis["is_likely_immutable"] = True
                analysis["causal_constraints"].append("Marked immutable by domain knowledge agent")

        # Default if no role identified
        if not analysis["likely_role"]:
            analysis["likely_role"] = ["potential_covariate"]

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=analysis
        )

    async def _get_dag_adjustment_set(
        self,
        state: AnalysisState,
        treatment: str | None = None,
        outcome: str | None = None
    ) -> ToolResult:
        """
        Get proper adjustment set from causal DAG using backdoor criterion.

        The backdoor criterion identifies variables that:
        1. Block all backdoor paths between treatment and outcome
        2. Do NOT include descendants of treatment (would block causal effect)
        3. Do NOT include colliders (would open new confounding paths)

        Returns adjustment set, mediators to avoid, and colliders.
        """
        # Get treatment/outcome from state if not provided
        if not treatment:
            treatment = state.treatment_variable
        if not outcome:
            outcome = state.outcome_variable

        if not treatment or not outcome:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Treatment and outcome variables must be specified or available in state"
            )

        # Check for DAG
        dag = state.proposed_dag
        if not dag:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": False,
                    "message": "No causal DAG available. Use confounder analysis as fallback.",
                    "fallback_confounders": (
                        state.data_profile.potential_confounders[:10]
                        if state.data_profile else []
                    )
                }
            )

        # Build adjacency structures for DAG analysis
        nodes = set(dag.nodes)
        if treatment not in nodes or outcome not in nodes:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Treatment '{treatment}' or outcome '{outcome}' not in DAG nodes"
            )

        # Build parent/child maps
        parents: dict[str, set[str]] = {n: set() for n in nodes}
        children: dict[str, set[str]] = {n: set() for n in nodes}

        for edge in dag.edges:
            if edge.edge_type == "directed":
                parents[edge.target].add(edge.source)
                children[edge.source].add(edge.target)

        # Find ancestors of treatment and outcome
        def get_ancestors(node: str, visited: set | None = None) -> set[str]:
            if visited is None:
                visited = set()
            if node in visited:
                return set()
            visited.add(node)
            ancestors = set()
            for parent in parents.get(node, set()):
                ancestors.add(parent)
                ancestors |= get_ancestors(parent, visited)
            return ancestors

        # Find descendants of treatment (mediators - should NOT adjust)
        def get_descendants(node: str, visited: set | None = None) -> set[str]:
            if visited is None:
                visited = set()
            if node in visited:
                return set()
            visited.add(node)
            descendants = set()
            for child in children.get(node, set()):
                descendants.add(child)
                descendants |= get_descendants(child, visited)
            return descendants

        treatment_ancestors = get_ancestors(treatment)
        outcome_ancestors = get_ancestors(outcome)
        treatment_descendants = get_descendants(treatment)

        # Backdoor criterion adjustment set:
        # Variables that are ancestors of BOTH treatment and outcome (common causes)
        # excluding treatment, outcome, and descendants of treatment
        adjustment_set = (treatment_ancestors & outcome_ancestors) - {treatment, outcome}

        # Mediators: descendants of treatment that are ancestors of outcome
        mediators = treatment_descendants & outcome_ancestors
        mediators -= {outcome}  # outcome itself is not a mediator

        # Colliders: nodes where two arrows point in (X -> C <- Y)
        # These should NOT be adjusted for as it opens confounding paths
        colliders = set()
        for node in nodes:
            if len(parents.get(node, set())) >= 2:
                # Check if it's on a path between treatment-related and outcome-related nodes
                node_parents = parents.get(node, set())
                # If one parent is treatment (or its descendant) and another is related to outcome
                if (
                    node_parents & (treatment_descendants | {treatment}) and
                    node_parents & (outcome_ancestors | {outcome})
                ):
                    colliders.add(node)

        # Additional confounders from domain knowledge
        domain_confounders = set()
        if state.domain_knowledge:
            dk = state.domain_knowledge
            # Add immutable variables as they're likely confounders
            for var in dk.get("immutable_vars", []):
                if var in nodes and var not in {treatment, outcome}:
                    domain_confounders.add(var)

        # Final adjustment set
        final_adjustment = adjustment_set | domain_confounders
        final_adjustment -= mediators  # Don't adjust for mediators
        final_adjustment -= colliders  # Don't adjust for colliders

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "treatment": treatment,
                "outcome": outcome,
                "adjustment_set": sorted(final_adjustment),
                "adjustment_rationale": (
                    "Variables that block backdoor paths (common causes of treatment and outcome)"
                ),
                "mediators_to_avoid": sorted(mediators),
                "mediator_rationale": (
                    "Descendants of treatment that are ancestors of outcome - adjusting would "
                    "block part of the causal effect"
                ),
                "colliders_warning": sorted(colliders),
                "collider_rationale": (
                    "Variables with multiple causes - adjusting could open confounding paths"
                ),
                "dag_method": dag.discovery_method,
                "total_dag_nodes": len(nodes),
                "total_dag_edges": len(dag.edges)
            }
        )
