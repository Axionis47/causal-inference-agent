"""DAG Expert Agent - Domain-informed causal graph construction.

This agent acts as a domain expert by:
1. Analyzing dataset metadata to understand the domain
2. Using LLM reasoning to infer causal relationships
3. Fusing domain knowledge with data-driven discovery
4. Producing a validated, high-quality DAG for causal analysis

The key insight: Neither pure data discovery nor pure assumptions work alone.
We need to combine domain expertise (from metadata + LLM) with statistical evidence.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import (
    AnalysisState,
    CausalDAG,
    CausalEdge,
    ReActAgent,
    ToolResult,
    ToolResultStatus,
)
from src.agents.base.context_tools import ContextTools
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class DAGExpertAgent(ReActAgent, ContextTools):
    """Domain expert agent that constructs validated causal DAGs.

    This agent combines:
    1. Domain knowledge from Kaggle metadata
    2. LLM reasoning about causal relationships
    3. Data-driven discovery results
    4. Semantic variable analysis

    The output is a high-confidence DAG with:
    - Edge confidence scores
    - Source attribution (domain vs data)
    - Warnings about assumptions
    """

    AGENT_NAME = "dag_expert"
    MAX_STEPS = 12

    SYSTEM_PROMPT = """You are a domain expert in causal inference, acting as a consultant
who designs causal DAGs based on domain knowledge and data evidence.

Your role is to construct a VALIDATED causal DAG by:
1. Understanding the domain from dataset metadata
2. Identifying variable roles (treatment, outcome, confounders, mediators, colliders)
3. Reasoning about causal relationships based on domain logic
4. Incorporating data-driven discovery as supporting evidence
5. Resolving conflicts between domain knowledge and data patterns

KEY PRINCIPLES:
- Domain knowledge takes precedence over data patterns for edge DIRECTION
- Data patterns help confirm or question domain assumptions
- Demographics (age, race, gender) are ALWAYS pre-treatment confounders
- Treatment cannot cause pre-treatment variables (temporal logic)
- Be explicit about assumptions and confidence levels

FORBIDDEN PATTERNS (domain knowledge):
- Outcome → Treatment (reverse causality)
- Treatment → Demographics (impossible)
- Post-treatment variable → Pre-treatment variable

OUTPUT: A validated DAG with confidence scores for each edge."""

    def __init__(self) -> None:
        """Initialize the DAG expert agent."""
        super().__init__()
        self.register_context_tools()
        self._domain_edges: list[dict] = []
        self._data_edges: list[dict] = []
        self._forbidden_edges: list[tuple[str, str]] = []
        self._required_edges: list[tuple[str, str]] = []
        self._variable_roles: dict[str, str] = {}
        self._register_dag_tools()

    def _register_dag_tools(self) -> None:
        """Register DAG construction tools."""

        self.register_tool(
            name="analyze_domain",
            description=(
                "Analyze the dataset domain from metadata to understand "
                "the causal context. Returns domain type, typical causal patterns, "
                "and variable role hints."
            ),
            parameters={"type": "object", "properties": {}},
            handler=self._analyze_domain,
        )

        self.register_tool(
            name="classify_variable_role",
            description=(
                "Classify a variable's causal role based on domain knowledge "
                "and metadata. Returns: treatment, outcome, confounder, mediator, "
                "collider, instrument, or covariate."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Variable name to classify",
                    },
                },
                "required": ["variable"],
            },
            handler=self._classify_variable_role,
        )

        self.register_tool(
            name="propose_edge",
            description=(
                "Propose a causal edge based on domain reasoning. "
                "Specify source, target, reasoning, and confidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Cause variable"},
                    "target": {"type": "string", "description": "Effect variable"},
                    "reasoning": {
                        "type": "string",
                        "description": "Domain reasoning for this edge",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence based on domain knowledge",
                    },
                },
                "required": ["source", "target", "reasoning", "confidence"],
            },
            handler=self._propose_edge,
        )

        self.register_tool(
            name="mark_forbidden_edge",
            description=(
                "Mark an edge as forbidden based on domain logic "
                "(e.g., outcome cannot cause treatment)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "reason": {"type": "string", "description": "Why this edge is impossible"},
                },
                "required": ["source", "target", "reason"],
            },
            handler=self._mark_forbidden_edge,
        )

        self.register_tool(
            name="get_discovery_edges",
            description=(
                "Get edges from data-driven causal discovery. "
                "Use to compare with domain-proposed edges."
            ),
            parameters={"type": "object", "properties": {}},
            handler=self._get_discovery_edges,
        )

        self.register_tool(
            name="fuse_and_validate",
            description=(
                "Fuse domain edges with discovery edges to create final DAG. "
                "Resolves conflicts, applies forbidden edge constraints, "
                "and assigns final confidence scores."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "conflict_resolution": {
                        "type": "string",
                        "enum": ["domain_priority", "data_priority", "consensus_only"],
                        "description": "How to resolve conflicts between domain and data",
                    },
                },
                "required": ["conflict_resolution"],
            },
            handler=self._fuse_and_validate,
        )

        self.register_tool(
            name="get_adjustment_set",
            description=(
                "Get the proper adjustment set for estimating the causal effect "
                "based on the validated DAG using backdoor criterion."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "treatment": {"type": "string"},
                    "outcome": {"type": "string"},
                },
                "required": ["treatment", "outcome"],
            },
            handler=self._get_adjustment_set,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Get initial observation for DAG construction."""
        obs = f"""Task: Construct a validated causal DAG for job {state.job_id}

Dataset: {state.dataset_info.name or state.dataset_info.url}
Treatment: {state.treatment_variable or "To be identified"}
Outcome: {state.outcome_variable or "To be identified"}

Your goal is to build a high-quality DAG by:
1. First, analyze the domain from metadata (analyze_domain)
2. Classify key variables into causal roles (classify_variable_role)
3. Propose edges based on domain reasoning (propose_edge)
4. Mark impossible edges as forbidden (mark_forbidden_edge)
5. Compare with data discovery results (get_discovery_edges)
6. Fuse domain and data to create final DAG (fuse_and_validate)
7. Extract the adjustment set for effect estimation (get_adjustment_set)

Start by analyzing the domain to understand the causal context."""

        return obs

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if DAG construction is complete."""
        return (
            state.proposed_dag is not None
            and len(self._variable_roles) > 0
        )

    async def _analyze_domain(self, state: AnalysisState) -> ToolResult:
        """Analyze domain from metadata."""
        domain_info = {
            "domain": state.dataset_info.kaggle_domain or "unknown",
            "tags": state.dataset_info.kaggle_tags or [],
            "description": None,
            "typical_patterns": [],
            "variable_hints": {},
        }

        # Get description
        if state.dataset_info.kaggle_description:
            domain_info["description"] = state.dataset_info.kaggle_description[:500]
        elif state.raw_metadata:
            desc = state.raw_metadata.get("description", "")
            domain_info["description"] = desc[:500] if desc else None

        # Infer typical causal patterns by domain
        domain = domain_info["domain"]
        if domain == "economics":
            domain_info["typical_patterns"] = [
                "Demographics → Treatment selection → Economic outcome",
                "Education → Employment → Earnings",
                "Policy/Program → Behavior change → Outcome",
            ]
        elif domain == "healthcare":
            domain_info["typical_patterns"] = [
                "Risk factors → Treatment → Health outcome",
                "Demographics → Disease risk → Treatment → Survival",
                "Lifestyle → Biomarkers → Disease",
            ]
        elif domain == "education":
            domain_info["typical_patterns"] = [
                "Background → Educational intervention → Academic outcome",
                "Prior achievement → Program participation → Future achievement",
            ]
        else:
            domain_info["typical_patterns"] = [
                "Pre-treatment covariates → Treatment → Outcome",
                "Confounders affect both treatment and outcome",
            ]

        # Extract variable hints from column descriptions
        if state.dataset_info.kaggle_column_descriptions:
            for col, desc in state.dataset_info.kaggle_column_descriptions.items():
                domain_info["variable_hints"][col] = desc

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=domain_info,
        )

    async def _classify_variable_role(
        self, state: AnalysisState, variable: str
    ) -> ToolResult:
        """Classify a variable's causal role."""
        var_lower = variable.lower()

        # Get semantic analysis
        semantic_result = await self._analyze_variable_semantics(state, variable)
        semantic = semantic_result.output if semantic_result.status == ToolResultStatus.SUCCESS else {}

        role = "covariate"  # default
        confidence = "medium"
        reasoning = []

        # Check if it's the designated treatment/outcome
        if state.treatment_variable and variable == state.treatment_variable:
            role = "treatment"
            confidence = "high"
            reasoning.append("Designated as treatment variable")
        elif state.outcome_variable and variable == state.outcome_variable:
            role = "outcome"
            confidence = "high"
            reasoning.append("Designated as outcome variable")

        # Check semantic indicators
        elif semantic.get("is_likely_immutable"):
            role = "confounder"
            confidence = "high"
            reasoning.append(f"Immutable variable: {semantic.get('causal_constraints', [])}")

        elif "treatment" in semantic.get("likely_role", []):
            role = "treatment_candidate"
            confidence = "medium"
            reasoning.append("Name suggests treatment")

        elif "outcome" in semantic.get("likely_role", []):
            role = "outcome_candidate"
            confidence = "medium"
            reasoning.append("Name suggests outcome")

        # Temporal analysis
        temporal = semantic.get("temporal_position", "unknown")
        if temporal == "pre":
            if role == "covariate":
                role = "confounder"
            reasoning.append("Pre-treatment timing")
        elif temporal == "post":
            if role == "covariate":
                role = "potential_mediator"
            reasoning.append("Post-treatment timing")

        # Store the classification
        self._variable_roles[variable] = role

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "variable": variable,
                "role": role,
                "confidence": confidence,
                "reasoning": reasoning,
                "semantic_analysis": semantic,
            },
        )

    async def _propose_edge(
        self,
        state: AnalysisState,
        source: str,
        target: str,
        reasoning: str,
        confidence: str,
    ) -> ToolResult:
        """Propose a causal edge based on domain reasoning."""
        # Check if edge is forbidden
        if (source, target) in self._forbidden_edges:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Edge {source} → {target} is marked as forbidden",
            )

        # Validate based on roles
        source_role = self._variable_roles.get(source, "unknown")
        target_role = self._variable_roles.get(target, "unknown")

        warnings = []

        # Check for impossible patterns
        if target_role == "confounder" and source_role in ["treatment", "outcome"]:
            warnings.append("Warning: Treatment/outcome cannot cause a confounder (pre-treatment)")

        if source_role == "outcome" and target_role == "treatment":
            warnings.append("Warning: Outcome → Treatment suggests reverse causality")

        edge = {
            "source": source,
            "target": target,
            "reasoning": reasoning,
            "confidence": confidence,
            "source_type": "domain",
            "warnings": warnings,
        }

        self._domain_edges.append(edge)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "edge_added": f"{source} → {target}",
                "confidence": confidence,
                "warnings": warnings,
                "total_domain_edges": len(self._domain_edges),
            },
        )

    async def _mark_forbidden_edge(
        self, state: AnalysisState, source: str, target: str, reason: str
    ) -> ToolResult:
        """Mark an edge as forbidden."""
        self._forbidden_edges.append((source, target))

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "forbidden": f"{source} → {target}",
                "reason": reason,
                "total_forbidden": len(self._forbidden_edges),
            },
        )

    async def _get_discovery_edges(self, state: AnalysisState) -> ToolResult:
        """Get edges from data-driven discovery."""
        if not state.proposed_dag:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "available": False,
                    "message": "No data-driven DAG available. Using domain knowledge only.",
                },
            )

        dag = state.proposed_dag
        discovery_edges = []

        for edge in dag.edges:
            edge_info = {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
                "confidence": edge.confidence,
                "source_type": "data_discovery",
            }
            discovery_edges.append(edge_info)
            self._data_edges.append(edge_info)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "discovery_method": dag.discovery_method,
                "n_edges": len(discovery_edges),
                "edges": discovery_edges[:20],  # Limit for readability
                "interpretation": dag.interpretation,
            },
        )

    async def _fuse_and_validate(
        self, state: AnalysisState, conflict_resolution: str
    ) -> ToolResult:
        """Fuse domain and data edges to create final DAG."""
        final_edges: list[CausalEdge] = []
        conflicts = []
        edge_sources: dict[tuple[str, str], str] = {}

        # Collect all nodes
        all_nodes = set()
        for edge in self._domain_edges + self._data_edges:
            all_nodes.add(edge["source"])
            all_nodes.add(edge["target"])

        # Add treatment and outcome
        if state.treatment_variable:
            all_nodes.add(state.treatment_variable)
        if state.outcome_variable:
            all_nodes.add(state.outcome_variable)

        # Process domain edges first (higher priority for direction)
        domain_edge_set = {(e["source"], e["target"]) for e in self._domain_edges}
        data_edge_set = {(e["source"], e["target"]) for e in self._data_edges}

        # Find conflicts (same nodes, different direction)
        for s, t in domain_edge_set:
            if (t, s) in data_edge_set:
                conflicts.append({
                    "domain": f"{s} → {t}",
                    "data": f"{t} → {s}",
                    "resolution": conflict_resolution,
                })

        # Build final edge list based on resolution strategy
        processed = set()

        for edge in self._domain_edges:
            key = (edge["source"], edge["target"])
            reverse_key = (edge["target"], edge["source"])

            if key in self._forbidden_edges:
                continue

            if key in processed:
                continue

            # Check for conflict
            if reverse_key in data_edge_set:
                if conflict_resolution == "data_priority":
                    # Use data direction instead
                    key = reverse_key
                elif conflict_resolution == "consensus_only":
                    # Skip this edge
                    continue
                # else domain_priority: keep domain direction

            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
            final_edges.append(CausalEdge(
                source=key[0],
                target=key[1],
                edge_type="directed",
                confidence=confidence_map.get(edge.get("confidence", "medium"), 0.7),
            ))
            edge_sources[key] = "domain"
            processed.add(key)

        # Add data edges not in domain (if not forbidden)
        for edge in self._data_edges:
            key = (edge["source"], edge["target"])
            reverse_key = (edge["target"], edge["source"])

            if key in processed or reverse_key in processed:
                continue
            if key in self._forbidden_edges:
                continue

            final_edges.append(CausalEdge(
                source=edge["source"],
                target=edge["target"],
                edge_type=edge.get("edge_type", "directed"),
                confidence=edge.get("confidence", 0.6),
            ))
            edge_sources[key] = "data"
            processed.add(key)

        # Create the validated DAG
        validated_dag = CausalDAG(
            nodes=list(all_nodes),
            edges=final_edges,
            discovery_method=f"domain_expert_fusion ({conflict_resolution})",
            treatment_variable=state.treatment_variable,
            outcome_variable=state.outcome_variable,
            interpretation=self._generate_interpretation(
                final_edges, conflicts, edge_sources
            ),
        )

        # Update state
        state.proposed_dag = validated_dag

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_nodes": len(validated_dag.nodes),
                "n_edges": len(validated_dag.edges),
                "n_domain_edges": len([e for e in edge_sources.values() if e == "domain"]),
                "n_data_edges": len([e for e in edge_sources.values() if e == "data"]),
                "n_conflicts_resolved": len(conflicts),
                "conflicts": conflicts,
                "forbidden_edges_applied": len(self._forbidden_edges),
                "interpretation": validated_dag.interpretation[:500],
            },
        )

    def _generate_interpretation(
        self,
        edges: list[CausalEdge],
        conflicts: list[dict],
        edge_sources: dict,
    ) -> str:
        """Generate human-readable interpretation of the DAG."""
        lines = ["## Validated Causal DAG\n"]

        lines.append("### Edge Sources:")
        domain_count = sum(1 for v in edge_sources.values() if v == "domain")
        data_count = sum(1 for v in edge_sources.values() if v == "data")
        lines.append(f"- Domain-derived: {domain_count} edges")
        lines.append(f"- Data-derived: {data_count} edges")

        if conflicts:
            lines.append(f"\n### Conflicts Resolved: {len(conflicts)}")
            for c in conflicts[:3]:
                lines.append(f"- Domain said {c['domain']}, Data said {c['data']}")

        lines.append("\n### Key Assumptions:")
        lines.append("- Demographics are pre-treatment (cannot be caused by treatment)")
        lines.append("- Temporal ordering respected where identified")
        lines.append("- Unobserved confounders may exist (interpret with caution)")

        lines.append("\n### Recommendations:")
        lines.append("- Use sensitivity analysis to test robustness")
        lines.append("- Consider multiple adjustment strategies")

        return "\n".join(lines)

    async def _get_adjustment_set(
        self, state: AnalysisState, treatment: str, outcome: str
    ) -> ToolResult:
        """Get proper adjustment set using backdoor criterion."""
        if not state.proposed_dag:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No DAG available. Run fuse_and_validate first.",
            )

        dag = state.proposed_dag

        # Build adjacency for ancestor computation
        children: dict[str, set[str]] = {n: set() for n in dag.nodes}
        parents: dict[str, set[str]] = {n: set() for n in dag.nodes}

        for edge in dag.edges:
            if edge.edge_type == "directed":
                children[edge.source].add(edge.target)
                parents[edge.target].add(edge.source)

        # Compute ancestors
        def get_ancestors(node: str) -> set[str]:
            ancestors = set()
            queue = list(parents.get(node, []))
            while queue:
                p = queue.pop()
                if p not in ancestors:
                    ancestors.add(p)
                    queue.extend(parents.get(p, []))
            return ancestors

        # Compute descendants
        def get_descendants(node: str) -> set[str]:
            descendants = set()
            queue = list(children.get(node, []))
            while queue:
                c = queue.pop()
                if c not in descendants:
                    descendants.add(c)
                    queue.extend(children.get(c, []))
            return descendants

        t_ancestors = get_ancestors(treatment)
        o_ancestors = get_ancestors(outcome)
        t_descendants = get_descendants(treatment)

        # Confounders: ancestors of BOTH treatment and outcome
        confounders = t_ancestors & o_ancestors

        # Mediators: descendants of treatment AND ancestors of outcome
        mediators = t_descendants & o_ancestors

        # Colliders: descendants of BOTH (never adjust!)
        o_descendants = get_descendants(outcome)
        colliders = t_descendants & o_descendants

        # Adjustment set for total effect: confounders only (not mediators)
        adjustment_set = list(confounders - {treatment, outcome})

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "treatment": treatment,
                "outcome": outcome,
                "adjustment_set": adjustment_set,
                "confounders": list(confounders),
                "mediators": list(mediators),
                "colliders_do_not_adjust": list(colliders),
                "recommendation": (
                    f"Adjust for {adjustment_set} to block backdoor paths. "
                    f"Do NOT adjust for {list(mediators)} (mediators) or "
                    f"{list(colliders)} (colliders)."
                ),
            },
        )

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute DAG expert construction."""
        self.logger.info(
            "dag_expert_start",
            job_id=state.job_id,
            has_discovery_dag=state.proposed_dag is not None,
        )

        # Reset internal state
        self._domain_edges = []
        self._data_edges = []
        self._forbidden_edges = []
        self._variable_roles = {}

        # Run the ReAct loop
        state = await super().execute(state)

        self.logger.info(
            "dag_expert_complete",
            n_edges=len(state.proposed_dag.edges) if state.proposed_dag else 0,
            n_roles_classified=len(self._variable_roles),
        )

        return state
