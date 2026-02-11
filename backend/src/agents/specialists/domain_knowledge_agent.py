"""
Domain Knowledge Agent - ReAct agent for causal domain understanding.

This agent investigates dataset metadata to build causal understanding:
- Reads and analyzes description
- Investigates column semantics
- Forms hypotheses about treatment, outcome, confounders
- Identifies temporal ordering and immutable variables
- Flags uncertainties for downstream agents

Uses ReAct loop: Observe → Think → Act → Observe...
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AnalysisState, ReActAgent, ToolResult, ToolResultStatus
from src.agents.registry import register_agent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


@register_agent("domain_knowledge")
class DomainKnowledgeAgent(ReActAgent):
    """
    Investigates dataset metadata to build causal understanding.

    Uses ReAct loop to iteratively explore and hypothesize about:
    - What is the treatment variable?
    - What is the outcome variable?
    - What variables are immutable (can't be caused)?
    - What is the temporal ordering?
    - What are potential confounders?
    - What uncertainties exist?

    This agent does NOT look at the actual data - only metadata.
    """

    AGENT_NAME = "domain_knowledge"
    MAX_STEPS = 12

    SYSTEM_PROMPT = """You are a causal inference researcher investigating a new dataset.

Your job is to understand what this data is about and what causal questions it can answer.
You only have access to METADATA (description, column names, tags) - not the actual data.

Work like a detective:
1. Read the description carefully
2. Investigate column names to understand what they represent
3. Form hypotheses about treatment, outcome, and confounders
4. Look for temporal clues (what came before what)
5. Identify immutable variables (age, sex, race - things that can't be caused)
6. Flag uncertainties that downstream agents should know about

Key causal concepts:
- TREATMENT: The intervention/exposure (often binary: treated vs control)
- OUTCOME: What we're measuring the effect on
- CONFOUNDERS: Variables that affect BOTH treatment and outcome
- IMMUTABLE: Variables that can't be caused by others (demographics, pre-study characteristics)
- TEMPORAL ORDER: Treatment must come before outcome; confounders before treatment

Be curious. Question your assumptions. Revise hypotheses when you find new evidence.
When confident, call finish with your findings.
"""

    def __init__(self):
        """Initialize the domain knowledge agent."""
        super().__init__()
        self._hypotheses: list[dict[str, Any]] = []
        self._uncertainties: list[dict[str, Any]] = []
        self._temporal_understanding: str | None = None
        self._immutable_vars: list[str] = []

        # Register investigation tools
        self._register_investigation_tools()

    def _register_investigation_tools(self) -> None:
        """Register tools for investigating metadata."""

        self.register_tool(
            name="read_description",
            description=(
                "Read the dataset description. This often contains crucial information "
                "about what the study is, how data was collected, and what variables mean. "
                "Call this FIRST."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._read_description,
        )

        self.register_tool(
            name="list_columns",
            description=(
                "Get list of all column names in the dataset. "
                "Column names often reveal what variables represent."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._list_columns,
        )

        self.register_tool(
            name="investigate_column",
            description=(
                "Investigate what a specific column might mean based on its name "
                "and any available description. Use this to understand potential "
                "treatment, outcome, or confounder variables."
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
            handler=self._investigate_column,
        )

        self.register_tool(
            name="search_metadata",
            description=(
                "Search the metadata for specific keywords or phrases. "
                "Use this to find evidence for your hypotheses. "
                "Examples: 'random', 'treatment', 'outcome', 'baseline', 'before', 'after'"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or phrase to search for"
                    }
                },
                "required": ["query"]
            },
            handler=self._search_metadata,
        )

        self.register_tool(
            name="get_tags",
            description=(
                "Get the dataset tags/categories. Tags indicate the domain "
                "(healthcare, economics, etc.) which helps interpret variables."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._get_tags,
        )

        self.register_tool(
            name="hypothesize",
            description=(
                "Record a hypothesis about the causal structure. "
                "You can have multiple hypotheses. They can be revised later."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "Your hypothesis about a causal relationship in this dataset"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How confident you are"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "What evidence supports this hypothesis"
                    }
                },
                "required": ["claim", "confidence", "evidence"]
            },
            handler=self._hypothesize,
        )

        self.register_tool(
            name="revise_hypothesis",
            description=(
                "Revise a previous hypothesis based on new evidence. "
                "Use this when you find information that changes your understanding."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "original_claim": {
                        "type": "string",
                        "description": "The original hypothesis to revise"
                    },
                    "new_claim": {
                        "type": "string",
                        "description": "The revised hypothesis"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you're revising"
                    }
                },
                "required": ["original_claim", "new_claim", "reason"]
            },
            handler=self._revise_hypothesis,
        )

        self.register_tool(
            name="set_temporal_ordering",
            description=(
                "Record your understanding of temporal ordering - which variables "
                "came before others. This is crucial for causal inference."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ordering": {
                        "type": "string",
                        "description": "Description of temporal ordering (e.g., 'Demographics at baseline, treatment assigned, then outcome measured')"
                    },
                    "pre_treatment_vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables measured before treatment"
                    },
                    "post_treatment_vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables measured after treatment"
                    }
                },
                "required": ["ordering"]
            },
            handler=self._set_temporal_ordering,
        )

        self.register_tool(
            name="mark_immutable",
            description=(
                "Mark a variable as immutable - it cannot be caused by other variables. "
                "Examples: age, sex, race, birth year, country of origin."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "The immutable variable"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this variable is immutable"
                    }
                },
                "required": ["variable", "reason"]
            },
            handler=self._mark_immutable,
        )

        self.register_tool(
            name="flag_uncertainty",
            description=(
                "Flag something you're uncertain about that downstream agents "
                "should be aware of. This helps them be cautious."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "What you're uncertain about"
                    },
                    "impact": {
                        "type": "string",
                        "description": "How this uncertainty might affect causal analysis"
                    }
                },
                "required": ["issue", "impact"]
            },
            handler=self._flag_uncertainty,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Get the initial observation from state."""
        metadata = state.raw_metadata or {}

        # Build lean initial observation
        obs = f"""You are investigating a new dataset for causal analysis.

Dataset: {metadata.get('title', state.dataset_info.name or 'Unknown')}
Source: Kaggle
Metadata quality: {metadata.get('metadata_quality', 'unknown')}

You have tools to investigate the metadata. Start by reading the description.
"""
        return obs

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if we have enough understanding."""
        # Need at least one treatment and one outcome hypothesis with medium+ confidence
        has_treatment = any(
            "treatment" in h["claim"].lower() and h["confidence"] in ["medium", "high"]
            for h in self._hypotheses
        )
        has_outcome = any(
            "outcome" in h["claim"].lower() and h["confidence"] in ["medium", "high"]
            for h in self._hypotheses
        )

        return has_treatment and has_outcome

    async def _read_description(self, state: AnalysisState) -> ToolResult:
        """Read the dataset description."""
        metadata = state.raw_metadata or {}

        description = metadata.get("description", "")
        subtitle = metadata.get("subtitle", "")
        title = metadata.get("title", "")

        if not description and not subtitle:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "title": title,
                    "description": "No description available",
                    "has_description": False
                }
            )

        # Combine available text
        full_text = f"{title}\n\n{subtitle}\n\n{description}" if subtitle else f"{title}\n\n{description}"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "title": title,
                "description": full_text.strip(),
                "has_description": True,
                "length": len(description)
            }
        )

    async def _list_columns(self, state: AnalysisState) -> ToolResult:
        """List all column names."""
        # Try to get columns from profile if available
        if state.data_profile:
            columns = state.data_profile.feature_names
        else:
            # Try to get from metadata
            metadata = state.raw_metadata or {}
            columns = list(metadata.get("column_descriptions", {}).keys())
            if not columns:
                # Try files info
                files = metadata.get("files", [])
                if files:
                    return ToolResult(
                        status=ToolResultStatus.SUCCESS,
                        output={
                            "columns": [],
                            "message": f"Column names not available yet. Dataset has {len(files)} files."
                        }
                    )
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "columns": [],
                        "message": "Column names not available in metadata. Will be discovered during profiling."
                    }
                )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "columns": columns,
                "count": len(columns)
            }
        )

    async def _investigate_column(self, state: AnalysisState, column: str) -> ToolResult:
        """Investigate what a column might mean."""
        metadata = state.raw_metadata or {}

        # Check for column description
        col_descriptions = metadata.get("column_descriptions", {})
        description = col_descriptions.get(column, None)

        # Analyze column name for clues
        name_lower = column.lower()
        clues = []

        # Treatment indicators
        if any(t in name_lower for t in ["treat", "intervention", "program", "policy", "exposed"]):
            clues.append("Name suggests this could be a TREATMENT variable")

        # Outcome indicators
        if any(o in name_lower for o in ["outcome", "result", "earnings", "income", "score", "effect", "response"]):
            clues.append("Name suggests this could be an OUTCOME variable")

        # Demographic/immutable indicators
        if any(d in name_lower for d in ["age", "sex", "gender", "race", "ethnicity", "birth", "dob"]):
            clues.append("Name suggests this is a DEMOGRAPHIC/IMMUTABLE variable")

        # Time indicators
        if any(t in name_lower for t in ["year", "month", "date", "time", "period", "wave"]):
            clues.append("Name suggests this is a TIME variable")

        # ID indicators
        if any(i in name_lower for i in ["id", "index", "key", "code"]):
            clues.append("Name suggests this is an IDENTIFIER (not for analysis)")

        # Pre/post indicators
        if any(p in name_lower for p in ["pre", "before", "baseline", "prior"]):
            clues.append("Name suggests this is measured BEFORE treatment (potential confounder)")
        if any(p in name_lower for p in ["post", "after", "follow"]):
            clues.append("Name suggests this is measured AFTER treatment")

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "column": column,
                "description": description,
                "has_description": description is not None,
                "name_clues": clues if clues else ["No obvious clues from name"],
            }
        )

    async def _search_metadata(self, state: AnalysisState, query: str) -> ToolResult:
        """Search metadata for a query."""
        metadata = state.raw_metadata or {}

        # Build searchable text
        searchable = ""
        searchable += metadata.get("title", "") + "\n"
        searchable += metadata.get("subtitle", "") + "\n"
        searchable += metadata.get("description", "") + "\n"
        searchable += " ".join(metadata.get("tags", [])) + "\n"
        searchable += " ".join(metadata.get("keywords", [])) + "\n"

        # Add column descriptions
        for col, desc in metadata.get("column_descriptions", {}).items():
            searchable += f"{col}: {desc}\n"

        searchable_lower = searchable.lower()
        query_lower = query.lower()

        # Find matches with context
        matches = []
        lines = searchable.split("\n")
        for line in lines:
            if query_lower in line.lower():
                matches.append(line.strip())

        if matches:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "query": query,
                    "found": True,
                    "matches": matches[:5],  # Limit to 5
                    "total_matches": len(matches)
                }
            )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "query": query,
                "found": False,
                "message": f"No matches found for '{query}' in metadata"
            }
        )

    async def _get_tags(self, state: AnalysisState) -> ToolResult:
        """Get dataset tags."""
        metadata = state.raw_metadata or {}

        tags = metadata.get("tags", [])
        keywords = metadata.get("keywords", [])

        # Infer domain from tags
        domain_hints = []
        all_tags = [t.lower() for t in tags + keywords]

        if any(t in all_tags for t in ["healthcare", "health", "medical", "medicine", "clinical"]):
            domain_hints.append("Healthcare/Medical domain")
        if any(t in all_tags for t in ["economics", "economic", "finance", "income", "employment"]):
            domain_hints.append("Economics/Finance domain")
        if any(t in all_tags for t in ["education", "school", "student", "learning"]):
            domain_hints.append("Education domain")
        if any(t in all_tags for t in ["social", "sociology", "demographics"]):
            domain_hints.append("Social science domain")
        if any(t in all_tags for t in ["causal", "treatment", "experiment", "rct"]):
            domain_hints.append("Explicitly tagged as causal/experimental")

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "tags": tags,
                "keywords": keywords,
                "domain_hints": domain_hints if domain_hints else ["Domain not clear from tags"]
            }
        )

    async def _hypothesize(
        self,
        state: AnalysisState,
        claim: str,
        confidence: str,
        evidence: str
    ) -> ToolResult:
        """Record a hypothesis."""
        hypothesis = {
            "claim": claim,
            "confidence": confidence,
            "evidence": evidence,
            "revised": False
        }
        self._hypotheses.append(hypothesis)

        self.logger.info(
            "hypothesis_recorded",
            claim=claim,
            confidence=confidence
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "recorded": True,
                "hypothesis": hypothesis,
                "total_hypotheses": len(self._hypotheses)
            }
        )

    async def _revise_hypothesis(
        self,
        state: AnalysisState,
        original_claim: str,
        new_claim: str,
        reason: str
    ) -> ToolResult:
        """Revise a previous hypothesis."""
        # Find and mark original as revised
        found = False
        for h in self._hypotheses:
            if h["claim"].lower() == original_claim.lower():
                h["revised"] = True
                h["superseded_by"] = new_claim
                found = True
                break

        # Add new hypothesis
        self._hypotheses.append({
            "claim": new_claim,
            "confidence": "medium",  # Default for revisions
            "evidence": f"Revised from: {original_claim}. Reason: {reason}",
            "revised": False
        })

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "original_found": found,
                "revision_recorded": True,
                "new_claim": new_claim
            }
        )

    async def _set_temporal_ordering(
        self,
        state: AnalysisState,
        ordering: str,
        pre_treatment_vars: list[str] | None = None,
        post_treatment_vars: list[str] | None = None
    ) -> ToolResult:
        """Record temporal ordering understanding."""
        self._temporal_understanding = ordering

        # Add pre-treatment vars to immutable candidates
        if pre_treatment_vars:
            for var in pre_treatment_vars:
                if var not in self._immutable_vars:
                    self._immutable_vars.append(var)

        self.logger.info("temporal_ordering_set", ordering=ordering)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "ordering": ordering,
                "pre_treatment_vars": pre_treatment_vars or [],
                "post_treatment_vars": post_treatment_vars or []
            }
        )

    async def _mark_immutable(
        self,
        state: AnalysisState,
        variable: str,
        reason: str
    ) -> ToolResult:
        """Mark a variable as immutable."""
        if variable not in self._immutable_vars:
            self._immutable_vars.append(variable)

        self.logger.info("variable_marked_immutable", variable=variable)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "variable": variable,
                "reason": reason,
                "total_immutable": len(self._immutable_vars)
            }
        )

    async def _flag_uncertainty(
        self,
        state: AnalysisState,
        issue: str,
        impact: str
    ) -> ToolResult:
        """Flag an uncertainty."""
        uncertainty = {
            "issue": issue,
            "impact": impact
        }
        self._uncertainties.append(uncertainty)

        self.logger.info("uncertainty_flagged", issue=issue)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "recorded": True,
                "uncertainty": uncertainty,
                "total_uncertainties": len(self._uncertainties)
            }
        )

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the domain knowledge investigation."""
        # Reset state for this execution
        self._hypotheses = []
        self._uncertainties = []
        self._temporal_understanding = None
        self._immutable_vars = []

        # Run the ReAct loop
        state = await super().execute(state)

        # Store findings in state
        state.domain_knowledge = {
            "hypotheses": [h for h in self._hypotheses if not h.get("revised", False)],
            "all_hypotheses": self._hypotheses,  # Include revised for traceability
            "uncertainties": self._uncertainties,
            "temporal_understanding": self._temporal_understanding,
            "immutable_vars": self._immutable_vars,
            "investigation_complete": True,
        }

        self.logger.info(
            "domain_knowledge_complete",
            num_hypotheses=len(state.domain_knowledge["hypotheses"]),
            num_uncertainties=len(self._uncertainties),
            num_immutable=len(self._immutable_vars)
        )

        return state
