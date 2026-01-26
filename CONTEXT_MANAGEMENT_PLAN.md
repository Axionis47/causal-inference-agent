# Context Management Analysis & Improvement Plan

## Executive Summary

The current agentic system has two critical context problems:
1. **Context bloat** - Agents receive far more information than they need
2. **Context poverty** - Agents lack semantic understanding of what variables mean

This document analyzes both problems and provides implementation plans to fix them.

---

## Part 1: Context Bloat Problem

### Current State Analysis

#### Problem 1: DataProfile Contains Everything

```python
# Current DataProfile structure
class DataProfile(BaseModel):
    n_samples: int
    n_features: int
    feature_names: list[str]                    # ALL columns
    feature_types: dict[str, str]               # Type for EVERY column
    missing_values: dict[str, int]              # Missing count for EVERY column
    numeric_stats: dict[str, dict[str, float]]  # Stats for EVERY numeric column
    categorical_stats: dict[str, dict[str, int]] # Stats for EVERY categorical column
```

**Impact**: For a 100-column dataset:
- `feature_names`: 100 strings
- `feature_types`: 100 key-value pairs
- `numeric_stats`: ~70 columns × 6 stats = 420 values
- `categorical_stats`: ~30 columns × N categories each

**Token estimate**: 2000-5000 tokens just for profile

#### Problem 2: Effect Estimator Initial Prompt

```python
# Current _build_initial_prompt() - Lines 544-565
prompt = f"""You need to estimate the causal effect of treatment on outcome.

Treatment variable: {self._treatment_var}
Outcome variable: {self._outcome_var}
Available covariates for adjustment: {self._covariates[:20]}{'...' if len(self._covariates) > 20 else ''}

"""
if state.confounder_discovery:
    prompt += f"""
Confounder Discovery Results:
- Confirmed confounders: {state.confounder_discovery.get('ranked_confounders', [])}
- Discovery reasoning: {state.confounder_discovery.get('reasoning', 'N/A')[:500]}
"""

if state.data_profile:
    prompt += f"""
Data Profile:
- Samples: {state.data_profile.n_samples}
- Has time dimension: {state.data_profile.has_time_dimension}
- Potential instruments: {state.data_profile.potential_instruments}
- Discontinuity candidates: {state.data_profile.discontinuity_candidates}
"""
```

**Problems**:
- Dumps 20 covariates upfront (agent can request via tool)
- Dumps 500 chars of confounder reasoning (agent can request via tool)
- Dumps profile info that may not be relevant

**Token estimate**: 400-800 tokens before agent even starts

#### Problem 3: Full State Passed to Every Agent

```python
# Every agent's execute() receives full AnalysisState
async def execute(self, state: AnalysisState) -> AnalysisState:
    # state contains:
    # - Full DataProfile
    # - Full EDA results
    # - Full causal DAG
    # - All treatment effects
    # - All sensitivity results
    # - Full confounder discovery
    # - Full PS diagnostics
    # - All data repairs
    # - Full debate history
    # - Full critique history
    # - All agent traces
```

**Impact**: Later agents receive accumulated context from all previous agents.

#### Problem 4: Traces Accumulate Indefinitely

```python
class AnalysisState(BaseModel):
    agent_traces: list[AgentTrace] = Field(default_factory=list)

    def add_trace(self, trace: AgentTrace) -> None:
        self.agent_traces.append(trace)  # Never truncated
```

### Quantified Impact

| Component | Tokens (est.) | Used By Agent? |
|-----------|---------------|----------------|
| Full DataProfile | 2000-5000 | Partially |
| Confounder discovery dump | 200-500 | Sometimes |
| Previous agent traces | 500-2000 | Rarely |
| EDA results | 500-1500 | Sometimes |
| Critique history | 200-500 | Only critique agent |

**Total wasted context per agent call**: 1500-4000 tokens

---

## Part 2: Context Poverty Problem

### Current State Analysis

The LLM doesn't know what variables mean:

```python
# What the agent sees:
treatment_candidates: ['treat', 'black', 'hisp']
outcome_candidates: ['re78', 're75', 're74']

# What the agent needs to know:
treat: "Whether participant received job training (1=yes, 0=control)"
re78: "Real earnings in 1978 (USD)"
black: "Race indicator - African American (demographic, not treatment)"
```

Without semantic context, the LLM:
- Can't distinguish real treatments from demographic indicators
- Can't identify temporal relationships (re74 is pre-treatment, re78 is post)
- Makes guesses based on variable names alone

### Why This Matters

```
LLM sees: treat, black, hisp as treatment candidates
LLM guesses: All three are valid treatments

Reality:
- treat: Actual experimental intervention ✅
- black: Demographic indicator (not manipulable) ❌
- hisp: Demographic indicator (not manipulable) ❌
```

---

## Part 3: Implementation Plan - Context Optimization

### Phase 1: Lean Initial Prompts (Effort: Low)

**File**: `effect_estimator.py`

**Current**:
```python
def _build_initial_prompt(self, state: AnalysisState) -> str:
    prompt = f"""You need to estimate the causal effect...
    Available covariates: {self._covariates[:20]}...
    Confounder Discovery: {state.confounder_discovery}...
    Data Profile: {state.data_profile}...
    """
```

**Proposed**:
```python
def _build_initial_prompt(self, state: AnalysisState) -> str:
    return f"""Estimate the causal effect of '{self._treatment_var}' on '{self._outcome_var}'.

Dataset: {state.data_profile.n_samples} samples, {len(self._covariates)} covariates available.

Use tools to:
1. get_data_summary - See treatment/outcome distributions
2. check_covariate_balance - Assess confounding
3. run_estimation_method - Execute causal methods

Start by getting a data summary."""
```

**Savings**: ~500 tokens per agent call

### Phase 2: On-Demand Context Tools (Effort: Medium)

**Add new tools to effect estimator**:

```python
TOOLS = [
    # ... existing tools ...
    {
        "name": "get_confounder_analysis",
        "description": "Get the results of confounder discovery including ranked confounders and reasoning.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_profile_for_variables",
        "description": "Get detailed statistics for specific variables.",
        "parameters": {
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variable names to get stats for"
                }
            },
            "required": ["variables"]
        },
    },
]

def _tool_get_confounder_analysis(self) -> str:
    """Return confounder discovery results on demand."""
    if not self._state.confounder_discovery:
        return "No confounder discovery results available."

    cd = self._state.confounder_discovery
    return f"""Confounder Discovery Results:
- Ranked confounders: {cd.get('ranked_confounders', [])}
- Reasoning: {cd.get('reasoning', 'N/A')}
- Method: {cd.get('method', 'N/A')}"""

def _tool_get_profile_for_variables(self, variables: list[str]) -> str:
    """Return profile stats for specific variables only."""
    profile = self._state.data_profile
    if not profile:
        return "No profile available."

    result = []
    for var in variables:
        if var in profile.numeric_stats:
            stats = profile.numeric_stats[var]
            result.append(f"{var}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        elif var in profile.categorical_stats:
            cats = profile.categorical_stats[var]
            result.append(f"{var}: {len(cats)} categories, top={list(cats.keys())[:3]}")

    return "\n".join(result) if result else "Variables not found in profile."
```

### Phase 3: Agent-Specific State Views (Effort: High)

**Create filtered state views**:

```python
# New file: backend/src/agents/base/state_views.py

from dataclasses import dataclass
from .state import AnalysisState, DataProfile

@dataclass
class EffectEstimatorView:
    """Minimal view for effect estimation."""
    dataframe_path: str
    treatment_candidates: list[str]
    outcome_candidates: list[str]
    potential_confounders: list[str]
    user_treatment: str | None
    user_outcome: str | None
    n_samples: int
    has_time_dimension: bool

    @classmethod
    def from_state(cls, state: AnalysisState) -> "EffectEstimatorView":
        profile = state.data_profile
        return cls(
            dataframe_path=state.dataframe_path,
            treatment_candidates=profile.treatment_candidates if profile else [],
            outcome_candidates=profile.outcome_candidates if profile else [],
            potential_confounders=profile.potential_confounders[:20] if profile else [],
            user_treatment=state.treatment_variable,
            user_outcome=state.outcome_variable,
            n_samples=profile.n_samples if profile else 0,
            has_time_dimension=profile.has_time_dimension if profile else False,
        )


@dataclass
class SensitivityAnalystView:
    """Minimal view for sensitivity analysis."""
    dataframe_path: str
    treatment_variable: str
    outcome_variable: str
    treatment_effects: list  # Only the effect results
    n_samples: int


@dataclass
class CritiqueAgentView:
    """View for critique agent - needs more context."""
    treatment_effects: list
    sensitivity_results: list
    confounder_discovery: dict | None
    iteration_count: int
    max_iterations: int
```

### Phase 4: Trace Truncation (Effort: Low)

```python
# In AnalysisState class
def add_trace(self, trace: AgentTrace) -> None:
    """Add trace with automatic truncation."""
    self.agent_traces.append(trace)

    # Keep only last 10 detailed traces
    if len(self.agent_traces) > 10:
        # Summarize older traces
        old_traces = self.agent_traces[:-10]
        summary_trace = AgentTrace(
            agent_name="system",
            action="trace_summary",
            reasoning=f"Summarized {len(old_traces)} earlier traces",
            outputs={"summarized_agents": [t.agent_name for t in old_traces]},
        )
        self.agent_traces = [summary_trace] + self.agent_traces[-10:]

def get_recent_traces(self, n: int = 5) -> list[AgentTrace]:
    """Get only recent traces for context."""
    return self.agent_traces[-n:]
```

---

## Part 4: Implementation Plan - Kaggle Metadata Integration

### Goal

Give the LLM semantic understanding of what each variable represents.

### Kaggle API Capabilities

```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Get dataset metadata
metadata = api.dataset_view("owner/dataset-name")
# Returns: title, subtitle, description, tags, columns (if provided)

# Get dataset files
files = api.dataset_list_files("owner/dataset-name")
# Returns: file names, sizes
```

### Implementation

#### Step 1: Add Metadata Fetcher (in data_profiler.py)

```python
async def _fetch_kaggle_metadata(self, url: str) -> dict | None:
    """Fetch dataset metadata from Kaggle for semantic context."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        # Extract dataset reference from URL
        parts = url.rstrip("/").split("/")
        if "datasets" in parts:
            idx = parts.index("datasets")
            dataset_ref = "/".join(parts[idx + 1:idx + 3])
        else:
            return None

        # Fetch metadata
        metadata = api.dataset_view(dataset_ref)

        return {
            "title": getattr(metadata, 'title', None),
            "subtitle": getattr(metadata, 'subtitle', None),
            "description": getattr(metadata, 'description', None),
            "tags": getattr(metadata, 'tags', []),
            "total_bytes": getattr(metadata, 'totalBytes', None),
            "usability_rating": getattr(metadata, 'usabilityRating', None),
        }
    except Exception as e:
        self.logger.warning("kaggle_metadata_fetch_failed", error=str(e))
        return None
```

#### Step 2: Add Metadata to State

```python
# In state.py - Update DatasetInfo
class DatasetInfo(BaseModel):
    url: str
    name: str | None = None
    local_path: str | None = None
    n_samples: int | None = None
    n_features: int | None = None

    # NEW: Kaggle metadata
    kaggle_title: str | None = None
    kaggle_description: str | None = None
    kaggle_tags: list[str] = Field(default_factory=list)
```

#### Step 3: Use Metadata in Profiler

```python
# In data_profiler.py - Update execute()
async def execute(self, state: AnalysisState) -> AnalysisState:
    # ... existing code ...

    # Fetch Kaggle metadata for semantic context
    if "kaggle.com" in state.dataset_info.url:
        metadata = await self._fetch_kaggle_metadata(state.dataset_info.url)
        if metadata:
            state.dataset_info.kaggle_title = metadata.get("title")
            state.dataset_info.kaggle_description = metadata.get("description")
            state.dataset_info.kaggle_tags = metadata.get("tags", [])

            self.logger.info(
                "kaggle_metadata_fetched",
                title=metadata.get("title"),
                tags=metadata.get("tags"),
            )
```

#### Step 4: Add Tool for Agents to Request Context

```python
# In data_profiler.py - Add tool
{
    "name": "get_dataset_context",
    "description": "Get the dataset description from Kaggle to understand what variables represent and the domain context.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

def _tool_get_dataset_context(self) -> str:
    """Return Kaggle metadata for semantic understanding."""
    info = self._state.dataset_info

    if not info.kaggle_description:
        return "No dataset description available from Kaggle."

    return f"""Dataset: {info.kaggle_title or info.name or 'Unknown'}

Description:
{info.kaggle_description[:2000]}

Tags: {', '.join(info.kaggle_tags) if info.kaggle_tags else 'None'}

Use this context to understand what each variable represents."""
```

#### Step 5: Integrate into Pair Selection Prompt

```python
# In effect_estimator.py - Update _build_pair_selection_prompt()
def _build_pair_selection_prompt(self, profile, dataset_info: DatasetInfo | None = None) -> str:
    context_section = ""
    if dataset_info and dataset_info.kaggle_description:
        context_section = f"""
Dataset Context (from Kaggle):
Title: {dataset_info.kaggle_title}
Description: {dataset_info.kaggle_description[:500]}
Tags: {', '.join(dataset_info.kaggle_tags[:5]) if dataset_info.kaggle_tags else 'None'}

Use this context to understand what variables represent.
"""

    return f"""You are evaluating potential causal relationships in a dataset.
{context_section}
Dataset Profile:
- Total features: {profile.n_features}
- Feature names: {profile.feature_names[:30]}
- Treatment candidates: {profile.treatment_candidates}
- Outcome candidates: {profile.outcome_candidates}

... rest of prompt ...
"""
```

---

## Part 5: Implementation Priority

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| Lean initial prompts | High | Low | 1 |
| Kaggle metadata fetcher | High | Low | 2 |
| On-demand context tools | Medium | Medium | 3 |
| Trace truncation | Low | Low | 4 |
| Agent-specific state views | Medium | High | 5 |

### Recommended Order

1. **Week 1**: Implement lean prompts + Kaggle metadata
2. **Week 2**: Add on-demand context tools
3. **Week 3**: Trace truncation + state views (if needed)

---

## Part 6: Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Tokens per effect estimator prompt | ~800 | ~200 |
| Context relevance (% used) | ~30% | ~80% |
| Pair selection accuracy | Unknown | Measurable with metadata |
| Agent iterations to completion | ~15-20 | ~10-12 |

---

## Files to Modify

1. `backend/src/agents/specialists/effect_estimator.py`
   - Lean prompts
   - Add context tools
   - Use metadata in pair selection

2. `backend/src/agents/specialists/data_profiler.py`
   - Add `_fetch_kaggle_metadata()`
   - Add `get_dataset_context` tool
   - Store metadata in state

3. `backend/src/agents/base/state.py`
   - Add metadata fields to DatasetInfo
   - Add trace truncation
   - (Later) Add state view methods

4. `backend/src/agents/base/state_views.py` (NEW)
   - Agent-specific filtered views
