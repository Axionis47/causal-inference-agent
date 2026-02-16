# Development Guide

This document covers how to extend the system: adding agents, causal methods, LLM providers, storage backends, and notebook sections. It also covers testing patterns and code conventions.

---

## Adding a New Agent

### 1. Create the agent file

Create a new file in `backend/src/agents/specialists/`. For a ReAct agent (recommended for most use cases):

```python
"""My new agent."""

from src.agents.base import AnalysisState
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent, ToolResult, ToolResultStatus
from src.agents.registry import register_agent


@register_agent("my_agent")
class MyAgent(ReActAgent, ContextTools):
    AGENT_NAME = "my_agent"
    MAX_STEPS = 15

    SYSTEM_PROMPT = """You are an expert at [task description].
    Analyze the data and [specific instructions]."""

    # State contracts
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    WRITES_STATE_FIELDS = ["my_output_field"]

    # Pipeline metadata
    JOB_STATUS = "estimating_effects"  # Must be a valid JobStatus value
    PROGRESS_WEIGHT = 50

    def __init__(self):
        super().__init__()
        self._register_context_tools()  # Register pull-based context tools
        self._register_domain_tools()

    def _register_domain_tools(self):
        self.register_tool(
            name="analyze_something",
            description="Analyze something specific in the data",
            parameters={
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Variable to analyze",
                    },
                },
                "required": ["variable"],
            },
            handler=self._handle_analyze,
        )

        self.register_tool(
            name="finalize",
            description="Finalize the analysis with results",
            parameters={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "object",
                        "description": "Final results",
                    },
                },
                "required": ["results"],
            },
            handler=self._handle_finalize,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        return (
            f"Job {state.job_id}: Analyze dataset with "
            f"{state.data_profile.n_samples} samples and "
            f"{state.data_profile.n_features} features."
        )

    async def is_task_complete(self, state: AnalysisState) -> bool:
        # Return True when the agent's output field is populated
        return getattr(state, "my_output_field", None) is not None

    async def _handle_analyze(self, state, variable: str) -> ToolResult:
        # Implement analysis logic
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"variable": variable, "finding": "..."},
        )

    async def _handle_finalize(self, state, results: dict) -> ToolResult:
        state.my_output_field = results
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"finalized": True},
        )
```

### 2. Add the state field

If your agent writes a new field, add it to `AnalysisState` in `backend/src/agents/base/state.py`:

```python
class AnalysisState(BaseModel):
    # ... existing fields ...

    # Populated by MyAgent
    my_output_field: dict[str, Any] | None = None
```

### 3. Import trigger

The agent module must be imported so the `@register_agent` decorator runs. Add an import in `backend/src/agents/specialists/__init__.py`:

```python
from .my_agent import MyAgent
```

### 4. Verify registration

```bash
cd backend
python3 -c "from src.agents.registry import create_all_agents; agents = create_all_agents(); print(f'{len(agents)} agents: {sorted(agents.keys())}')"
```

### 5. Add the orchestrator dispatch

Update the orchestrator's system prompt in `orchestrator_agent.py` to include your agent in the pipeline workflow description. The tool schemas are auto-generated from registered agents, so `dispatch_to_agent` will already accept your agent's name.

---

## Adding a New Causal Method

### 1. Create the method file

Create a new file in `backend/src/causal/methods/`:

```python
"""My estimation method."""

import numpy as np
import pandas as pd

from .base import BaseCausalMethod, MethodResult, register_method


@register_method("my_method")
class MyMethod(BaseCausalMethod):
    METHOD_NAME = "my_method"
    ESTIMAND = "ATE"  # ATE, ATT, CATE, or LATE

    def fit(self, df, treatment_col, outcome_col, covariates=None, **kwargs):
        self._df = df
        self._treatment_col = treatment_col
        self._outcome_col = outcome_col
        self._covariates = covariates or []

        treatment = self._binarize_treatment(df[treatment_col])
        outcome = df[outcome_col].values.astype(float)

        # Implement estimation logic here
        treated = outcome[treatment == 1]
        control = outcome[treatment == 0]
        estimate = treated.mean() - control.mean()
        se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
        ci_lower, ci_upper = self._compute_ci(estimate, se, n=len(df))
        p_value = self._compute_p_value(estimate, se, n=len(df))

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=estimate,
            std_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_treated=int(treatment.sum()),
            n_control=int((1 - treatment).sum()),
        )
        self._fitted = True
        return self

    def estimate(self):
        if not self._fitted:
            raise ValueError("Must call fit() before estimate()")
        return self._result

    def validate_assumptions(self, df, treatment_col, outcome_col):
        violations = super().validate_assumptions(df, treatment_col, outcome_col)
        # Add method-specific assumption checks
        return violations
```

### 2. Import trigger

Add an import in `backend/src/causal/methods/__init__.py`:

```python
from .my_method import MyMethod
```

### 3. Verify registration

```bash
cd backend
python3 -c "from src.causal.methods.base import get_method_registry; methods = get_method_registry(); print(f'{len(methods)} methods: {sorted(methods.keys())}')"
```

---

## Adding a Notebook Section

### 1. Create the renderer

Create a new file in `backend/src/agents/specialists/notebook/sections/`:

```python
"""My section renderer."""

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState


def render_my_section(state: AnalysisState) -> list:
    """Render my section cells."""
    cells = []

    md = "## My Section\n\n"
    md += "*Findings from MyAgent.*\n\n"
    # Add content from state.my_output_field
    cells.append(new_markdown_cell(md))

    return cells
```

### 2. Export the renderer

Add to `backend/src/agents/specialists/notebook/sections/__init__.py`:

```python
from .my_section import render_my_section
```

### 3. Call from the agent

Add the renderer call in `backend/src/agents/specialists/notebook/agent.py`:

```python
from .sections import render_my_section

# Inside execute(), add at the appropriate pipeline position:
if state.my_output_field:
    cells.extend(render_my_section(state))
```

---

## Adding an LLM Provider

Implement the `LLMClient` protocol (structural typing, no inheritance needed):

```python
"""My LLM provider client."""

from src.llm.client import LLMResponse


class MyLLMClient:
    async def generate(self, prompt, system_instruction=None, tools=None):
        # Call your LLM API
        return LLMResponse(text="...", raw=raw_response, usage={"input": 100, "output": 50})

    async def generate_with_function_calling(self, prompt, system_instruction, tools, max_iterations=5):
        # Implement function calling loop
        return {"response": "...", "tool_calls": [...], "pending_calls": [...]}

    async def generate_structured(self, prompt, response_schema, system_instruction=None):
        # Return a Pydantic model instance matching response_schema
        return response_schema(...)
```

Then add a case in `backend/src/llm/client.py`:

```python
def get_llm_client():
    # ... existing cases ...
    elif settings.llm_provider == "my_provider":
        from .my_client import MyLLMClient
        _llm_client = MyLLMClient()
```

---

## Adding a Storage Backend

Implement the 10-method `StorageProtocol`:

```python
class MyStorageClient:
    async def create_job(self, state): ...
    async def update_job(self, state): ...
    async def get_job(self, job_id): ...
    async def list_jobs(self, status, limit, offset): ...
    async def update_job_status(self, job_id, status, error_message): ...
    async def delete_job(self, job_id, cascade): ...
    async def save_results(self, state): ...
    async def get_results(self, job_id): ...
    async def save_traces(self, state): ...
    async def get_traces(self, job_id): ...
```

Add a case in `backend/src/storage/__init__.py`:

```python
def get_storage_client():
    # ... existing cases ...
    elif settings.storage_provider == "my_storage":
        from .my_storage import MyStorageClient
        return MyStorageClient()
```

---

## Testing

### Running tests

```bash
cd backend

# All unit tests (345 tests, ~3.5 minutes)
pytest tests/ --ignore=tests/integration -x

# Specific test file
pytest tests/test_agents.py -x

# With verbose output
pytest tests/ --ignore=tests/integration -x -v
```

### Verifying agent count

```bash
python3 -c "from src.agents.registry import create_all_agents; print(len(create_all_agents()))"
# Expected: 13
```

### Verifying method count

```bash
python3 -c "from src.causal.methods.base import get_method_registry; print(len(get_method_registry()))"
# Expected: 12
```

### Mocking LLM calls

Tests should mock the LLM client to avoid real API calls:

```python
from unittest.mock import AsyncMock, patch

from src.llm.client import LLMResponse


@patch("src.llm.client.get_llm_client")
async def test_my_agent(mock_llm):
    client = AsyncMock()
    client.generate.return_value = LLMResponse(text="mock response")
    client.generate_with_function_calling.return_value = {
        "response": "mock thought",
        "pending_calls": [{"name": "my_tool", "args": {"key": "value"}}],
    }
    mock_llm.return_value = client

    # Test agent logic...
```

### Testing agents in isolation

```python
from src.agents.base import AnalysisState, DatasetInfo

# Create minimal state
state = AnalysisState(
    job_id="test-123",
    dataset_info=DatasetInfo(url="https://example.com/data"),
)

# Set required fields
state.data_profile = DataProfile(
    n_samples=1000,
    n_features=10,
    feature_names=["x1", "x2", "y"],
    feature_types={"x1": "numeric", "x2": "binary", "y": "numeric"},
    missing_values={},
    numeric_stats={},
    categorical_stats={},
)

# Run agent
agent = MyAgent()
result = await agent.execute(state)
assert result.my_output_field is not None
```

### E2E testing

Full pipeline tests use synthetic data (no Kaggle dependency):

```bash
python3 tests/integration/run_e2e_tests.py
```

These take 8-10 minutes and run the complete pipeline from dataset loading to notebook generation.

---

## Code Conventions

### Import ordering

1. Standard library
2. Third-party packages
3. Local imports (`from src.agents...`)

### Structured logging

Use `get_logger(__name__)` from `src.logging_config.structured`. Log structured events, not formatted strings:

```python
# Good
self.logger.info("agent_completed", job_id=state.job_id, duration_ms=elapsed)

# Bad
self.logger.info(f"Agent completed job {state.job_id} in {elapsed}ms")
```

### Error handling

- Agents should catch expected errors (numeric failures, convergence issues) and return graceful fallbacks
- Do not use bare `except Exception: pass` without logging
- Use `state.mark_failed(error, agent_name)` for unrecoverable errors
- `StateValidationError` should be raised (not caught) when required state fields are missing

### LLM response handling

All LLM clients return `LLMResponse(text, raw, usage)`. Always access `response.text`:

```python
response = await self.llm.generate(prompt, system_instruction=self.SYSTEM_PROMPT)
text = response.text  # Always a string
```

Do not check `isinstance(response, dict)` or access `response["text"]`.
