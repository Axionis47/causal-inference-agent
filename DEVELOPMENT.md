# Development Guide

This document explains how to extend the Causal Orchestrator system.

---

## Overview

The system is designed to be extensible. You can add:

1. **New Specialist Agents**: For new analysis capabilities
2. **New Causal Methods**: For additional estimation techniques
3. **New LLM Providers**: For different AI backends
4. **New Storage Backends**: For different databases

Each extension point follows clear patterns.

---

## Adding a New Specialist Agent

**Location:** `backend/src/agents/specialists/`

### Step 1: Create the Agent File

```python
# backend/src/agents/specialists/your_agent.py

from typing import Any
from src.agents.base.react_agent import ReActAgent
from src.agents.base.context_tools import ContextTools
from src.agents.base.state import AnalysisState

class YourAgent(ReActAgent, ContextTools):
    """Your agent description.

    This agent does X by investigating Y and producing Z.
    """

    AGENT_NAME = "your_agent"
    MAX_STEPS = 15  # Adjust based on task complexity

    SYSTEM_PROMPT = """You are an expert at X.

Your job is to:
1. Investigate the data
2. Produce findings
3. Call finalize_analysis when done

Available tools:
- your_tool_1: Does X
- your_tool_2: Does Y
- finalize_analysis: Locks in your findings
"""

    def __init__(self):
        super().__init__()

        # Register tools from ContextTools mixin (domain knowledge, etc.)
        self.register_context_tools()

        # Register your custom tools
        self._register_tools()

        # Internal state for tracking
        self._findings = {}
        self._finalized = False

    def _register_tools(self):
        """Register agent-specific tools."""

        self.register_tool(
            name="your_tool_1",
            description="What this tool does",
            parameters={
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to analyze"
                    }
                },
                "required": ["column"]
            },
            handler=self._tool_your_tool_1,
        )

        self.register_tool(
            name="finalize_analysis",
            description="Finalize your findings",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of findings"
                    }
                },
                "required": ["summary"]
            },
            handler=self._tool_finalize,
        )

    async def _tool_your_tool_1(self, column: str) -> dict[str, Any]:
        """Tool implementation. Returns dict that becomes tool result."""

        # Access data from state
        df = self._get_dataframe()  # Helper from base class

        if column not in df.columns:
            return {"error": f"Column {column} not found"}

        # Do your analysis
        result = {
            "column": column,
            "mean": float(df[column].mean()),
            "std": float(df[column].std()),
        }

        # Store internally
        self._findings[column] = result

        return result

    async def _tool_finalize(self, summary: str) -> dict[str, Any]:
        """Finalize agent findings."""
        self._finalized = True

        return {
            "status": "finalized",
            "summary": summary,
            "findings": self._findings
        }

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """What the agent sees at the start."""

        profile = state.data_profile
        return f"""Dataset has {profile.n_samples} rows and {profile.n_features} columns.

Treatment: {state.treatment_variable}
Outcome: {state.outcome_variable}

Your task: Analyze the data and produce findings."""

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """When should the agent stop?"""
        return self._finalized

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Main execution. Uses the ReAct loop from base class."""

        # Load data
        self._load_dataframe(state)

        # Run ReAct loop
        state = await self.run_react_loop(state)

        # Write results to state
        state.your_result = self._findings

        return state
```

### Step 2: Register with Orchestrator

In `backend/src/agents/orchestrator/orchestrator_agent.py`, add your agent:

```python
# In the SPECIALIST_AGENTS dict
SPECIALIST_AGENTS = {
    "data_profiler": DataProfilerAgent,
    "eda_agent": EDAAgent,
    "your_agent": YourAgent,  # Add this
    ...
}
```

### Step 3: Add Status (Optional)

If your agent needs a specific job status, add to `backend/src/agents/base/state.py`:

```python
class JobStatus(str, Enum):
    ...
    YOUR_ANALYSIS = "your_analysis"  # Add this
```

### Key Patterns

**Tool Registration:**
- Tools are JSON Schema described functions
- The LLM sees descriptions and decides which to call
- Handlers return dicts that become tool results

**Context Tools Mixin:**
- Provides ready-made tools for querying previous results
- `ask_domain_knowledge()`: Query domain knowledge findings
- `get_column_info()`: Get stats for specific columns
- `get_eda_finding()`: Query EDA results
- `get_dag_adjustment_set()`: Get backdoor criterion sets

**ReAct Loop:**
- Agent observes state, reasons about what to do, calls tools
- Continues until `is_task_complete()` returns True
- Has MAX_STEPS limit to prevent infinite loops

---

## Adding a New Causal Method

**Location:** `backend/src/causal/methods/`

### Step 1: Create the Method File

```python
# backend/src/causal/methods/your_method.py

import numpy as np
import pandas as pd
from typing import Any
from .base import BaseCausalMethod, MethodResult

class YourCausalMethod(BaseCausalMethod):
    """Your method description.

    Estimates ATE using your approach.

    Assumptions:
    - Assumption 1
    - Assumption 2
    """

    METHOD_NAME = "Your Method"
    ESTIMAND = "ATE"  # ATE, ATT, CATE, or LATE

    def __init__(self, confidence_level: float = 0.95, **kwargs):
        """Initialize method.

        Args:
            confidence_level: For confidence intervals (default 0.95)
            **kwargs: Method-specific parameters
        """
        super().__init__(confidence_level)
        self._model = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "YourCausalMethod":
        """Fit the causal model.

        Args:
            df: Data
            treatment_col: Treatment variable name
            outcome_col: Outcome variable name
            covariates: List of covariate names

        Returns:
            Self for chaining
        """
        # Clean data
        cols = [treatment_col, outcome_col] + (covariates or [])
        df_clean = df[cols].dropna()

        # Extract arrays
        T = df_clean[treatment_col].values
        Y = df_clean[outcome_col].values
        X = df_clean[covariates].values if covariates else None

        # Your fitting logic here
        # self._model = ...

        # Store for estimate()
        self._T = T
        self._Y = Y
        self._X = X

        # Required: mark fitted and store counts
        self._fitted = True
        self._n_treated = int(T.sum())
        self._n_control = int(len(T) - T.sum())

        return self

    def estimate(self) -> MethodResult:
        """Compute treatment effect estimate."""

        if not self._fitted:
            raise ValueError("Call fit() before estimate()")

        # Compute your estimate
        ate = self._compute_ate()
        se = self._compute_se()

        # Use inherited helpers
        ci_lower, ci_upper = self._compute_ci(ate, se)
        p_value = self._compute_p_value(ate, se)

        return MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=ate,
            std_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=[
                "Assumption 1: checked",
                "Assumption 2: checked",
            ],
            diagnostics={
                "your_stat": some_value,
            }
        )

    def _compute_ate(self) -> float:
        """Your ATE computation."""
        # Implement your estimator
        pass

    def _compute_se(self) -> float:
        """Standard error via bootstrap or analytical."""
        # Implement SE computation
        pass

    def validate_assumptions(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str
    ) -> list[str]:
        """Check method assumptions.

        Returns:
            List of violations (empty if all pass)
        """
        violations = super().validate_assumptions(df, treatment_col, outcome_col)

        # Add method-specific checks
        T = df[treatment_col]

        # Example: check treatment is binary
        if T.nunique() > 2:
            violations.append("Treatment must be binary")

        return violations
```

### Step 2: Register the Method

In the effect estimator agent, add your method to the available methods:

```python
# In effect_estimator.py

AVAILABLE_METHODS = {
    "ols": OLSMethod,
    "ipw": IPWMethod,
    "your_method": YourCausalMethod,  # Add this
    ...
}
```

### Key Patterns

**Method Interface:**
- `fit()`: Prepare the model with data
- `estimate()`: Return MethodResult with estimate, SE, CI, p-value
- `validate_assumptions()`: Check if method is appropriate

**MethodResult Structure:**
```python
MethodResult(
    method="Name",           # Display name
    estimand="ATE",          # What is being estimated
    estimate=1.5,            # Point estimate
    std_error=0.3,           # Standard error
    ci_lower=0.9,            # Lower CI bound
    ci_upper=2.1,            # Upper CI bound
    p_value=0.001,           # Two-sided p-value
    n_treated=100,           # Treated count
    n_control=200,           # Control count
    assumptions_tested=[],   # List of checks performed
    diagnostics={},          # Method-specific diagnostics
)
```

---

## Adding a New LLM Provider

**Location:** `backend/src/llm/`

### Step 1: Create the Client

```python
# backend/src/llm/your_provider_client.py

from typing import Any
import httpx

class YourProviderClient:
    """Client for YourProvider API."""

    def __init__(self):
        from src.config import get_settings
        settings = get_settings()

        self.api_key = settings.your_provider_api_key
        self.model = settings.your_provider_model
        self.temperature = settings.your_provider_temperature
        self.max_tokens = settings.your_provider_max_tokens

        self.client = httpx.AsyncClient(
            base_url="https://api.yourprovider.com/v1",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Generate content from model.

        Args:
            prompt: User prompt
            system_instruction: System message
            tools: Function definitions for tool use

        Returns:
            Raw API response
        """
        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)

        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()

        return response.json()

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert internal tool format to provider format."""
        # The internal format is:
        # {
        #     "name": "tool_name",
        #     "description": "What it does",
        #     "parameters": { JSON Schema }
        # }
        #
        # Convert to your provider's format
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
            }
            for t in tools
        ]

    async def generate_with_function_calling(
        self,
        prompt: str,
        system_instruction: str,
        tools: list[dict[str, Any]],
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """Generate with automatic function calling loop.

        Returns:
            {
                "response": str,
                "tool_calls": list[dict],
                "pending_calls": list[dict],
                "iterations": int,
            }
        """
        all_tool_calls = []
        pending_calls = []
        iterations = 0
        final_response = ""

        while iterations < max_iterations:
            iterations += 1

            response = await self.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                tools=tools
            )

            # Parse response for your provider
            message = response["choices"][0]["message"]

            if message.get("tool_calls"):
                # Provider wants to call tools
                for tc in message["tool_calls"]:
                    pending_calls.append({
                        "name": tc["function"]["name"],
                        "args": tc["function"]["arguments"]
                    })
                break  # Return pending calls to agent
            else:
                # No more tool calls, we have final response
                final_response = message.get("content", "")
                break

        return {
            "response": final_response,
            "tool_calls": all_tool_calls,
            "pending_calls": pending_calls,
            "iterations": iterations,
        }

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type,
        system_instruction: str | None = None,
    ) -> Any:
        """Generate structured output matching schema."""
        import json

        schema_prompt = f"""Respond with valid JSON matching this schema:
{response_schema.model_json_schema()}

{prompt}"""

        response = await self.generate(
            prompt=schema_prompt,
            system_instruction=system_instruction
        )

        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)

        return response_schema(**data)


# Singleton pattern
_client: YourProviderClient | None = None

def get_your_provider_client() -> YourProviderClient:
    global _client
    if _client is None:
        _client = YourProviderClient()
    return _client
```

### Step 2: Add Settings

In `backend/src/config/settings.py`:

```python
class Settings(BaseSettings):
    ...
    # Your provider settings
    your_provider_api_key: SecretStr | None = Field(
        default=None,
        description="API key for YourProvider"
    )
    your_provider_model: str = Field(
        default="your-model-name",
        description="Model to use"
    )
    your_provider_temperature: float = Field(
        default=0.7,
        description="Sampling temperature"
    )
    your_provider_max_tokens: int = Field(
        default=2048,
        description="Max tokens in response"
    )
```

### Step 3: Register in Client Factory

In `backend/src/llm/client.py`:

```python
def get_llm_client() -> LLMClient:
    global _llm_client

    if _llm_client is None:
        settings = get_settings()

        if settings.llm_provider == "your_provider":
            from .your_provider_client import get_your_provider_client
            _llm_client = get_your_provider_client()
        elif settings.llm_provider == "claude":
            ...

    return _llm_client
```

---

## Adding a New Storage Backend

**Location:** `backend/src/storage/`

### Step 1: Implement the Interface

```python
# backend/src/storage/your_storage.py

from typing import Any
from src.agents.base.state import AnalysisState, JobStatus

class YourStorageClient:
    """Storage client for YourDatabase."""

    def __init__(self):
        from src.config import get_settings
        settings = get_settings()

        # Initialize your database connection
        self.connection = ...

    async def create_job(self, state: AnalysisState) -> str:
        """Create a new job record."""

        data = {
            "id": state.job_id,
            "status": state.status.value,
            "dataset_url": state.dataset_info.url,
            "dataset_name": state.dataset_info.name,
            "treatment": state.treatment_variable,
            "outcome": state.outcome_variable,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }

        # Insert into your database
        await self.connection.insert("jobs", data)

        return state.job_id

    async def update_job(self, state: AnalysisState) -> None:
        """Update existing job."""

        data = {
            "status": state.status.value,
            "treatment": state.treatment_variable,
            "outcome": state.outcome_variable,
            "updated_at": state.updated_at.isoformat(),
            "error_message": state.error_message,
        }

        await self.connection.update("jobs", state.job_id, data)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get job by ID."""

        result = await self.connection.get("jobs", job_id)
        return result

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List jobs with optional status filter.

        Returns:
            (list of jobs, total count)
        """

        query = {}
        if status:
            query["status"] = status.value

        results = await self.connection.find(
            "jobs",
            query,
            limit=limit,
            skip=offset
        )

        total = await self.connection.count("jobs", query)

        return results, total

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
    ) -> bool:
        """Update just the status field."""

        data = {"status": status.value}
        if error_message:
            data["error_message"] = error_message

        await self.connection.update("jobs", job_id, data)
        return True

    async def delete_job(self, job_id: str, cascade: bool = True) -> dict:
        """Delete job and optionally related data."""

        await self.connection.delete("jobs", job_id)

        if cascade:
            await self.connection.delete("results", {"job_id": job_id})
            await self.connection.delete("traces", {"job_id": job_id})

        return {"deleted": True, "job_id": job_id}

    async def save_results(self, state: AnalysisState) -> None:
        """Save analysis results."""

        data = {
            "job_id": state.job_id,
            "treatment_effects": [
                te.model_dump() for te in state.treatment_effects
            ],
            "sensitivity_results": [
                sr.model_dump() for sr in state.sensitivity_results
            ],
            "recommendations": state.recommendations,
        }

        await self.connection.upsert("results", state.job_id, data)

    async def get_results(self, job_id: str) -> dict[str, Any] | None:
        """Get results for job."""

        return await self.connection.get("results", job_id)

    async def save_traces(self, state: AnalysisState) -> None:
        """Save agent traces for observability."""

        for trace in state.agent_traces:
            data = {
                "job_id": state.job_id,
                "agent_name": trace.agent_name,
                "started_at": trace.started_at.isoformat(),
                "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
                "steps": trace.steps,
                "status": trace.status,
            }
            await self.connection.insert("traces", data)

    async def get_traces(self, job_id: str) -> list[dict]:
        """Get all traces for job."""

        return await self.connection.find("traces", {"job_id": job_id})


# Singleton
_client: YourStorageClient | None = None

def get_your_storage_client() -> YourStorageClient:
    global _client
    if _client is None:
        _client = YourStorageClient()
    return _client
```

### Step 2: Register in Factory

In `backend/src/storage/firestore.py`:

```python
def get_storage_client():
    settings = get_settings()

    if settings.storage_type == "your_storage":
        from .your_storage import get_your_storage_client
        return get_your_storage_client()
    elif settings.storage_type == "firestore":
        return get_firestore_client()
    else:
        return get_local_storage_client()
```

---

## Testing

**Location:** `backend/tests/`

### Test Fixtures

```python
# tests/conftest.py

import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_dataframe():
    """Create test data with known causal structure.

    True ATE is 2000 (treatment adds 2000 to outcome).
    """
    np.random.seed(42)
    n = 200

    # Confounders
    age = np.random.normal(40, 10, n)
    income = np.random.normal(50000, 15000, n)

    # Treatment depends on confounders
    propensity = 1 / (1 + np.exp(-(0.02 * (age - 40) + 0.00001 * (income - 50000))))
    treatment = np.random.binomial(1, propensity)

    # Outcome depends on treatment + confounders
    # TRUE ATE = 2000
    outcome = (
        5000
        + 2000 * treatment  # Treatment effect
        + 50 * age
        + 0.1 * income
        + np.random.normal(0, 1000, n)
    )

    return pd.DataFrame({
        "age": age,
        "income": income,
        "treatment": treatment,
        "outcome": outcome,
    })

@pytest.fixture
def analysis_state():
    """Create test AnalysisState."""
    from src.agents.base.state import AnalysisState, DatasetInfo
    from datetime import datetime

    return AnalysisState(
        job_id="test-job-123",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/test",
            name="test_dataset",
        ),
        treatment_variable="treatment",
        outcome_variable="outcome",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
```

### Testing an Agent

```python
# tests/unit/test_your_agent.py

import pytest
from src.agents.specialists.your_agent import YourAgent
from src.agents.base.state import DataProfile

class TestYourAgent:
    """Tests for YourAgent."""

    def test_agent_has_correct_tools(self):
        """Verify tools are registered."""
        agent = YourAgent()

        tool_names = [t["name"] for t in agent._tool_schemas]

        assert "your_tool_1" in tool_names
        assert "finalize_analysis" in tool_names

    @pytest.mark.asyncio
    async def test_tool_execution(self, sample_dataframe, analysis_state):
        """Test individual tool works."""
        agent = YourAgent()
        agent._df = sample_dataframe  # Inject data

        result = await agent._tool_your_tool_1(column="age")

        assert "mean" in result
        assert "std" in result
        assert result["column"] == "age"

    @pytest.mark.asyncio
    async def test_tool_handles_missing_column(self, sample_dataframe, analysis_state):
        """Test error handling."""
        agent = YourAgent()
        agent._df = sample_dataframe

        result = await agent._tool_your_tool_1(column="nonexistent")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_finalize_marks_complete(self, analysis_state):
        """Test finalization."""
        agent = YourAgent()

        await agent._tool_finalize(summary="Done")

        assert agent._finalized is True
        assert await agent.is_task_complete(analysis_state) is True
```

### Testing a Causal Method

```python
# tests/unit/test_your_method.py

import pytest
from src.causal.methods.your_method import YourCausalMethod

class TestYourCausalMethod:
    """Tests for YourCausalMethod."""

    def test_fit_stores_data(self, sample_dataframe):
        """Test fitting stores necessary data."""
        method = YourCausalMethod()

        method.fit(
            df=sample_dataframe,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["age", "income"]
        )

        assert method._fitted is True
        assert method._n_treated > 0
        assert method._n_control > 0

    def test_estimate_returns_result(self, sample_dataframe):
        """Test estimation produces valid result."""
        method = YourCausalMethod()

        method.fit(
            df=sample_dataframe,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["age", "income"]
        )

        result = method.estimate()

        assert result.method == "Your Method"
        assert result.estimand == "ATE"
        assert result.estimate is not None
        assert result.std_error > 0
        assert result.ci_lower < result.ci_upper
        assert 0 <= result.p_value <= 1

    def test_estimate_near_true_ate(self, sample_dataframe):
        """Test estimate is close to true ATE (2000)."""
        method = YourCausalMethod()

        method.fit(
            df=sample_dataframe,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["age", "income"]
        )

        result = method.estimate()

        # Should be within reasonable range of true ATE=2000
        assert 1000 < result.estimate < 3000

    def test_unfitted_raises_error(self):
        """Test estimate without fit raises error."""
        method = YourCausalMethod()

        with pytest.raises(ValueError, match="fit"):
            method.estimate()

    def test_assumption_validation(self, sample_dataframe):
        """Test assumption checking."""
        method = YourCausalMethod()

        violations = method.validate_assumptions(
            df=sample_dataframe,
            treatment_col="treatment",
            outcome_col="outcome"
        )

        # Sample data should pass basic checks
        assert len(violations) == 0
```

### Mocking the LLM

```python
# tests/unit/test_agent_with_mock_llm.py

import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_llm_client():
    """Mock LLM that immediately finalizes."""
    mock = AsyncMock()
    mock.generate_with_function_calling = AsyncMock(
        return_value={
            "response": "I have completed the analysis.",
            "tool_calls": [],
            "pending_calls": [
                {
                    "name": "finalize_analysis",
                    "args": {"summary": "Analysis complete"}
                }
            ],
            "iterations": 1,
        }
    )
    return mock

@pytest.mark.asyncio
async def test_agent_with_mocked_llm(analysis_state, mock_llm_client):
    """Test agent with mocked LLM."""

    with patch("src.llm.get_llm_client", return_value=mock_llm_client):
        from src.agents.specialists.your_agent import YourAgent

        agent = YourAgent()
        result = await agent.execute(analysis_state)

        # LLM was called
        mock_llm_client.generate_with_function_calling.assert_called()

        # Agent completed
        assert agent._finalized is True
```

### Running Tests

```bash
cd backend

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/unit/test_your_agent.py -v
```

---

## Code Organization

```
backend/
├── src/
│   ├── agents/
│   │   ├── base/
│   │   │   ├── agent.py           # Base agent class
│   │   │   ├── react_agent.py     # ReAct loop implementation
│   │   │   ├── context_tools.py   # Pull-based context mixin
│   │   │   └── state.py           # AnalysisState and related models
│   │   ├── orchestrator/
│   │   │   └── orchestrator_agent.py
│   │   ├── specialists/
│   │   │   ├── data_profiler.py
│   │   │   ├── eda_agent.py
│   │   │   ├── effect_estimator.py
│   │   │   └── your_agent.py      # Add your agent here
│   │   └── critique/
│   │       └── critique_agent.py
│   ├── causal/
│   │   ├── dag/
│   │   │   └── discovery.py       # DAG discovery algorithms
│   │   └── methods/
│   │       ├── base.py            # BaseCausalMethod
│   │       ├── ols.py
│   │       ├── propensity.py
│   │       └── your_method.py     # Add your method here
│   ├── llm/
│   │   ├── client.py              # LLM client factory
│   │   ├── claude_client.py
│   │   ├── gemini_client.py
│   │   └── your_provider.py       # Add your provider here
│   ├── storage/
│   │   ├── firestore.py           # Firestore + factory
│   │   ├── local_storage.py       # JSON file storage
│   │   └── your_storage.py        # Add your storage here
│   ├── api/
│   │   └── routes/
│   │       └── jobs.py            # API endpoints
│   ├── jobs/
│   │   └── manager.py             # Job lifecycle management
│   └── config/
│       └── settings.py            # Configuration
└── tests/
    ├── conftest.py                # Shared fixtures
    ├── unit/
    │   └── test_your_agent.py
    └── integration/
        └── test_pipeline.py
```

---

## Key Architectural Principles

### 1. Pull-Based Context

Agents query for what they need instead of receiving everything upfront:

```python
# Good: Pull what you need
result = await self.ask_domain_knowledge("Is there a time dimension?")

# Bad: Dump everything to LLM
prompt = f"Here is all the data: {all_data}\nNow analyze..."
```

### 2. State as Contract

AnalysisState is the contract between agents:

```python
# Each agent reads what it needs
profile = state.data_profile
dag = state.proposed_dag

# Each agent writes its results
state.your_result = findings

# State flows through the pipeline
orchestrator -> profiler -> eda -> discovery -> estimator -> ...
```

### 3. Tools as Interface

The LLM decides which tools to call. No hardcoded if-else:

```python
# Good: Register tools, let LLM decide
self.register_tool(name="check_balance", ...)
self.register_tool(name="detect_outliers", ...)

# Bad: Hardcode sequence
if step == 1:
    check_balance()
elif step == 2:
    detect_outliers()
```

### 4. Error Recovery

Use `execute_with_tracing()` for automatic error handling:

```python
async def execute(self, state: AnalysisState) -> AnalysisState:
    # This handles errors, retries, and tracing
    return await self.execute_with_tracing(state)
```

### 5. Storage Abstraction

Same code works with different backends:

```python
# Code doesn't care which storage is used
storage = get_storage_client()  # Returns Firestore or LocalStorage
await storage.create_job(state)
```
