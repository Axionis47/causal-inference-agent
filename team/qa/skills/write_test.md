# Skill: Write Pytest Tests

Load when: creating or modifying test files.

## Test location

```
backend/tests/unit/           <- Unit tests (mock LLM, fast)
backend/tests/integration/    <- Integration tests (may use real data, slow)
```

## Test patterns

### Agent test (mock LLM)
```python
from unittest.mock import AsyncMock, patch
from src.agents.base import AnalysisState, DatasetInfo
from src.llm.client import LLMResponse

@patch("src.llm.client.get_llm_client")
async def test_agent_produces_output(mock_llm):
    client = AsyncMock()
    client.generate.return_value = LLMResponse(text="mock response")
    client.generate_with_function_calling.return_value = {
        "response": "mock thought",
        "pending_calls": [{"name": "finish", "args": {"summary": "done", "success": True}}],
    }
    mock_llm.return_value = client

    state = AnalysisState(
        job_id="test-123",
        dataset_info=DatasetInfo(url="https://example.com/data"),
    )
    agent = MyAgent()
    result = await agent.execute(state)
    assert result.my_output_field is not None
```

### Causal method test
```python
import numpy as np
import pandas as pd
from src.causal.methods.ols import OLSMethod

def test_ols_returns_valid_result():
    df = pd.DataFrame({
        "treatment": np.random.binomial(1, 0.5, 1000),
        "outcome": np.random.normal(0, 1, 1000),
        "x1": np.random.normal(0, 1, 1000),
    })
    method = OLSMethod()
    method.fit(df, "treatment", "outcome", covariates=["x1"])
    result = method.estimate()
    assert result.method == "ols"
    assert not np.isnan(result.estimate)
    assert result.ci_lower < result.ci_upper
```

### State contract test
```python
def test_agent_declares_state_contracts():
    agent = MyAgent()
    assert hasattr(agent, "REQUIRED_STATE_FIELDS")
    assert hasattr(agent, "WRITES_STATE_FIELDS")
    assert len(agent.WRITES_STATE_FIELDS) > 0
```

## Rules

1. Always run: pytest tests/ -q --tb=short (NEVER use -v)
2. Mock LLM calls — never make real API calls in unit tests
3. Use AsyncMock for async agent methods
4. Each test function tests ONE thing
5. Test state contracts: verify agents declare what they read and write
6. Max 80 lines per test file commit
