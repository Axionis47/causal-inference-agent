# Skill: Write ReAct Agent

Load when: creating or modifying agent files in backend/src/agents/specialists/.

## Agent contract

Every specialist agent follows this pattern:

```python
from src.agents.base import AnalysisState
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent, ToolResult, ToolResultStatus
from src.agents.registry import register_agent


@register_agent("my_agent")
class MyAgent(ReActAgent, ContextTools):
    AGENT_NAME = "my_agent"
    MAX_STEPS = 15
    SYSTEM_PROMPT = """You are an expert at [task]."""
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    WRITES_STATE_FIELDS = ["my_output_field"]
    JOB_STATUS = "estimating_effects"
    PROGRESS_WEIGHT = 50

    def __init__(self):
        super().__init__()
        self._register_context_tools()
        self._register_domain_tools()

    def _register_domain_tools(self):
        self.register_tool(
            name="tool_name",
            description="What this tool does",
            parameters={"type": "object", "properties": {...}, "required": [...]},
            handler=self._handle_tool,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        return f"Job {state.job_id}: [lean context, 150-200 tokens max]"

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return getattr(state, "my_output_field", None) is not None

    async def _handle_tool(self, state, **kwargs) -> ToolResult:
        return ToolResult(status=ToolResultStatus.SUCCESS, output={...})
```

## Rules

1. Always inherit ReActAgent + ContextTools (unless BaseAgent is simpler)
2. Always use @register_agent decorator
3. Always declare REQUIRED_STATE_FIELDS and WRITES_STATE_FIELDS
4. Call self._register_context_tools() in __init__ for pull-based context
5. Keep _get_initial_observation lean — 150-200 tokens, not data dumps
6. Every agent needs a finalize tool that writes results to state
7. LLM calls through self.llm (property that calls get_llm_client())
8. Max 200 lines per file. Split into packages if longer.
9. Add state field to AnalysisState in state.py
10. Import in specialists/__init__.py to trigger registration

## Tool handler pattern

```python
async def _handle_my_tool(self, state, variable: str) -> ToolResult:
    try:
        df = pd.read_parquet(state.dataframe_path)
        result = compute_something(df, variable)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"variable": variable, "result": result},
        )
    except Exception as e:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=str(e),
        )
```

Never raise exceptions from tool handlers. Always return ToolResult.

## Commit pattern

1. First commit: agent module with tools registered (~60-80 lines)
2. Second commit: tool handlers implemented (~60-80 lines)
3. Third commit: state field added, import trigger, integration (~30 lines)
