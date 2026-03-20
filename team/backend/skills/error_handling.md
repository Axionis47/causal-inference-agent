# Skill: Error Handling

Load when: implementing error handling in agents or pipeline.

## Error handling layers

### Tool handlers (innermost)
```python
async def _handle_tool(self, state, **kwargs) -> ToolResult:
    try:
        result = do_computation()
        return ToolResult(status=ToolResultStatus.SUCCESS, output=result)
    except (ValueError, np.linalg.LinAlgError) as e:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error=str(e))
```
Expected errors (convergence, singular matrix, insufficient data) return ToolResult.ERROR.
The ReAct loop sees this as an observation and can try a different approach.

### ReActAgent.execute() (middle)
- Wrapped in asyncio.wait_for(timeout=300s)
- TimeoutError: adds trace, returns state unchanged, does NOT mark failed
- 3 consecutive tool errors: loop breaks early
- Uncaught exception: caught by execute_with_tracing()

### BaseAgent.execute_with_tracing() (outer)
- Catches any remaining exceptions
- Calls state.mark_failed(error, agent_name)
- Adds error trace

### JobManager._run_job() (outermost)
- Catches all exceptions from orchestrator
- Calls state.mark_failed() with full context
- Total job timeout: agent_timeout_seconds * 10

## Rules

1. Tool handlers: catch expected errors, return ToolResult.ERROR
2. Never use bare except: pass — always log
3. Numerical failures (NaN, infinity, singular matrix) are expected — handle gracefully
4. state.mark_failed() is for unrecoverable errors only
5. Timeout returns state unchanged — does not mark as failed
6. Agent timeout (300s) and job timeout (3000s) are configurable via settings
