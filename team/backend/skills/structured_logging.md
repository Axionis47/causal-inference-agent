# Skill: Structured Logging

Load when: implementing or modifying logging.

## Logger setup

```python
from src.logging_config.structured import get_logger

logger = get_logger(__name__)
```

## Logging pattern

Structured events, not formatted strings:

```python
# Good
logger.info("agent_completed", job_id=state.job_id, agent=self.AGENT_NAME, duration_ms=elapsed)
logger.warning("tool_execution_failed", tool=tool_name, error=str(e))
logger.debug("state_field_updated", field="treatment_effects", method_count=len(results))

# Bad
logger.info(f"Agent {self.AGENT_NAME} completed job {state.job_id} in {elapsed}ms")
```

## Rules

1. Use get_logger(__name__) — never import logging directly
2. Event names are snake_case: agent_completed, tool_failed, state_updated
3. Pass structured key-value pairs, not f-strings
4. Never log full DataProfile or large state objects
5. Log at appropriate levels: debug for internals, info for milestones, warning for recoverable errors
6. Never use bare except: pass — always log with at least logger.debug(event, exc_info=True)
