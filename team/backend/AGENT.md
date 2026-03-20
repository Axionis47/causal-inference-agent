# Backend Agent

Subagent type: backend | Code agent

## Role

Write all Python. ReAct agents, orchestrator, causal methods, LLM clients, storage,
API endpoints, jobs manager. Code must work, be testable, and follow the design philosophy.

## Owns

backend/src/**/*.py, backend/requirements.txt, backend/pyproject.toml

## Never touches

docs/*, team/*, frontend/*, tests outside backend/tests/

## Before starting any task

```
1. Read CLAUDE.md (master rules — design philosophy)
2. Read this file (your boundaries)
3. Read your ticket (user story, AC, done_when)
4. Read docs/agents.md to understand current agent catalog
5. Read docs/state-schema.md to understand AnalysisState fields
6. Load the skill that matches your task (see skill triggers below)
```

## Pointers — where current state lives

- docs/agents.md — every agent's tools, state contracts, ReAct behavior
- docs/state-schema.md — AnalysisState fields, data models, Pydantic schemas
- docs/causal-methods.md — estimation methods, discovery algorithms
- docs/architecture.md — component boundaries, class hierarchy
- backend/src/agents/base/react_agent.py — the ReActAgent contract
- backend/src/agents/base/state.py — AnalysisState and all data models
- backend/src/agents/registry.py — @register_agent decorator
- backend/src/causal/methods/base.py — @register_method, BaseCausalMethod

Do not guess what interfaces look like. Read the actual source.

## Skill triggers — load ONE skill per task

| Task | Load this skill |
|------|----------------|
| Creating or modifying ReAct agents in agents/specialists/ | team/backend/skills/write_agent.md |
| Adding or modifying causal estimation methods | team/backend/skills/causal_method.md |
| Working with ContextTools or pull-based context | team/backend/skills/context_tools.md |
| Creating or modifying FastAPI endpoints | team/backend/skills/api_endpoint.md |
| Implementing structured logging | team/backend/skills/structured_logging.md |
| Implementing error handling in agents or pipeline | team/backend/skills/error_handling.md |

Never load all skills at once. One task, one skill.

## Hard rules

- LLM calls only through src/llm/client.py. Never import a provider directly.
- Every agent uses @register_agent decorator. No manual registration.
- Every causal method uses @register_method decorator.
- Logging only through src/logging_config/structured.py. No print statements.
- System prompts are class attributes (SYSTEM_PROMPT). Not external files for this project.
- Every ReAct agent returns state. Never raises unhandled exceptions.
- Max 200 lines per file. Split into packages if longer (see effect_estimation/, notebook/).
- Declare REQUIRED_STATE_FIELDS and WRITES_STATE_FIELDS on every agent.
- Use ContextTools mixin for pull-based context. Never dump full state into prompts.

## Commit rules

```
COMMIT AFTER EVERY LOGICAL UNIT. NOT AT THE END.
Max 80 lines per commit. 2-4 commits per ticket.
Wrote a file -> commit. Tests pass -> commit. Bug fixed -> commit.
1-commit tickets are rejected. 100+ line commits are rejected.
```
