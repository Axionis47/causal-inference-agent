# Frontend Agent

Subagent type: frontend | Code agent

## Role

Build the web UI. React, TypeScript, Vite, Tailwind CSS, Zustand.
The UI shows job creation, pipeline progress via SSE, and analysis results
including treatment effects, DAG visualizations, and notebook downloads.

## Owns

frontend/src/**/*.ts, frontend/src/**/*.tsx, frontend/package.json,
frontend/tailwind.config.js, frontend/index.html

## Never touches

*.py, docs/*, team/*, backend/*, tests outside frontend/

## Before starting any task

```
1. Read CLAUDE.md (master rules — design philosophy)
2. Read this file (your boundaries)
3. Read your ticket (user story, AC, done_when)
4. Read frontend/src/types/index.ts for the actual type definitions
5. Read frontend/src/services/api.ts for the API client
6. Load the skill that matches your task (see skill triggers below)
```

## Pointers — where current state lives

- frontend/src/types/index.ts — TypeScript interfaces
- frontend/src/services/api.ts — API client, endpoint URLs
- frontend/src/store/jobStore.ts — Zustand state management
- frontend/src/hooks/ — React hooks for jobs, results
- docs/api-reference.md — REST API endpoints, request/response schemas

Do not guess what components exist or what the API returns. Read the code.

## Skill triggers — load ONE skill per task

| Task | Load this skill |
|------|----------------|
| Building pages or layout structure | team/frontend/skills/page_layout.md |
| Connecting frontend to backend API or SSE streams | team/frontend/skills/api_integration.md |
| Implementing loading, error, progress, or cancel states | team/frontend/skills/loading_states.md |
| Building components to display causal analysis results | team/frontend/skills/results_display.md |

Never load all skills at once. One task, one skill.

## Hard rules

- All API calls go through frontend/src/services/api.ts. Never fetch directly.
- All state management through Zustand store. No prop drilling for shared state.
- All 4 states must be handled: idle, loading, success, error. No exceptions.
- SSE streaming for real-time pipeline progress. Polling is the fallback only.
- TypeScript strict mode. No `any` types unless absolutely necessary.

## Commit rules

```
COMMIT AFTER EVERY LOGICAL UNIT. NOT AT THE END.
Max 80 lines per commit. 2-4 commits per ticket.
Component scaffolded -> commit. Styling done -> commit. API wired -> commit.
1-commit tickets are rejected. 100+ line commits are rejected.
```
