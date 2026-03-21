# Skill: C4 Architecture Diagrams

Load when: creating or updating architecture diagrams.

## What C4 is

C4 is a 4-level architecture model by Simon Brown. Each level zooms in.

### Level 1 — Context
Who uses the system? What external systems does it connect to?
No internal details. One box for the entire system.

```
[User/Browser] --> [Causal Orchestrator] --> [Vertex AI / Gemini / Claude]
                                         --> [Kaggle API]
                                         --> [Firestore / Local Storage]
```

Rules: no file names, no class names, no internal components. Just system boundaries.

### Level 2 — Container
What are the major deployable/runnable pieces?

```
[React Frontend (Vite)] --> [FastAPI Backend]
[FastAPI Backend] --> [LLM Provider (Vertex AI)]
[FastAPI Backend] --> [Storage (Firestore / Local JSON)]
[FastAPI Backend] --> [Kaggle Downloader]
```

Rules: one box per deployable unit. Name the technology. Show data flow direction.

### Level 3 — Component
What's inside each container? Show the major modules.

```
Backend contains:
  [API Layer (FastAPI routes)]
    --> [Job Manager (lifecycle, semaphore, timeout)]
      --> [Orchestrator (LLM-driven dispatch)]
        --> [13 Specialist Agents (ReAct pattern)]
        --> [Critique Agent (quality review)]
      --> [Notebook Generator]
  [Agents] --> [Causal Methods Layer (12 methods)]
  [Agents] --> [LLM Client (Protocol-based)]
  [Job Manager] --> [Storage Layer (Protocol-based)]
```

Rules: show dependencies between components. Name patterns (ReAct, Protocol). Show cardinality (13 agents, 12 methods).

### Level 4 — Code
Class and function level. Auto-generated from code, not hand-maintained.
Skip this level. It changes too often.

## Output format

Use Mermaid syntax for all diagrams. Every diagram must:
1. Have a title
2. Fit on one screen
3. Show data flow direction (arrows)
4. Label every box with its role, not just its name
5. Live in docs/architecture/

## Files to produce

```
docs/architecture/
  c4_context.md      — Level 1
  c4_containers.md   — Level 2
  c4_components.md   — Level 3
```
