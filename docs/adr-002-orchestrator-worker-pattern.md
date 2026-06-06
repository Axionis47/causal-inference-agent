# ADR-002: Orchestrator-Worker Pattern with Structured Briefs

**Status**: Accepted
**Date**: 2026-05-29

This document records the design decisions for the `analyse` stage refactor. It explains why the orchestrator and its specialist workers are being reshaped around a capability menu plus typed-brief contract, and why we are staying with custom Python rather than migrating to LangGraph, CrewAI, AutoGen, or OpenAI Agents SDK.

Coding practices for the change live in `CLAUDE.md` (300 LOC cap, one agent per file, test audit checklist, commit style, SSE contract). This ADR is about *what* we are building, not *how* to commit it.

---

## 1. The pattern: orchestrator-workers on a Pydantic blackboard

**Choice**: A central LLM-driven orchestrator delegates to specialist workers. Each worker declares a one-line capability (the menu entry) and returns a typed `AgentBrief` summarising what it did. Full worker output stays in `AnalysisState` (the blackboard); the brief is the only thing the orchestrator reads.

**Why this pattern**:
- Named in [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) as the "orchestrator-workers" pattern. Same shape powers Anthropic's [multi-agent research system in production](https://www.anthropic.com/engineering/multi-agent-research-system).
- LangGraph calls it the "supervisor" pattern. OpenAI Agents SDK calls the typed-return variant "structured handoffs".
- The pull-based `ContextTools` mixin already in this codebase is a blackboard read API. The brief just adds a typed write channel for orchestrator-relevant signal.

**Why not free-form messages or full-state dumps**:
- Free-form messages bloat the orchestrator's prompt linearly in worker output size. After 6 dispatches, the orchestrator is reading nested JSON to decide one move.
- Full-state dumps dilute LLM attention. Bigger prompt, worse routing decisions, longer latency.

## 2. Stay custom, do not migrate to a framework

**Choice**: Keep the bespoke Python (FastAPI, Pydantic state, `@register_agent` decorator). Add an `AgentBrief` Pydantic class and an `AgentCapability` declarative class attribute. Do not adopt LangGraph, CrewAI, AutoGen, or OpenAI Agents SDK.

**Why custom wins here**:
- Anthropic's own guidance: "Start by using LLM APIs directly. Many patterns can be implemented in a few lines of code. Frameworks often create extra layers of abstraction that can obscure the underlying prompts and responses."
- The repo already has 90% of what frameworks ship: registry, Pydantic state, pull-based context, critique loop, multi-LLM adapter (Vertex AI, Claude, Gemini), SSE streaming, Firestore checkpointing.
- The two missing pieces (typed brief, declarative capability) are roughly 30 lines of Pydantic.

**Why not LangGraph**: 2 to 4 week migration. Pydantic-vs-TypedDict friction with our state schema. Lose the Firestore checkpointer (replace with custom `BaseCheckpointSaver`). Net gain is small relative to cost.

**Why not OpenAI Agents SDK**: `handoff(input_type=PydanticModel)` is the cleanest typed-brief primitive in any framework, but the SDK is OpenAI-models-first. Adopting it costs the multi-LLM goal.

**Why not CrewAI**: Hierarchical process plus `output_pydantic` has [open bugs](https://github.com/crewAIInc/crewAI/issues/792). The framework assumes per-task isolated context, not a shared blackboard. Multi-month rewrite for a worse fit.

**Why not AutoGen**: Message-passing first. Replacing the pull-based context model with chat history would be a regression. The `SelectorGroupChat` prompt template is worth borrowing as a reference for the capability menu, nothing more.

## 3. Flag vocabulary: closed enum

**Choice**: A closed `Flag` enum defined centrally in `domain/briefs.py`. Every worker raises flags from this enum only. Initial set is roughly 20 kinds, growing only when a new condition genuinely does not fit existing kinds.

**Why closed enum**:
- Orchestrator can scan flags uniformly and detect contradictions, convergence, or stalls.
- Critique's `reroute_to` reasoning depends on shared vocabulary across workers.
- Tests can assert specific flag kinds rather than fuzzy-matching strings.

**Why not free-form strings**: Two workers raising the same issue with different wording would be treated as separate problems by the orchestrator. Same-issue-different-wording bugs are silent until reroutes misfire.

**Why not open strings with a normalizer layer**: You end up maintaining the strings AND the normalizer. When new wording appears the normalizer does not recognise, the orchestrator silently miss-classifies. The cost reappears as flaky reroutes weeks later.

## 4. Reasoning pattern per agent (informational, not prescriptive)

**Choice**: The contract migration is **purely additive**. Every agent keeps its existing reasoning pattern. We do not convert ReAct agents to CoT, or vice versa, as part of this work. Adding the contract layer (capability declaration, typed brief, preflight refusal) does not require touching the LLM loop inside each agent.

The pattern catalogue below is informational. It tells future readers how the contract layer integrates with each agent's internal flow, so the brief-production code knows whether to plug into the tail of a ReAct loop, the return value of a single LLM call, or the end of a pure-compute function.

**Current patterns in the repo (as observed, not as a target)**:
- **Direct, no LLM in the analysis loop**: `notebook_generator` template assembly.
- **CoT (one LLM call after compute)**: parts of `data_profiler`, `eda_agent`, `causal_discovery`, `sensitivity_analyst`, `domain_knowledge`, `critique`.
- **Full ReAct**: `ps_diagnostics`, `effect_estimator`, `data_repair`, `confounder_discovery`, `dag_expert`.

If a future iteration of this codebase decides to reshape one of these (for example, to bound LLM cost), that is a separate ADR. ADR-002 is silent on internal pattern changes.

## 5. Orchestrator prompt shape

**Choice**: A deterministic (non-LLM) assembler builds the orchestrator's prompt every turn from four sources:

1. **Capability menu**: static, joined from `AgentCapability` class attributes (~12 lines).
2. **Status board**: one row per worker, built from briefs (`done` / `pending` / `failed` plus headline).
3. **Open issues**: recent `raised_issues` strings, normalised by `Flag` kind.
4. **Unblocked next**: workers whose `needs` are satisfied, computed mechanically from state.

**Why mechanical assembly**:
- Prompt grows linearly in number of dispatches (one line per finished worker), not in worker output size.
- Always under ~50 lines regardless of dataset size or analysis depth.
- Reproducible, auditable, testable independent of the LLM.

**Why not an LLM-summarised state**: Adds latency and cost. Re-introduces the diluted-attention problem we are trying to fix.

## 6. Migration order

**Choice**: Six phases, one PR per phase.

0. **Foundations**: `AgentBrief`, `AgentCapability`, `Flag` enum in `backend/src/domain/briefs.py`. Tests for the Pydantic shape, not behaviour. No agents touched.
1. **Reference agent**: `ps_diagnostics` returns `AgentBrief` alongside current state writes. Additive change, orchestrator still works as-is.
2. **Orchestrator slice**: orchestrator consumes `ps_diagnostics`'s brief and capability. Falls back to existing logic for the other 11 agents.
3. **Migrate remaining agents in batches**: A (CoT small), B (CoT foundational, `data_profiler`), C (Simple ReAct), D (Full ReAct, `effect_estimator`), E (terminal: critique, notebook).
4. **Orchestrator cut-over**: drop the old state-dump prompt path. Critique routes via `reroute_to`.
5. **Cleanup**: remove dead code, unused metadata fields, stale prompt templates.

Each phase ends with a test audit and a recap per the working-mode rule in `CLAUDE.md` §1.

---

**Sources**

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Anthropic, Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [LangGraph Supervisor pattern](https://github.com/langchain-ai/langgraph-supervisor-py)
- [OpenAI Agents SDK, Handoffs](https://openai.github.io/openai-agents-python/handoffs/)
- [CrewAI Tasks documentation](https://docs.crewai.com/en/concepts/tasks)
- [AutoGen Selector Group Chat](https://microsoft.github.io/autogen/dev//user-guide/agentchat-user-guide/selector-group-chat.html)
