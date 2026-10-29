# 16. Shared infrastructure and verification

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md) · [Evidence and discrepancies](evidence.md)

## Purpose and boundary

Supplies the cross-cutting rules all components rely on. Configuration, model access, evidence addresses, tracing, dependency enforcement and verification are separate concerns with named owners. This chapter is not a destination for unrelated business logic.

## Internal parts

| Concern | Definition/entrypoint | Consumers |
|---|---|---|
| Configuration | [config](../../causal_agent/common/config.py) | All path/model/sandbox users; server Settings |
| Model calls | [structured/decide](../../causal_agent/common/llm.py), [decider](../../causal_agent/common/decider.py), [run_episode](../../causal_agent/lane/episode.py) | Desk judgments, family reasoning, exploration |
| Addresses/numbers | [addresses](../../causal_agent/common/addresses.py), [numbers](../../causal_agent/common/numbers.py) | Memory, scoped views, chat, figures and gates |
| Shared records | [contracts](../../causal_agent/common/contracts), [prompts](../../causal_agent/common/prompts.py) | Component boundary types and shared wording |
| Tracing | [trace](../../causal_agent/common/trace.py) | Every model/tool/gate/process/file chokepoint |
| Dependency rules | [import-linter contracts](../../pyproject.toml) | Python packages and family sublayers |
| Verification/tooling | [Makefile](../../Makefile), [test isolation](../../causal_agent/conftest.py), [evals](../../causal_agent/evals), [CI](../../.github/workflows/ci.yml), [web commands](../../web/package.json) | Developers and acceptance work |

## Inputs and outputs

Config turns environment values into typed path/model/sandbox/decider settings. Model wrappers consume prompts and schemas and return records with thoughts. Trace helpers consume inputs/outputs, gate errors and files. Evaluators consume case declarations and real or scripted runs, depending on the entrypoint.

## Contracts

| Concern | Enforcement and failure |
|---|---|
| Model entrypoints | Calls go through `structured`, `decide` or `run_episode`; trace-policy tests inspect bypasses |
| Citation authority | Addresses resolve in the supplied material; numeric checks guard claims, not only formatting |
| Trace continuity | Named graphs; `child_env` passes parent/thread; child uses `trace.parent`; `trace.gate` records rejected tries; `trace.write` attaches products |
| Import ownership | Bottom to top: common → profile → memory → viz → lane → explore → families → desk → server; lower layers cannot import higher ones except declared test exceptions |
| Family separation | Core cannot name family/engine policy; reasoning and effect execution have additional forbidden-import contracts |
| Wire freshness | Server models → OpenAPI → generated TypeScript; schema generation must leave no unexpected diff |
| Ladder freshness | Family declarations → generated ladder docs/trajectory figures; source declarations remain authoritative |

The current discontinuity import contract names several forbidden engine modules that are not current implementation files. Record this as an enforcement uncertainty, not proof that every intended effect boundary is statically protected.

## Decision authority

Code controls configuration, gates and structural checks. Gemini performs structured judgments; Laya/Jev classifies turns through the decider interface. Humans supply causal context and authorize runs. Tests and real-model acceptance demonstrate different things.

## Dependencies

No new external service is introduced by this handbook. Vertex/Gemini, the configured decider, LangSmith and optional Docker are existing runtime dependencies. Operational details must be checked against installed code/config before changing them.

## State and storage

`.env` provides local configuration without becoming documentation evidence to publish. Runtime stores are in [persistence](09-persistence.md). Eval fixtures and handoffs are source inputs; generated build/cache output and live analysis output have distinct lifecycles. Archived ADRs describe historical decisions and do not override current code or project rules.

## Worked walkthrough

When a component contract changes later: update its authoritative schema and consumers → run relevant unit/numerical checks → verify import and trace policy → regenerate affected wire/ladder products → for changed judgments, gates, prompts, rungs or desk behavior, run real Gemini analysis and inspect its trace. A green scripted test alone does not establish model behavior.

## Verification

**Code-verified:** configuration, build targets and enforcement definitions. **Test-supported:** [trace policy](../../causal_agent/common/tests/test_trace_policy.py), [trace artifacts](../../causal_agent/common/tests/test_trace_artifacts.py), [core ownership](../../causal_agent/families/tests/test_core_names_no_family.py), [ladders](../../causal_agent/families/tests/test_ladders.py) and package suites.

Useful commands: `LANGSMITH_TRACING=false uv run pytest <package> -q`, `uv run lint-imports`, `npm --prefix web test`, and `make check`. `make check` includes schema/ladder regeneration; `make fmt` edits source and `make studio` clears Studio state. None is a read-only documentation checker. This pass validates handbook structure/links/ownership and records its exact checks in [evidence](evidence.md), without claiming a fresh model run or full suite result.
