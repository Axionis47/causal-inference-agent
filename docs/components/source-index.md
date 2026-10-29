# Source ownership index

[Component map](README.md) · [Evidence](evidence.md)

This inventory assigns every current Python source module, frontend source/style file, family declaration and retained eval input, plus shared build/tooling files, to one primary functional owner. Shared consumers are described in the chapters. Package initializers follow their package owner. Test fixtures/fakes follow the behavior they support; cross-family architecture tests belong to infrastructure.

It describes the working tree, including untracked diagnostic modules, rather than only committed HEAD. Generated API schema is assigned to the wire owner. Source data under `data/` belongs to dataset intake as registered input, with profiling/memory consuming it; it is not enumerated row by row. Documentation belongs to its subject component. Ignored runtime output, `.venv`, node modules, bytecode, build output and `.git` are outside the production-source inventory.

A file-level owner is a navigation aid, not a claim that every function belongs to one component. For example, server `datasets.py` owns intake but projects profile summaries for component 03; server models define wire representations of many components; desk interview nodes call readiness and exploration. Those seams are explicit in the component chapters.

The tables retain the component ownership inventory and include the consolidated chat additions below.

## 01. [landing](01-landing.md)

| Directory | Owned files |
|---|---|
| `web/src` | [App.tsx](../../web/src/App.tsx), [main.tsx](../../web/src/main.tsx) |
| `web/src/components` | [ConfirmDialog.tsx](../../web/src/components/ConfirmDialog.tsx), [DatasetCard.tsx](../../web/src/components/DatasetCard.tsx), [Shell.tsx](../../web/src/components/Shell.tsx), [Sidebar.tsx](../../web/src/components/Sidebar.tsx), [TopBar.tsx](../../web/src/components/TopBar.tsx) |
| `web/src/pages` | [Landing.tsx](../../web/src/pages/Landing.tsx), [DatasetDetails.tsx](../../web/src/pages/DatasetDetails.tsx) |
| `web/src/hooks` | [useDatasetLibrary.ts](../../web/src/hooks/useDatasetLibrary.ts) |
| `causal_agent/server/ingestion` | [library.py](../../causal_agent/server/ingestion/library.py) |

## 02. [intake](02-intake.md)

| Directory | Owned files |
|---|---|
| `causal_agent/profile` | [datasets.py](../../causal_agent/profile/datasets.py) |
| `causal_agent/server` | [datasets.py](../../causal_agent/server/datasets.py) |
| `causal_agent/server/tests` | [test_datasets.py](../../causal_agent/server/tests/test_datasets.py) |
| `web/src/components` | [UploadDrop.tsx](../../web/src/components/UploadDrop.tsx) |
| `web/src/pages` | [NewDataset.tsx](../../web/src/pages/NewDataset.tsx) |

## 03. [profiling](03-profiling.md)

| Directory | Owned files |
|---|---|
| `causal_agent/profile` | [__init__.py](../../causal_agent/profile/__init__.py), [data.py](../../causal_agent/profile/data.py), [pack.py](../../causal_agent/profile/pack.py), [profiler.py](../../causal_agent/profile/profiler.py) |
| `causal_agent/profile/tests` | [__init__.py](../../causal_agent/profile/tests/__init__.py), [test_pack.py](../../causal_agent/profile/tests/test_pack.py), [test_profiler.py](../../causal_agent/profile/tests/test_profiler.py) |

Phase 3 additions: [inspection](../../causal_agent/profile/inspection.py), [report contracts](../../causal_agent/profile/report.py), [owned service/schema/routes](../../causal_agent/server/profiling/), [migration](../../migrations/versions/0003_profiling.py), [storage/API tests](../../causal_agent/server/tests/profiling/), [column calculations](../../causal_agent/profile/tests/test_inspection_columns.py), [dataset calculations](../../causal_agent/profile/tests/test_inspection_dataset.py), [failure tests](../../causal_agent/profile/tests/test_inspection_failures.py), [profile UI](../../web/src/components/DatasetProfile.tsx) and [UI tests](../../web/src/components/DatasetProfile.test.tsx).

## 04. [conversation ui](04-conversation-ui.md)

| Directory | Owned files |
|---|---|
| `web/src` | [layout.test.ts](../../web/src/layout.test.ts), [layout.ts](../../web/src/layout.ts),  |
| `web/src/components` | ,  |
| `web/src/hooks` | [useNarrow.ts](../../web/src/hooks/useNarrow.ts), , [useStoredState.ts](../../web/src/hooks/useStoredState.ts) |
| `web/src/pages` | [Chat.tsx](../../web/src/pages/Chat.tsx) |
| `web/src/styles` | [app.css](../../web/src/styles/app.css), [tokens.css](../../web/src/styles/tokens.css) |

## 05. [api sessions](05-api-sessions.md)

| Directory | Owned files |
|---|---|
| `causal_agent/server` | [__init__.py](../../causal_agent/server/__init__.py), [__main__.py](../../causal_agent/server/__main__.py), [app.py](../../causal_agent/server/app.py), [artifacts.py](../../causal_agent/server/artifacts.py), [models.py](../../causal_agent/server/models.py), [runs.py](../../causal_agent/server/runs.py), [settings.py](../../causal_agent/server/settings.py), [views.py](../../causal_agent/server/views.py) |
| `causal_agent/server/tests` | [__init__.py](../../causal_agent/server/tests/__init__.py), [conftest.py](../../causal_agent/server/tests/conftest.py), [test_sessions.py](../../causal_agent/server/tests/test_sessions.py) |
| `web` | [openapi.json](../../web/openapi.json) |
| `web/src` | [api.ts](../../web/src/api.ts), [types.ts](../../web/src/types.ts) |
| `web/src/generated` | [schema.ts](../../web/src/generated/schema.ts) |

## 06. [desk](06-desk.md)

| Directory | Owned files |
|---|---|
| `causal_agent/desk` | [__init__.py](../../causal_agent/desk/__init__.py), [contracts.py](../../causal_agent/desk/contracts.py), [decider.yaml](../../causal_agent/desk/decider.yaml), [explainer.py](../../causal_agent/desk/explainer.py), [graph.py](../../causal_agent/desk/graph.py), [readback.py](../../causal_agent/desk/readback.py), [reader.py](../../causal_agent/desk/reader.py), [relations.py](../../causal_agent/desk/relations.py), [route.py](../../causal_agent/desk/route.py), [state.py](../../causal_agent/desk/state.py), [turns.py](../../causal_agent/desk/turns.py) |
| `causal_agent/desk/evals` | [turns.jsonl](../../causal_agent/desk/evals/turns.jsonl) |
| `causal_agent/desk/nodes` | [__init__.py](../../causal_agent/desk/nodes/__init__.py), [frame.py](../../causal_agent/desk/nodes/frame.py), [interview.py](../../causal_agent/desk/nodes/interview.py), [question.py](../../causal_agent/desk/nodes/question.py), [shared.py](../../causal_agent/desk/nodes/shared.py) |
| `causal_agent/desk/prompts` | [__init__.py](../../causal_agent/desk/prompts/__init__.py), [journey.py](../../causal_agent/desk/prompts/journey.py), [routing.py](../../causal_agent/desk/prompts/routing.py) |
| `causal_agent/desk/tests` | [__init__.py](../../causal_agent/desk/tests/__init__.py), [fakes.py](../../causal_agent/desk/tests/fakes.py), [test_relations.py](../../causal_agent/desk/tests/test_relations.py), [test_route.py](../../causal_agent/desk/tests/test_route.py), [test_turn_kinds.py](../../causal_agent/desk/tests/test_turn_kinds.py) |

## 07. [memory](07-memory.md)

| Directory | Owned files |
|---|---|
| `causal_agent/memory` | [__init__.py](../../causal_agent/memory/__init__.py), [catalogue.py](../../causal_agent/memory/catalogue.py), [checks.py](../../causal_agent/memory/checks.py), [checks.yaml](../../causal_agent/memory/checks.yaml), [claims.py](../../causal_agent/memory/claims.py), [facts.py](../../causal_agent/memory/facts.py), [fields.yaml](../../causal_agent/memory/fields.yaml), [ops.py](../../causal_agent/memory/ops.py), [probes.py](../../causal_agent/memory/probes.py), [records.py](../../causal_agent/memory/records.py), [table.py](../../causal_agent/memory/table.py), [views.py](../../causal_agent/memory/views.py) |
| `causal_agent/memory/tests` | [__init__.py](../../causal_agent/memory/tests/__init__.py), [test_facts.py](../../causal_agent/memory/tests/test_facts.py), [test_memory.py](../../causal_agent/memory/tests/test_memory.py) |

## 08. [readiness handoff](08-readiness-handoff.md)

| Directory | Owned files |
|---|---|
| `causal_agent/common/contracts` | [brief.py](../../causal_agent/common/contracts/brief.py), [endpoints.py](../../causal_agent/common/contracts/endpoints.py), [frame.py](../../causal_agent/common/contracts/frame.py), [pack.py](../../causal_agent/common/contracts/pack.py), [text.py](../../causal_agent/common/contracts/text.py) |
| `causal_agent/desk` | [designer.py](../../causal_agent/desk/designer.py), [handoff.py](../../causal_agent/desk/handoff.py), [pipeline.py](../../causal_agent/desk/pipeline.py) |
| `causal_agent/desk/nodes` | [decide.py](../../causal_agent/desk/nodes/decide.py) |
| `causal_agent/desk/tests` | [test_handoff.py](../../causal_agent/desk/tests/test_handoff.py), [test_pipeline.py](../../causal_agent/desk/tests/test_pipeline.py) |
| `causal_agent/families` | [__init__.py](../../causal_agent/families/__init__.py), [base.py](../../causal_agent/families/base.py), [registry.py](../../causal_agent/families/registry.py) |
| `causal_agent/families/declared` | [__init__.py](../../causal_agent/families/declared/__init__.py), [instrument.yaml](../../causal_agent/families/declared/instrument.yaml), [interrupted_series.yaml](../../causal_agent/families/declared/interrupted_series.yaml), [probes.py](../../causal_agent/families/declared/probes.py), [root_cause.yaml](../../causal_agent/families/declared/root_cause.yaml), [synthetic_control.yaml](../../causal_agent/families/declared/synthetic_control.yaml) |
| `causal_agent/lane` | [endpoints.py](../../causal_agent/lane/endpoints.py) |
| `causal_agent/memory` | [matrix.py](../../causal_agent/memory/matrix.py) |
| `causal_agent/memory/tests` | [test_matrix.py](../../causal_agent/memory/tests/test_matrix.py) |

## 09. [persistence](09-persistence.md)

| Directory | Owned files |
|---|---|
| `causal_agent/memory` | [artifacts.py](../../causal_agent/memory/artifacts.py), [journal.py](../../causal_agent/memory/journal.py), [store.py](../../causal_agent/memory/store.py), [versions.py](../../causal_agent/memory/versions.py) |
| `causal_agent/memory/tests` | [test_artifacts.py](../../causal_agent/memory/tests/test_artifacts.py), [test_journal.py](../../causal_agent/memory/tests/test_journal.py), [test_versions.py](../../causal_agent/memory/tests/test_versions.py) |

## 10. [analysis runtime](10-analysis-runtime.md)

| Directory | Owned files |
|---|---|
| `causal_agent/common/contracts` | [artifacts.py](../../causal_agent/common/contracts/artifacts.py), [frozen.py](../../causal_agent/common/contracts/frozen.py) |
| `causal_agent/lane` | [__init__.py](../../causal_agent/lane/__init__.py), [episode.py](../../causal_agent/lane/episode.py), [figures.py](../../causal_agent/lane/figures.py), [graph.py](../../causal_agent/lane/graph.py), [intake.py](../../causal_agent/lane/intake.py), [knowledge.py](../../causal_agent/lane/knowledge.py), [ladder.py](../../causal_agent/lane/ladder.py), [ledger.py](../../causal_agent/lane/ledger.py), [nodes.py](../../causal_agent/lane/nodes.py), [prompts.py](../../causal_agent/lane/prompts.py), [records.py](../../causal_agent/lane/records.py), [rungs.py](../../causal_agent/lane/rungs.py), [space.py](../../causal_agent/lane/space.py), [state.py](../../causal_agent/lane/state.py), [terminal.py](../../causal_agent/lane/terminal.py), [tools.py](../../causal_agent/lane/tools.py), [words.py](../../causal_agent/lane/words.py) |
| `causal_agent/lane/tests` | [__init__.py](../../causal_agent/lane/tests/__init__.py), [test_episode.py](../../causal_agent/lane/tests/test_episode.py), [test_figures.py](../../causal_agent/lane/tests/test_figures.py), [test_intake.py](../../causal_agent/lane/tests/test_intake.py), [test_knowledge.py](../../causal_agent/lane/tests/test_knowledge.py), [test_ledger.py](../../causal_agent/lane/tests/test_ledger.py), [test_rungs.py](../../causal_agent/lane/tests/test_rungs.py), [test_space.py](../../causal_agent/lane/tests/test_space.py), [test_tools.py](../../causal_agent/lane/tests/test_tools.py), [test_words.py](../../causal_agent/lane/tests/test_words.py) |

## 11. [adjustment](11-adjustment.md)

| Directory | Owned files |
|---|---|
| `causal_agent/families/adjustment` | [__init__.py](../../causal_agent/families/adjustment/__init__.py), [design.py](../../causal_agent/families/adjustment/design.py), [family.yaml](../../causal_agent/families/adjustment/family.yaml), [handoff.py](../../causal_agent/families/adjustment/handoff.py), [overlap.py](../../causal_agent/families/adjustment/overlap.py), [postviz.py](../../causal_agent/families/adjustment/postviz.py), [probes.py](../../causal_agent/families/adjustment/probes.py) |
| `causal_agent/families/adjustment/evals` | [__init__.py](../../causal_agent/families/adjustment/evals/__init__.py), [cases.yaml](../../causal_agent/families/adjustment/evals/cases.yaml), [evaluators.py](../../causal_agent/families/adjustment/evals/evaluators.py), [summarise.py](../../causal_agent/families/adjustment/evals/summarise.py) |
| `causal_agent/families/adjustment/evals/handoffs` | [gov_transfers_forced.json](../../causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json) |
| `causal_agent/families/adjustment/lane` | [__init__.py](../../causal_agent/families/adjustment/lane/__init__.py), [artifacts.py](../../causal_agent/families/adjustment/lane/artifacts.py), [checks.py](../../causal_agent/families/adjustment/lane/checks.py), [contracts.py](../../causal_agent/families/adjustment/lane/contracts.py), [execute.py](../../causal_agent/families/adjustment/lane/execute.py), [graph.py](../../causal_agent/families/adjustment/lane/graph.py), [nodes.py](../../causal_agent/families/adjustment/lane/nodes.py), [prompts.py](../../causal_agent/families/adjustment/lane/prompts.py), [reason.py](../../causal_agent/families/adjustment/lane/reason.py), [selection.py](../../causal_agent/families/adjustment/lane/selection.py), [state.py](../../causal_agent/families/adjustment/lane/state.py), [support.py](../../causal_agent/families/adjustment/lane/support.py), [views.py](../../causal_agent/families/adjustment/lane/views.py) |
| `causal_agent/families/adjustment/lane/engine` | [__init__.py](../../causal_agent/families/adjustment/lane/engine/__init__.py), [estimators.py](../../causal_agent/families/adjustment/lane/engine/estimators.py), [falsify.py](../../causal_agent/families/adjustment/lane/engine/falsify.py), [graph_tests.py](../../causal_agent/families/adjustment/lane/engine/graph_tests.py), [model.py](../../causal_agent/families/adjustment/lane/engine/model.py), [sensitivity.py](../../causal_agent/families/adjustment/lane/engine/sensitivity.py) |
| `causal_agent/families/adjustment/lane/knowledge` | [__init__.py](../../causal_agent/families/adjustment/lane/knowledge/__init__.py), [checks.yaml](../../causal_agent/families/adjustment/lane/knowledge/checks.yaml), [endpoints.yaml](../../causal_agent/families/adjustment/lane/knowledge/endpoints.yaml), [estimators.yaml](../../causal_agent/families/adjustment/lane/knowledge/estimators.yaml), [falsifications.yaml](../../causal_agent/families/adjustment/lane/knowledge/falsifications.yaml), [motifs.yaml](../../causal_agent/families/adjustment/lane/knowledge/motifs.yaml), [rivals.yaml](../../causal_agent/families/adjustment/lane/knowledge/rivals.yaml), [rungs.yaml](../../causal_agent/families/adjustment/lane/knowledge/rungs.yaml), [sensitivity.yaml](../../causal_agent/families/adjustment/lane/knowledge/sensitivity.yaml), [space.yaml](../../causal_agent/families/adjustment/lane/knowledge/space.yaml) |
| `causal_agent/families/adjustment/tests` | [__init__.py](../../causal_agent/families/adjustment/tests/__init__.py), [test_lane.py](../../causal_agent/families/adjustment/tests/test_lane.py), [test_motifs.py](../../causal_agent/families/adjustment/tests/test_motifs.py), [test_postviz.py](../../causal_agent/families/adjustment/tests/test_postviz.py), [test_selection.py](../../causal_agent/families/adjustment/tests/test_selection.py), [test_space.py](../../causal_agent/families/adjustment/tests/test_space.py) |

## 12. [difference in differences](12-difference-in-differences.md)

| Directory | Owned files |
|---|---|
| `causal_agent/families/diff_in_diff` | [__init__.py](../../causal_agent/families/diff_in_diff/__init__.py), [design.py](../../causal_agent/families/diff_in_diff/design.py), [family.yaml](../../causal_agent/families/diff_in_diff/family.yaml), [handoff.py](../../causal_agent/families/diff_in_diff/handoff.py), [postviz.py](../../causal_agent/families/diff_in_diff/postviz.py), [probes.py](../../causal_agent/families/diff_in_diff/probes.py) |
| `causal_agent/families/diff_in_diff/evals` | [__init__.py](../../causal_agent/families/diff_in_diff/evals/__init__.py), [cases.yaml](../../causal_agent/families/diff_in_diff/evals/cases.yaml), [evaluators.py](../../causal_agent/families/diff_in_diff/evals/evaluators.py), [summarise.py](../../causal_agent/families/diff_in_diff/evals/summarise.py) |
| `causal_agent/families/diff_in_diff/evals/handoffs` | [card_krueger_forced.json](../../causal_agent/families/diff_in_diff/evals/handoffs/card_krueger_forced.json), [cigar_forced.json](../../causal_agent/families/diff_in_diff/evals/handoffs/cigar_forced.json), [marketing_forced.json](../../causal_agent/families/diff_in_diff/evals/handoffs/marketing_forced.json) |
| `causal_agent/families/diff_in_diff/lane` | [__init__.py](../../causal_agent/families/diff_in_diff/lane/__init__.py), [artifacts.py](../../causal_agent/families/diff_in_diff/lane/artifacts.py), [contracts.py](../../causal_agent/families/diff_in_diff/lane/contracts.py), [execute.py](../../causal_agent/families/diff_in_diff/lane/execute.py), [graph.py](../../causal_agent/families/diff_in_diff/lane/graph.py), [history.py](../../causal_agent/families/diff_in_diff/lane/history.py), [nodes.py](../../causal_agent/families/diff_in_diff/lane/nodes.py), [shape.py](../../causal_agent/families/diff_in_diff/lane/shape.py), [state.py](../../causal_agent/families/diff_in_diff/lane/state.py), [targets.py](../../causal_agent/families/diff_in_diff/lane/targets.py) |
| `causal_agent/families/diff_in_diff/lane/engine` | [__init__.py](../../causal_agent/families/diff_in_diff/lane/engine/__init__.py), [diagnostics.py](../../causal_agent/families/diff_in_diff/lane/engine/diagnostics.py), [estimators.py](../../causal_agent/families/diff_in_diff/lane/engine/estimators.py), [falsify.py](../../causal_agent/families/diff_in_diff/lane/engine/falsify.py), [sensitivity.py](../../causal_agent/families/diff_in_diff/lane/engine/sensitivity.py) |
| `causal_agent/families/diff_in_diff/lane/knowledge` | [__init__.py](../../causal_agent/families/diff_in_diff/lane/knowledge/__init__.py), [checks.yaml](../../causal_agent/families/diff_in_diff/lane/knowledge/checks.yaml), [endpoints.yaml](../../causal_agent/families/diff_in_diff/lane/knowledge/endpoints.yaml), [estimators.yaml](../../causal_agent/families/diff_in_diff/lane/knowledge/estimators.yaml), [falsifications.yaml](../../causal_agent/families/diff_in_diff/lane/knowledge/falsifications.yaml), [rungs.yaml](../../causal_agent/families/diff_in_diff/lane/knowledge/rungs.yaml), [sensitivity.yaml](../../causal_agent/families/diff_in_diff/lane/knowledge/sensitivity.yaml), [space.yaml](../../causal_agent/families/diff_in_diff/lane/knowledge/space.yaml) |
| `causal_agent/families/diff_in_diff/lane/reason` | [__init__.py](../../causal_agent/families/diff_in_diff/lane/reason/__init__.py), [diagnostics.py](../../causal_agent/families/diff_in_diff/lane/reason/diagnostics.py), [ledger.py](../../causal_agent/families/diff_in_diff/lane/reason/ledger.py), [nodes.py](../../causal_agent/families/diff_in_diff/lane/reason/nodes.py), [rivals.py](../../causal_agent/families/diff_in_diff/lane/reason/rivals.py), [selection.py](../../causal_agent/families/diff_in_diff/lane/reason/selection.py), [views.py](../../causal_agent/families/diff_in_diff/lane/reason/views.py) |
| `causal_agent/families/diff_in_diff/tests` | [__init__.py](../../causal_agent/families/diff_in_diff/tests/__init__.py), [test_diagnostic_code.py](../../causal_agent/families/diff_in_diff/tests/test_diagnostic_code.py), [test_policy.py](../../causal_agent/families/diff_in_diff/tests/test_policy.py), [test_probes.py](../../causal_agent/families/diff_in_diff/tests/test_probes.py), [test_shape.py](../../causal_agent/families/diff_in_diff/tests/test_shape.py) |

## 13. [discontinuity](13-discontinuity.md)

| Directory | Owned files |
|---|---|
| `causal_agent/families/discontinuity` | [__init__.py](../../causal_agent/families/discontinuity/__init__.py), [design.py](../../causal_agent/families/discontinuity/design.py), [family.yaml](../../causal_agent/families/discontinuity/family.yaml), [handoff.py](../../causal_agent/families/discontinuity/handoff.py), [probes.py](../../causal_agent/families/discontinuity/probes.py) |
| `causal_agent/families/discontinuity/evals` | [__init__.py](../../causal_agent/families/discontinuity/evals/__init__.py), [cases.yaml](../../causal_agent/families/discontinuity/evals/cases.yaml), [evaluators.py](../../causal_agent/families/discontinuity/evals/evaluators.py), [summarise.py](../../causal_agent/families/discontinuity/evals/summarise.py) |
| `causal_agent/families/discontinuity/evals/handoffs` | [card_krueger_forced.json](../../causal_agent/families/discontinuity/evals/handoffs/card_krueger_forced.json), [senate_context.json](../../causal_agent/families/discontinuity/evals/handoffs/senate_context.json), [students_forced.json](../../causal_agent/families/discontinuity/evals/handoffs/students_forced.json) |
| `causal_agent/families/discontinuity/lane` | [__init__.py](../../causal_agent/families/discontinuity/lane/__init__.py), [artifacts.py](../../causal_agent/families/discontinuity/lane/artifacts.py), [contracts.py](../../causal_agent/families/discontinuity/lane/contracts.py), [execute.py](../../causal_agent/families/discontinuity/lane/execute.py), [graph.py](../../causal_agent/families/discontinuity/lane/graph.py), [nodes.py](../../causal_agent/families/discontinuity/lane/nodes.py), [state.py](../../causal_agent/families/discontinuity/lane/state.py) |
| `causal_agent/families/discontinuity/lane/engine` | [__init__.py](../../causal_agent/families/discontinuity/lane/engine/__init__.py), [diagnostics.py](../../causal_agent/families/discontinuity/lane/engine/diagnostics.py), [fit.py](../../causal_agent/families/discontinuity/lane/engine/fit.py), [postchecks.py](../../causal_agent/families/discontinuity/lane/engine/postchecks.py), [shape.py](../../causal_agent/families/discontinuity/lane/engine/shape.py) |
| `causal_agent/families/discontinuity/lane/knowledge` | [__init__.py](../../causal_agent/families/discontinuity/lane/knowledge/__init__.py), [checks.yaml](../../causal_agent/families/discontinuity/lane/knowledge/checks.yaml), [endpoints.yaml](../../causal_agent/families/discontinuity/lane/knowledge/endpoints.yaml), [estimators.yaml](../../causal_agent/families/discontinuity/lane/knowledge/estimators.yaml), [falsifications.yaml](../../causal_agent/families/discontinuity/lane/knowledge/falsifications.yaml), [inference.yaml](../../causal_agent/families/discontinuity/lane/knowledge/inference.yaml), [rungs.yaml](../../causal_agent/families/discontinuity/lane/knowledge/rungs.yaml), [sensitivity.yaml](../../causal_agent/families/discontinuity/lane/knowledge/sensitivity.yaml), [space.yaml](../../causal_agent/families/discontinuity/lane/knowledge/space.yaml) |
| `causal_agent/families/discontinuity/lane/reason` | [__init__.py](../../causal_agent/families/discontinuity/lane/reason/__init__.py), [diagnostics.py](../../causal_agent/families/discontinuity/lane/reason/diagnostics.py), [nodes.py](../../causal_agent/families/discontinuity/lane/reason/nodes.py), [prompts.py](../../causal_agent/families/discontinuity/lane/reason/prompts.py), [provenance.py](../../causal_agent/families/discontinuity/lane/reason/provenance.py), [selection.py](../../causal_agent/families/discontinuity/lane/reason/selection.py), [views.py](../../causal_agent/families/discontinuity/lane/reason/views.py) |
| `causal_agent/families/discontinuity/tests` | [__init__.py](../../causal_agent/families/discontinuity/tests/__init__.py), [test_diagnostics.py](../../causal_agent/families/discontinuity/tests/test_diagnostics.py), [test_lane.py](../../causal_agent/families/discontinuity/tests/test_lane.py), [test_local_random.py](../../causal_agent/families/discontinuity/tests/test_local_random.py) |

## 14. [exploration](14-exploration.md)

| Directory | Owned files |
|---|---|
| `causal_agent/explore` | [__init__.py](../../causal_agent/explore/__init__.py), [agent.py](../../causal_agent/explore/agent.py), [checks.yaml](../../causal_agent/explore/checks.yaml), [client.py](../../causal_agent/explore/client.py), [contracts.py](../../causal_agent/explore/contracts.py), [diagnostics.py](../../causal_agent/explore/diagnostics.py), [gates.py](../../causal_agent/explore/gates.py), [project.py](../../causal_agent/explore/project.py), [prompts.py](../../causal_agent/explore/prompts.py), [sandbox.py](../../causal_agent/explore/sandbox.py), [server.py](../../causal_agent/explore/server.py), [tools.py](../../causal_agent/explore/tools.py) |
| `causal_agent/explore/tests` | [__init__.py](../../causal_agent/explore/tests/__init__.py), [fakes.py](../../causal_agent/explore/tests/fakes.py), [test_agent.py](../../causal_agent/explore/tests/test_agent.py), [test_client.py](../../causal_agent/explore/tests/test_client.py), [test_diagnostics.py](../../causal_agent/explore/tests/test_diagnostics.py), [test_gates.py](../../causal_agent/explore/tests/test_gates.py), [test_project.py](../../causal_agent/explore/tests/test_project.py), [test_sandbox.py](../../causal_agent/explore/tests/test_sandbox.py), [test_server.py](../../causal_agent/explore/tests/test_server.py), [test_table.py](../../causal_agent/explore/tests/test_table.py), [test_tools.py](../../causal_agent/explore/tests/test_tools.py) |

## 15. [visualization](15-visualization.md)

| Directory | Owned files |
|---|---|
| `causal_agent/desk` | [material.py](../../causal_agent/desk/material.py), [pull.py](../../causal_agent/desk/pull.py) |
| `causal_agent/desk/tests` | [test_material.py](../../causal_agent/desk/tests/test_material.py), [test_pull.py](../../causal_agent/desk/tests/test_pull.py) |
| `causal_agent/viz` | [__init__.py](../../causal_agent/viz/__init__.py), [spec.py](../../causal_agent/viz/spec.py) |
| `causal_agent/viz/postviz` | [__init__.py](../../causal_agent/viz/postviz/__init__.py), [common.py](../../causal_agent/viz/postviz/common.py) |
| `causal_agent/viz/tests` | [__init__.py](../../causal_agent/viz/tests/__init__.py), [test_postviz.py](../../causal_agent/viz/tests/test_postviz.py), [test_spec.py](../../causal_agent/viz/tests/test_spec.py) |
| `web/src` | [figure.test.ts](../../web/src/figure.test.ts), [figure.ts](../../web/src/figure.ts), [fmt.test.ts](../../web/src/fmt.test.ts), [fmt.ts](../../web/src/fmt.ts), [table.test.ts](../../web/src/table.test.ts), [table.ts](../../web/src/table.ts) |
| `web/src/components` | [Figure.test.ts](../../web/src/components/Figure.test.ts), [Figure.tsx](../../web/src/components/Figure.tsx), [JsonView.tsx](../../web/src/components/JsonView.tsx) |
| `web/src/components/inspector` | ,  |
| `web/src/inspector` | [rows.test.ts](../../web/src/inspector/rows.test.ts), [rows.ts](../../web/src/inspector/rows.ts) |

## 16. [infrastructure](16-infrastructure.md)

| Directory | Owned files |
|---|---|
| `.` | [.env.example](../../.env.example), [.pre-commit-config.yaml](../../.pre-commit-config.yaml), [Makefile](../../Makefile), [langgraph.json](../../langgraph.json), [pyproject.toml](../../pyproject.toml), [uv.lock](../../uv.lock) |
| `.github/workflows` | [ci.yml](../../.github/workflows/ci.yml) |
| `causal_agent` | [__init__.py](../../causal_agent/__init__.py), [conftest.py](../../causal_agent/conftest.py) |
| `causal_agent/common` | [__init__.py](../../causal_agent/common/__init__.py), [addresses.py](../../causal_agent/common/addresses.py), [config.py](../../causal_agent/common/config.py), [decider.py](../../causal_agent/common/decider.py), [llm.py](../../causal_agent/common/llm.py), [numbers.py](../../causal_agent/common/numbers.py), [prompts.py](../../causal_agent/common/prompts.py), [trace.py](../../causal_agent/common/trace.py) |
| `causal_agent/common/contracts` | [__init__.py](../../causal_agent/common/contracts/__init__.py), [base.py](../../causal_agent/common/contracts/base.py) |
| `causal_agent/common/tests` | [__init__.py](../../causal_agent/common/tests/__init__.py), [test_decider.py](../../causal_agent/common/tests/test_decider.py), [test_design.py](../../causal_agent/common/tests/test_design.py), [test_endpoint_policy.py](../../causal_agent/common/tests/test_endpoint_policy.py), [test_llm.py](../../causal_agent/common/tests/test_llm.py), [test_trace_artifacts.py](../../causal_agent/common/tests/test_trace_artifacts.py), [test_trace_policy.py](../../causal_agent/common/tests/test_trace_policy.py) |
| `causal_agent/evals` | [__init__.py](../../causal_agent/evals/__init__.py), [dataset.py](../../causal_agent/evals/dataset.py), [docfigs.py](../../causal_agent/evals/docfigs.py), [evaluators.py](../../causal_agent/evals/evaluators.py), [explore.py](../../causal_agent/evals/explore.py), [families.py](../../causal_agent/evals/families.py), [ladder.py](../../causal_agent/evals/ladder.py), [lane.py](../../causal_agent/evals/lane.py), [run.py](../../causal_agent/evals/run.py), [spec.py](../../causal_agent/evals/spec.py), [trace.py](../../causal_agent/evals/trace.py), [turns.py](../../causal_agent/evals/turns.py) |
| `causal_agent/evals/tests` | [__init__.py](../../causal_agent/evals/tests/__init__.py), [test_specs.py](../../causal_agent/evals/tests/test_specs.py), [test_turns.py](../../causal_agent/evals/tests/test_turns.py) |
| `causal_agent/families/tests` | [__init__.py](../../causal_agent/families/tests/__init__.py), [test_core_names_no_family.py](../../causal_agent/families/tests/test_core_names_no_family.py), [test_knowledge.py](../../causal_agent/families/tests/test_knowledge.py), [test_ladders.py](../../causal_agent/families/tests/test_ladders.py), [test_registry.py](../../causal_agent/families/tests/test_registry.py) |
| `docker` | [explore.Dockerfile](../../docker/explore.Dockerfile) |
| `scripts` | [forced_handoffs.sh](../../scripts/forced_handoffs.sh) |
| `web` | [eslint.config.js](../../web/eslint.config.js), [index.html](../../web/index.html), [package-lock.json](../../web/package-lock.json), [package.json](../../web/package.json), [tsconfig.json](../../web/tsconfig.json), [vite.config.ts](../../web/vite.config.ts) |

Ingestion additions: [server/ingestion](../../causal_agent/server/ingestion/), [Alembic migrations](../../migrations/), [rendered upload tests](../../web/src/pages/NewDataset.test.tsx), and [storage tests](../../causal_agent/server/tests/test_ingestion.py) belong to component 02.

## Conversation publication and Compose launcher additions

The API/persistence owners include `server/conversations/{schema,service,publication,routes}.py`, `migrations/versions/0005_conversations.py` and `server/tests/test_conversations.py`. Exploration owns `explore/queue.py`, `docker/explore_worker.py`, `docker/isolation.py` and `docker/sandbox_probe.py`. API runtime owns `server/runtime.py`.

Phase verification tooling: `scripts/verify_phase4.py` orchestrates existing checks and owns `scripts/tests/test_verify_phase4.py`. The shared mutation runners remain under `scripts/`, browser/deployment verification under `docker/smoke.py`, and real context acceptance plus its evidence grader under `causal_agent/evals/context.py` and `causal_agent/evals/tests/test_context.py`. Product tests retain the owners listed above.

## Consolidated chat ownership

| Owner | Current entrypoints |
|---|---|
| Context contracts | [review](../../causal_agent/memory/context/review.py), [projection](../../causal_agent/memory/context/projection.py), [commands](../../causal_agent/memory/context/conversation.py), [snapshot reader](../../causal_agent/memory/context/snapshot.py) |
| Context persistence and review | [repository](../../causal_agent/server/context/repository.py), [review routes](../../causal_agent/server/context/reviews.py) |
| Conversation execution | [graph](../../causal_agent/desk/graph.py), [manager](../../causal_agent/server/conversations/manager.py), [runtime](../../causal_agent/server/conversations/runtime.py), [receipts](../../causal_agent/server/conversations/commands.py) |
| Artifact catalogue | [publication](../../causal_agent/server/conversations/publication.py), [artifact contracts](../../causal_agent/memory/artifacts.py) |
| Review and chat | [ContextReview](../../web/src/pages/ContextReview.tsx), [Chat](../../web/src/pages/Chat.tsx), [ContextStatements](../../web/src/components/ContextStatements.tsx), [ArtifactCard](../../web/src/components/ArtifactCard.tsx), [AnalysisCard](../../web/src/components/AnalysisCard.tsx) |
| Verification | [runner](../../scripts/verify_chat.py), [Gemini chat](../../causal_agent/evals/chat.py), [lane boundaries](../../causal_agent/evals/chat_lanes.py), [browser journey](../../web/e2e/chat.spec.ts) |

The shared context read projections are [memory/context/views.py](../../causal_agent/memory/context/views.py). They project the same canonical statements used by extraction, review and chat, including human changes and provenance.
