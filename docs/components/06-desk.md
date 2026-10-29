# 06. Conversation desk

[Component map](README.md) · [Source ownership](source-index.md)

The desk discusses one dataset and the question collected before chat. [graph.py](../../causal_agent/desk/graph.py) uses one semantic interpretation path before and after analysis. It can retain discussion, explicit context corrections and visualization requests in the same message. Exact current-message passages authorize proposed operations. Unknown or conflicting references trigger targeted clarification before action.

The graph receives a repository-backed runtime, checkpoints only `operation_id`, and never saves a knowledge mirror. Domain edits and catalogue gates live in `memory/context` and `memory/ops.py`. The runtime derives a working Memory from the selected PostgreSQL revision, keeping original source provenance. Direct edits and conversational corrections reach the same transaction. Ordinary discussion and hypotheticals do not write.

Explicit message references take precedence over persistent focus. Focus is set by Discuss and cleared explicitly; opening an inspector does not change it. Artifact discussion reads registered evidence and files, with the existing citation and numerical checks. Incompatible causal definitions cannot support numerical comparison.

Fast-track uses the same endpoint obligations as ordinary preparation. The saved question frame, admission projection and pending question identities survive restarts. Questions follow prerequisites, have no default factual choice, and include unknown and free text. Unknown answers stay recorded and are not repeatedly requested. Topic changes pause preparation. Run analysis is a separate action against an admitted immutable snapshot.

Exploration remains an A2A call to the existing Docker-only service. Context updates, references and admission remain application operations. An essential lane deficiency terminates that attempt visibly. It never automatically starts another interview or analysis.

[route.py](../../causal_agent/desk/route.py) is a fixture-evaluation entrypoint using a disposable memory copy. It reuses frame, fit, decision, designer and handoff functions without writing durable conversation knowledge. The old interrupt-driven graph and prose choice parser have been removed.

See [readiness](08-readiness-handoff.md), [API runtime](05-api-sessions.md) and the [desk README](../../causal_agent/desk/README.md).
