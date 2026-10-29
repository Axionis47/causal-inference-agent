# 05. API and conversation runtime

[Component map](README.md) · [Source ownership](source-index.md)

[app.py](../../causal_agent/server/app.py) serves typed commands and bounded projections. [conversations/manager.py](../../causal_agent/server/conversations/manager.py) schedules one active turn per owned conversation, using PostgreSQL admission and a local worker pool. SQLite checkpoints hold only the operation reference. The application supplies persistence to the desk through [runtime.py](../../causal_agent/server/conversations/runtime.py), without importing HTTP code into the domain layers.

Conversation creation requires a completed review generation. The review contains the initial question, dataset version and exact context revision. Its ID is also the creation idempotency key. Replaying creation returns that conversation. Original dataset bytes are checksum verified when cached for analysis.

A command includes a client submission ID and selected revision. Replaying identical content returns the existing operation; reusing an ID with different content fails. Context corrections, structured question answers, focus, reuse, refresh, fast-track pause/start and analysis authorization are typed operations. Normal text is semantically interpreted and its accepted operations are persisted before side effects.

[commands.py](../../causal_agent/server/conversations/commands.py) owns transactional turn admission, messages, revision selection, pending questions and completion receipts. [context/repository.py](../../causal_agent/server/context/repository.py) locks the dataset version and rejects stale writes with changed statement identities. A second chat keeps its pin until explicitly refreshed. Context updates invalidate analysis preparation.

A restart exposes a running turn as interrupted and requires explicit Resume. Recovery reuses committed context receipts and published artifacts. Completed A2A task identities are retained beside attempts. A started external task with no completed result is visibly stopped instead of blindly submitted again. Analysis recovery imports a completed result without fitting again.

Every context, message, artifact, reuse and file request checks owner and dataset/conversation identity. Detailed evidence is read separately from polling. Hosted authentication and independent worker scheduling remain outside this deployment milestone.

Tests live in [context/test_chat_commands.py](../../causal_agent/server/tests/context/test_chat_commands.py), [test_conversations.py](../../causal_agent/server/tests/test_conversations.py) and the shared isolated PostgreSQL/private-storage harness. `make verify-chat` includes deliberate ownership, stale-revision, implicit-write and duplicate-submission defects.
