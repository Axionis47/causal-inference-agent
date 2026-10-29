# Conversation desk

The application collects a question and completes a context review before creating a chat. `graph.py` interprets every free-text turn through the same path before and after analysis. Its checkpoint contains only the accepted operation ID. PostgreSQL owns the selected context revision, messages, questions, accepted interpretation and operation receipts.

`memory/context` defines knowledge, explicit edits, requirement bindings and immutable snapshot references below the server layer. The application supplies the graph's runtime adapter through `server/conversations/runtime.py`. A `Memory` is an ephemeral projection of one selected revision. The desk does not save filesystem knowledge.

An interpretation can retain discussion, a correction and exploration together. Corrections and tool requests require exact passages from the current message. Unresolved references produce clarification before action. Ordinary discussion and hypotheticals never commit context. Human edits and structured answers use the same context revision gate. Explicit references override removable conversation focus.

Fast-track uses the existing endpoint admissions. It asks one prerequisite at a time within the three-question bound, stores stable question/option identities and offers an explicit unknown answer. It never authorizes a run. An admitted analysis review requires a separate typed authorization command. `memory/context/snapshot.py` seals the exact handoff and revision; `pipeline.py` launches the existing family in a subprocess using only that reference and its dataset. Essential deficiencies stop the attempt visibly.

Exploration remains A2A and generated scripts remain Docker-only. Exploration and assembled causal outputs share the owned publication catalogue. Discussion loads actual registered evidence and retains citation/numerical gates. A failed or interrupted external task is never silently resubmitted.

`route.py` supports fixture evaluations using an ephemeral copy of fixture memory and the same fit/decision/design functions. It is not a conversation writer. Family reasoning, ledgers, frozen execution and estimation are unchanged.

Verification: `make verify-chat`, with `WITH_MODEL=1` for separately reported Gemini and lane evidence. See [the component chapter](../../docs/components/06-desk.md) and [testing](../../docs/testing.md).
