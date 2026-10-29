# 04. Conversation interface

[Component map](README.md) · [Source ownership](source-index.md)

The journey is dataset selection, question entry, grouped context review, explicit confirmation, then chat. [DatasetDetails](../../web/src/pages/DatasetDetails.tsx) offers explicit preparation when profiling or extraction is missing. Reads and refreshes do not start model work. [ContextReview](../../web/src/pages/ContextReview.tsx) displays dataset statements, columns and relationships with qualifications and original passages. Every column needs a reviewed or unresolved disposition. Confirming mappings does not confirm causal assumptions.

[Chat](../../web/src/pages/Chat.tsx) displays the saved question and warm opening, bounded messages, backend readiness, typed fast-track answers and a separate Run analysis action. Unknown is an explicit choice; no factual answer is preselected. A topic change pauses preparation. Pending questions and answers survive reloads.

Context, Artifacts and Analyses are inspectors. Opening one does not change conversational focus. Discuss actions explicitly select a column, artifact or registered element; Clear focus removes it. Earlier dataset artifacts require explicit reuse. [ContextStatements](../../web/src/components/ContextStatements.tsx) edits statements and relationship participants through the shared revision API. Stale edits require review and explicit refresh. [ArtifactCard](../../web/src/components/ArtifactCard.tsx) and [AnalysisCard](../../web/src/components/AnalysisCard.tsx) load evidence and details on demand. [CitedText](../../web/src/components/CitedText.tsx) exposes readable sources without displaying internal field addresses.

All wire types are generated from server models. Routine polling fetches status, prompt, readiness, focus, counts and references. Messages and inspector pages are separate bounded reads. The frontend does not decide admission, causal meaning or method selection.

The [browser journey](../../web/e2e/chat.spec.ts) covers entry, explicit unresolved review, welcome, inspector/focus separation, correction and reload. The Compose smoke runner replaces containers and checks persisted conversation state. Real semantic acceptance is separately recorded by `make verify-chat WITH_MODEL=1`.
