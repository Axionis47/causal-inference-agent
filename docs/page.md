# Conversation page

The current page uses question-first review and consolidated chat. See the [conversation interface](components/04-conversation-ui.md), [API runtime](components/05-api-sessions.md), [desk](components/06-desk.md) and [snapshot boundary](components/08-readiness-handoff.md) for the owning code and contracts.

Select a saved dataset, enter a question, review mapped statements and explicitly mark all columns reviewed or unresolved. Confirming the review creates a chat pinned to that revision. Missing profiling or extraction has an explicit preparation action. Read operations never launch model work.

Chat supports free discussion, explicit context correction, visualization and fast-track preparation. Context, Artifacts and Analyses load readable details separately from bounded polling. Discuss selects a removable reference focus. Fast-track offers structured choices, unknown and free text without confirming a default. It prepares an analysis review; only the separate Run analysis action authorizes execution.

Corrections use the same PostgreSQL revision gate as direct editing. Other chats stay pinned until refreshed. A stale write returns changes for review. Repeated client submissions return their existing operation. Restarted work is shown as interrupted and resumes only explicitly, using recorded receipts and completed outputs.

Frontend contracts are generated from `server/models.py`. `make verify-chat` runs the relevant browser and container-replacement checks, with `WITH_MODEL=1` adding real conversations and trace readback.
