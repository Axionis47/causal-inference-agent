# Reading about it

Start with the [component handbook](components/README.md) for a functional map and an ordered component-by-component walkthrough. Use [architecture.md](architecture.md) for the architectural overview, then the generated endpoint ladders for exact supported trajectories.

| page | what it answers |
|---|---|
| [components/](components/README.md) | sixteen functional components: responsibilities, code ownership, contracts, decisions, dependencies, storage, walkthroughs and verification evidence |
| [production/](production/README.md) | consolidated plan: current container baseline, proposed private workspaces, frontend/backend organization, data model, lineage and implementation sequence |
| [production/hierarchy.md](production/hierarchy.md) | user/dataset/chat ownership hierarchy; exact current storage pointers; messages, journals, knowledge revisions and analysis ledgers |
| [production/tracing.md](production/tracing.md) | how LangSmith receives spans and attachments, links to application records, and independent retention/deletion |
| [production/containerization.md](production/containerization.md) | package the current app as one image, preserve data in volumes, run checks and understand sandbox/deployment limits |
| [production/implementation-plan.md](production/implementation-plan.md) | complete implementation phases, component responsibilities, cross-layer contracts and completion criteria |
| [production/verification.md](production/verification.md) | component-owned tests, real-storage and browser journeys, real-model acceptance and deployment release gates |
| [acceptance/context-lane-policy.md](acceptance/context-lane-policy.md) | real-data validation policy, retained empirical runs and coverage limits |
| [architecture.md](architecture.md) | how the code is laid out: the components, the layers, a family package, the stores, the wire, the checks |
| [desk.md](desk.md) | the desk graph: the question first, one question per turn, the run, the chat after; the three interrupts; the routing as code |
| [memory-and-matrix.md](memory-and-matrix.md) | a field with a status and a source; the one write path; the matrix that turns claims into a design choice |
| [gates.md](gates.md) | every model call in the system, what it is given, what it returns, what code checks, what happens on failure |
| [pack-and-addresses.md](pack-and-addresses.md) | the only input a lane sees, and the address grammar every cite is checked against |
| [lanes.md](lanes.md) | the three endpoint catalogues and shared lane boundaries |
| [ladders/](ladders/) | one table per lane, written by `make ladder` from its declarations: the rungs, the space, and the end states it can reach |
| [exploration.md](exploration.md) | how a look at the data is asked for, made by scripts in a sandbox one step at a time in a separate agent, gated, stored, and cited |
| [testing.md](testing.md) | testing policy, current trust limits, component and phase organization, scripted tests and evaluations |
| [page.md](page.md) | the routes, the one view, the generated wire, the inspector |
| [design.md](design.md) | what a turn, an artifact, a cited address, a question and the handoff look like on the page, and what the page never does |
| [diagrams/](diagrams/README.md) | the pictures and the visual language they share |
| [archive/adr/](archive/adr/) | past decision records; they describe earlier shapes and do not bind a change |

Phase 2 implementation and cleanup evidence: [Return to your datasets](audits/phase2-library.md).

Phase 3: [read-only dataset/column profiling](components/03-profiling.md), [verification evidence](audits/phase3-profiling.md).

Phase 4: [organized dataset and column context](components/phase4-context.md), [verification evidence](audits/phase4-context.md).

Consolidated chat: [review and conversation UI](components/04-conversation-ui.md), [context and artifact persistence](components/09-persistence.md), [verification and real-model readback](acceptance/consolidated-chat.md).
