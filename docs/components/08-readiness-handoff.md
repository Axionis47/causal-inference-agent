# 08. Readiness and analysis handoff

[Component map](README.md) · [Source ownership](source-index.md)

Admission, preparation and authorization are separate. The existing endpoint catalogue and causal write gates decide whether requirements are satisfied. Reviewed column mappings do not imply causal assumptions. Canonical statements can bind catalogue fields; the in-process Memory is a disposable projection, never a second writable claims store.

[graph.assess](../../causal_agent/desk/graph.py) computes readiness without a model call once the question frame is saved. Corrections and answers reassess affected context. Fast-track presents unresolved requirements in prerequisite order. It cannot run analysis. When context is admitted, the existing [decision](../../causal_agent/desk/nodes/decide.py) and [designer](../../causal_agent/desk/designer.py) functions prepare one family and a concise review.

The separate Run analysis command names the prepared snapshot identity and selected revision. A context update invalidates preparation. [snapshot.py](../../causal_agent/memory/context/snapshot.py) writes the immutable handoff and a compact reference containing revision, context, family and integrity hash. [pipeline.py](../../causal_agent/desk/pipeline.py) passes this reference to a separate process. The shared reader resolves only those stored bytes. It cannot consult newer conversation context. The lane still validates dataset identity, family contract and endpoint admissions through [endpoints.py](../../causal_agent/lane/endpoints.py).

Reference-based runs retain trace parentage under the desk's `run` node. Freeze, outcome restrictions, family ledgers and numerical engines are unchanged. Unexpected essential context deficiencies terminate the attempt. A correction needs a new prepared snapshot and a new authorization; no interview or fitting loop starts automatically.

Owned analysis snapshots and imported records live under `data/conversations/<conversation>/analyses/<operation>/`. Assembled outputs are registered in the shared PostgreSQL/private-S3 artifact catalogue. Fixture evaluation tools may explicitly pass their existing Handoff contract directly to the pipeline; there is no old-conversation compatibility loader.

Verification includes [snapshot tests](../../causal_agent/memory/context/tests/test_snapshot.py), [endpoint policy](../../causal_agent/common/tests/test_endpoint_policy.py), [chat command tests](../../causal_agent/server/tests/context/test_chat_commands.py), and real-source [lane acceptance](../../causal_agent/evals/chat_lanes.py). A retained insufficient handoff must fail admission rather than gain invented assumptions.
