# 12. The trace mirrors the graph

Decided 2026-10-06. ADR 0006 and 0008 stand: the desk operates, tools reason, a judgement is a bounded episode. This one says
what the LangSmith trace must show, where the tracing lives so that it keeps showing it when the graph changes, and what
holds that.

## The problem it answers

Tracing was on, and the trace did not show the workflow. The lane ran in a subprocess with a fresh thread id and no parent,
so the ladder was a separate root trace and the desk's `run` node was a blank span that lasted minutes. Every turn was a root
run named "LangGraph". Every judgement went through `structured()`, which took a node name and never gave it to the trace,
so under each node the tree read "RunnableSequence, ChatVertexAI, parser", all alike. An episode's tool calls ran in plain
Python, so there were no tool spans, no refusals, no budget, and the gate's errors, the one thing worth reading when a rung
goes wrong, never reached LangSmith. The files a run left behind were invisible.

Each of these was fixable by hand, once. The next node, the next lane, the next writer would have lost it again.

## The decision

**The trace is the graph as it ran.** One trace per turn, named for the dataset, the analysis, the turn and the phase. The
lane's graph nests under the desk's `run` node, on the same thread. Every judgement is a span named for the node, with the
prompt in and the typed record, the thoughts and the token counts out. An episode is a span with one `try n` under it per
attempt, a `look` under the try with one tool span per call (the args in, the addressed fact or the refusal out), and a `gate`
with the errors. Every other refusal that re-prompts the model is a `gate` span too. Every file written under a design
folder, a run folder or a viz folder is an `artifact:<file>` span with the file attached. A journal step and a memory save
are tool spans.

**The tracing lives in chokepoints.** Not in the nodes. `common/trace.py` holds the helpers; the places that call them are
the ones every new piece passes through anyway:

| anything new that | goes through | and is traced for free |
|---|---|---|
| calls the model | `common/llm.py::structured`, `lane/episode.py::run_episode` | yes |
| reads data for a judgement | `lane/tools.py::Tools.call`, via the episode | yes |
| compiles a graph | `.compile(name=...)` in `desk/graph.py`, `lane/graph.py` | yes |
| starts a process | `trace.child_env` in the parent, `trace.parent()` in the child | yes |
| writes under a design, run or viz folder | `trace.write` | yes |
| refuses a model answer | `trace.gate` | yes |
| appends a step or saves the memory | `Journal.append`, `store.save` | yes |

**A span is opened around a call, never set as a config on a runnable.** The fakes the tests inject have a bare `invoke`
and no `with_config`; a config on the model object would break every one of them. A span around the call never touches the
model object, and every helper is a no-op when `LANGSMITH_TRACING` is off, so the suite runs as before and never sends a run.

**A test holds it.** `causal_agent/common/tests/test_trace_policy.py` greps the package the way
`families/tests/test_core_names_no_family.py` does: the model is invoked only in the two chokepoints; every `.compile(` has a
name; a process is started only where listed, and the lane's starter carries the parent; a file is written directly only in
the listed modules, each with its reason; a file that re-prompts the model calls `trace.gate`. A new writer or caller that
may not use the chokepoint is added to the allowlist with the reason beside it, and the review reads the reason.

## What it costs

A span per judgement, try, tool call, gate and file: a few hundred small runs per conversation, posted in the background.
An attachment per artifact: the files are small (JSON, Markdown, one PNG); a file over 15 MB is logged without its bytes.
Nothing is sent when tracing is off.

## How it is read

In LangSmith, the project's root runs are the turns; open one and the tree is the graph. From the terminal, `make trace`
prints the newest trace as a tree with each gate's errors inline, and `make trace ARGS="--thoughts"` adds the model's
thoughts under each judgement. The real-model run is the check: a change to the graph is done when its trace reads right.
