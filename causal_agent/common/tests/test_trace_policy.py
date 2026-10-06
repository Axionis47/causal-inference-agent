"""The trace mirrors the graph: tracing lives in chokepoints, and nothing goes round them. A new model call, graph, process
boundary, artifact writer or gate either passes through the chokepoint, and is traced for free, or is listed here with the
reason it may not. See docs/adr/0012-the-trace-mirrors-the-graph.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# the only places that invoke the model
MODEL_CALLERS = {"common/llm.py", "lane/episode.py"}

# the only places that start a process; the first must carry the trace across
PROCESS_STARTERS = {
    "desk/pipeline.py": "the lane's process; carries the parent run and the thread",
    "viz/sandbox.py": "the drawing script, fenced; its span is opened around the call",
    "common/llm.py": "gcloud config, to find the project; no trace",
}

# the files that may write a file directly; everything else writes through trace.write so the file is attached
RAW_WRITERS = {
    "common/trace.py": "the chokepoint itself",
    "memory/store.py": "the memory's own yaml files; the save is one tool span",
    "memory/journal.py": "the journal; each step is one tool span",
    "server/transcript.py": "the page transcript, not an analysis artifact",
    "server/datasets.py": "the dataset index, meta and the uploaded CSV, not an analysis artifact",
    "profile/data.py": "the profile cache",
    "profile/profiler.py": "a command-line profile",
    "desk/handoff.py": "a command-line hand-off, outside a graph run",
    "viz/sandbox.py": "the seatbelt profile the fence runs under; code.py goes through trace.write",
    "evals/docfigs.py": "the docs' diagrams, outside the layers",
}

REPROMPT = re.compile(r"WAS REJECTED|FAILED THESE CHECKS|PREVIOUS ATTEMPT FAILED|_?rejected\(errors\)")
RAW_WRITE = re.compile(r"\.write_text\(|\.write_bytes\(|\.open\(\s*\"[aw]|open\([^)]*\"[aw]\"")


def _sources():
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        yield str(path.relative_to(ROOT)), path.read_text()


def test_only_the_chokepoints_invoke_the_model():
    hits = [rel for rel, src in _sources() if (".with_structured_output(" in src or ".bind_tools(" in src) and rel not in MODEL_CALLERS]
    assert not hits, f"these call the model outside structured() or run_episode(), so their judgements have no span: {hits}"


def test_every_compiled_graph_is_named():
    hits = []
    for rel, src in _sources():
        for n, line in enumerate(src.splitlines(), 1):
            if re.search(r"\.compile\(", line) and "re.compile(" not in line and "name=" not in line:
                hits.append(f"{rel}:{n}: {line.strip()}")
    assert not hits, "a graph compiled without a name is a root called 'LangGraph':\n" + "\n".join(hits)


def test_a_process_boundary_carries_the_trace():
    hits = [rel for rel, src in _sources() if re.search(r"subprocess\.run\(|Popen\(", src) and rel not in PROCESS_STARTERS]
    assert not hits, f"a new process would start a new trace root: {hits}"
    pipeline = (ROOT / "desk" / "pipeline.py").read_text()
    assert "trace.child_env(" in pipeline and "trace.parent()" in pipeline, "the lane's process must continue the desk's run"


def test_a_file_under_a_design_run_or_viz_folder_is_written_through_trace_write():
    hits = []
    for rel, src in _sources():
        if rel in RAW_WRITERS:
            continue
        for n, line in enumerate(src.splitlines(), 1):
            if RAW_WRITE.search(line):
                hits.append(f"{rel}:{n}: {line.strip()}")
    assert not hits, "written without trace.write, so the trace never shows the file:\n" + "\n".join(hits)


def test_a_re_prompt_has_a_gate_span():
    hits = []
    for rel, src in _sources():
        if "def rejected(" in src:  # the helper that renders the re-prompt text is not a gate
            continue
        if REPROMPT.search(src) and "trace.gate(" not in src:
            hits.append(rel)
    assert not hits, f"these re-prompt the model without a gate span saying why: {hits}"
