"""The lane in its own process: a hand-off file in, a result file out.

    uv run python -m causal_agent.desk.pipeline designs/1/handoff.json --json-file out.json

The desk's process holds the conversation only; a lane that misbehaves with the process (the discontinuity lane ended
the parent silently on macOS after the profiler had run in it) cannot take the conversation down, and the record comes
back as a file. The lane reads the Handoff and the CSV, nothing else."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
from causal_agent.common import trace
from causal_agent.common.contracts import FamilyDecision, Handoff, RunRecord


def run_lane(h: Handoff, question: str, *, thread_id: str | None = None) -> dict:
    """The specialist subgraph for the pack's family, in this process, as one named root (or child, under a parent handed in)."""
    from causal_agent.families import registry as R

    sub = R.lanes().get(h.family)
    if sub is None:
        return {"status": "not_supported", "family": h.family, "message": f"no specialist for {h.family}"}
    cfg = {
        "run_name": f"lane:{h.family}",
        "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
        "tags": [f"dataset:{h.pack_name}", f"lane:{h.specialist}", f"design:{h.design_id}", "desk"],
        "metadata": {"dataset": h.pack_name, "family": h.family, "specialist": h.specialist, "design": h.design_id},
    }
    out = sub.invoke({"question": question, "handoff": h, "dataset": h.pack_name}, cfg)
    return out.get("specialist_result") or {"status": "unknown"}


def run(
    handoff_path: str | Path,
    index: int,
    dataset: str,
    question: str,
    decision: dict | None = None,
    decision_record: str = "",
    thread_id: str | None = None,
    snapshot: bool = False,
) -> RunRecord:
    """The lane in a subprocess on a hand-off file; a RunRecord either way. The child continues the caller's trace and thread."""
    from causal_agent.profile.datasets import ROOT

    handoff_path = Path(handoff_path)
    from causal_agent.memory.context.snapshot import resolve

    h = resolve(handoff_path) if snapshot else Handoff.model_validate_json(handoff_path.read_text())
    out_path = handoff_path.parent / "result.json"
    cmd = [sys.executable, "-m", "causal_agent.desk.pipeline", str(handoff_path), "--json-file", str(out_path)]
    if snapshot:
        cmd.append("--snapshot")
    env = trace.child_env(dict(os.environ), thread_id)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if not out_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        reason = f"the analysis process exited {proc.returncode} without a record"
        sr = {
            "status": "pipeline_error",
            "context_id": h.context_id,
            "dataset_identity": h.dataset_identity,
            "context_admission": h.admission.model_dump() if h.admission else None,
            "context_facts": {a: f.model_dump() for a, f in h.facts.items()},
            "designs": {},
            "lead_design_id": None,
            "lead_selection": {},
            "terminal": {"kind": "execution_failure", "status": "pipeline_error"},
            "execution_attempts": [{"stage": "lane_process", "exit_code": proc.returncode, "error": reason}],
            "feasibility": {
                "terminal_kind": "execution_failure",
                "stage": "lane_process",
                "reason": reason,
                "facts": [tail],
                "what_would_fix": "resolve the recorded process failure, then authorize a new run",
            },
        }
        trace.write(out_path, json.dumps(sr, indent=2))
        return record(dataset, question, index, h, sr, decision or {}, f"{reason}:\n{tail}", design_dir=str(handoff_path.parent))
    sr = json.loads(out_path.read_text())
    return record(dataset, question, index, h, sr, decision or {}, decision_record, design_dir=str(handoff_path.parent))


def record(
    dataset: str, question: str, index: int, h: Handoff | None, sr: dict, decision: dict, decision_record: str, design_dir: str | None = None
) -> RunRecord:
    """A RunRecord from a lane's result, whether live or read back from the file it wrote."""
    lead = sr.get("lead_design_id")
    est = next(
        (e for e in sr.get("estimates") or [] if e.get("design_id") == lead and not e.get("secondary") and not e.get("modifier") and e.get("error") is None),
        None,
    )
    run_dir = sr.get("run_dir")
    artifacts = {}
    if run_dir and (Path(run_dir) / "artifacts.json").exists():
        try:
            artifacts = json.loads((Path(run_dir) / "artifacts.json").read_text())
        except Exception:
            artifacts = {}
    return RunRecord(
        index=index,
        dataset=dataset,
        question=question,
        family=h.family if h else None,
        specialist=h.specialist if h else None,
        status=sr.get("status") or ("no_handoff" if h is None else "unknown"),
        run_dir=run_dir,
        design_dir=design_dir,
        effect=est.get("value") if est else None,
        ci_low=est.get("ci_low") if est else None,
        ci_high=est.get("ci_high") if est else None,
        estimator=est.get("method") if est else None,
        lead_design_id=lead,
        decision_record=decision_record,
        decision=decision,
        specialist_result=sr,
        artifacts=artifacts,
    )


# ------------------------------------------------------------------ the record on disk


def decision_dict(d: FamilyDecision | None, h: Handoff | None) -> dict:
    """The decision as a RunRecord carries it: chosen, chosen_assumption, why, over, and the design brief when there is one."""
    out: dict = {}
    if d is not None:
        out = {"chosen": d.chosen, "chosen_assumption": d.chosen_assumption, "why": d.why_over_alternatives, "over": {r.family: r.reason for r in d.rejected}}
    if h is not None:
        out.setdefault("chosen", h.family)
        out["chosen_assumption"] = h.chosen_assumption
        if h.brief is not None:
            out["brief"] = h.brief.model_dump()
    return out


def save_record(rec: RunRecord) -> Path | None:
    """The record beside the design it came from, so a later conversation can list the run."""
    if not rec.design_dir:
        return None
    return trace.write(Path(rec.design_dir) / "record.json", rec.model_dump_json(indent=2))


def load_record(design_dir: str | Path) -> RunRecord | None:
    """The record a design dir holds, or None for a design that never ran."""
    p = Path(design_dir) / "record.json"
    return RunRecord.model_validate_json(p.read_text()) if p.exists() else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("handoff")
    ap.add_argument("--json-file", required=True)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args(argv)
    from causal_agent.memory.context.snapshot import resolve

    h = resolve(Path(args.handoff)) if args.snapshot else Handoff.model_validate_json(Path(args.handoff).read_text())
    with trace.parent():
        sr = run_lane(h, h.question, thread_id=trace.thread_id())
        trace.write(args.json_file, json.dumps(sr, default=str))


if __name__ == "__main__":
    main()
