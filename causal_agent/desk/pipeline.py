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
from causal_agent.common.contracts import FamilyDecision, Handoff, RunRecord


def run_lane(h: Handoff, question: str) -> dict:
    """The specialist subgraph for the pack's family, in this process."""
    from causal_agent.families import registry as R

    sub = R.lanes().get(h.family)
    if sub is None:
        return {"status": "not_supported", "family": h.family, "message": f"no specialist for {h.family}"}
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": [f"dataset:{h.pack_name}", f"lane:{h.specialist}", "desk"],
        "metadata": {"dataset": h.pack_name},
    }
    out = sub.invoke({"question": question, "handoff": h, "dataset": h.pack_name}, cfg)
    return out.get("specialist_result") or {"status": "unknown"}


def run(handoff_path: str | Path, index: int, dataset: str, question: str, decision: dict | None = None, decision_record: str = "") -> RunRecord:
    """The lane in a subprocess on a hand-off file; a RunRecord either way."""
    from causal_agent.profile.datasets import ROOT

    handoff_path = Path(handoff_path)
    out_path = handoff_path.parent / "result.json"
    cmd = [sys.executable, "-m", "causal_agent.desk.pipeline", str(handoff_path), "--json-file", str(out_path)]
    env = dict(os.environ)
    env.setdefault("LANGSMITH_TAGS", "desk")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if not out_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return RunRecord(
            index=index,
            dataset=dataset,
            question=question,
            status="pipeline_error",
            design_dir=str(handoff_path.parent),
            decision=decision or {},
            decision_record=f"the analysis process exited {proc.returncode} without a record:\n{tail}",
        )
    sr = json.loads(out_path.read_text())
    h = Handoff.model_validate(json.loads(handoff_path.read_text()))
    return record(dataset, question, index, h, sr, decision or {}, decision_record, design_dir=str(handoff_path.parent))


def record(
    dataset: str, question: str, index: int, h: Handoff | None, sr: dict, decision: dict, decision_record: str, design_dir: str | None = None
) -> RunRecord:
    """A RunRecord from a lane's result, whether live or read back from the file it wrote."""
    design = sr.get("design") or {}
    est = next((e for e in sr.get("estimates") or [] if not e.get("secondary") and e.get("error") is None), None)
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
        estimator=design.get("estimator") if isinstance(design.get("estimator"), str) else (est.get("method") if est else None),
        decision_record=decision_record,
        decision=decision,
        specialist_result=sr,
        artifacts=artifacts,
    )


# ------------------------------------------------------------------ the record on disk


def decision_dict(d: FamilyDecision | None, h: Handoff | None) -> dict:
    """The decision as a RunRecord carries it: chosen, chosen_assumption, why, over."""
    out: dict = {}
    if d is not None:
        out = {"chosen": d.chosen, "chosen_assumption": d.chosen_assumption, "why": d.why_over_alternatives, "over": {r.family: r.reason for r in d.rejected}}
    if h is not None:
        out.setdefault("chosen", h.family)
        out.setdefault("chosen_assumption", h.chosen_assumption)
    return out


def save_record(rec: RunRecord) -> Path | None:
    """The record beside the design it came from, so a later conversation can list the run."""
    if not rec.design_dir:
        return None
    p = Path(rec.design_dir) / "record.json"
    p.write_text(rec.model_dump_json(indent=2))
    return p


def load_record(design_dir: str | Path) -> RunRecord | None:
    """The record a design dir holds: record.json when the run wrote one, else rebuilt from the lane's result file; None for a
    design that never ran."""
    d = Path(design_dir)
    if (d / "record.json").exists():
        return RunRecord.model_validate_json((d / "record.json").read_text())
    if not (d / "result.json").exists():
        return None
    sr = json.loads((d / "result.json").read_text())
    h = Handoff.model_validate(json.loads((d / "handoff.json").read_text())) if (d / "handoff.json").exists() else None
    dec = FamilyDecision.model_validate_json((d / "decision.json").read_text()) if (d / "decision.json").exists() else None
    note = (d / "record.md").read_text() if (d / "record.md").exists() else ""
    index = int(d.name) if d.name.isdigit() else 0
    dataset = h.pack_name if h else d.parents[1].name
    rec = record(dataset, h.question if h else "", index, h, sr, decision_dict(dec, h), note, design_dir=str(d))
    if (d / "figures.json").exists():
        try:
            rec.figures = [f for f in json.loads((d / "figures.json").read_text()) if isinstance(f, dict)]
        except ValueError:
            rec.figures = []
    return rec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("handoff")
    ap.add_argument("--json-file", required=True)
    args = ap.parse_args(argv)
    raw = json.loads(Path(args.handoff).read_text())
    h = Handoff.model_validate(raw)
    sr = run_lane(h, h.question)
    Path(args.json_file).write_text(json.dumps(sr, default=str))


if __name__ == "__main__":
    main()
