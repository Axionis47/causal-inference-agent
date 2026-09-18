"""The analysis as a black box: dataset and question in, a RunRecord out. Tests replace `run`."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from causal_agent.chat.contracts import RunRecord


def _decision(state: dict) -> dict:
    d = state.get("decision")
    h = state.get("handoff")
    out: dict = {}
    if d is not None:
        dd = d.model_dump() if hasattr(d, "model_dump") else dict(d)
        out["chosen"] = dd.get("chosen")
        out["chosen_assumption"] = dd.get("chosen_assumption")
        out["why"] = dd.get("why_over_alternatives") or ""
        out["over"] = {r.get("family"): r.get("reason") for r in (dd.get("rejected") or dd.get("rejections") or []) if isinstance(r, dict)}
    if h is not None:
        hd = h.model_dump() if hasattr(h, "model_dump") else dict(h)
        out.setdefault("chosen", hd.get("family"))
        out.setdefault("chosen_assumption", hd.get("chosen_assumption"))
    return out


def run(dataset: str, question: str, index: int) -> RunRecord:
    """The router and the lane in their own process. The desk's process holds the interview and the chat only;
    a lane that misbehaves with the process (the discontinuity lane ended the parent silently on macOS after the
    profiler had run in it) cannot take the conversation down, and the record comes back as a file."""
    import os
    import subprocess
    import sys
    import tempfile

    from causal_agent.intake.datasets import ROOT

    out_path = Path(tempfile.mkdtemp(prefix="desk-")) / f"run-{index}.json"
    cmd = [sys.executable, "-m", "causal_agent.router.run", dataset, question, "--json-file", str(out_path)]
    env = dict(os.environ)
    env.setdefault("LANGSMITH_TAGS", "desk")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if not out_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return RunRecord(index=index, dataset=dataset, question=question, status="pipeline_error", decision_record=f"the analysis process exited {proc.returncode} without a record:\n{tail}")
    out = json.loads(out_path.read_text())
    return record(dataset, question, index, out)


def record(dataset: str, question: str, index: int, out: dict) -> RunRecord:
    """A RunRecord from the router's final state, whether live objects or the JSON the subprocess wrote."""
    sr = out.get("specialist_result") or {}
    h = out.get("handoff")
    family = (h.family if h is not None and hasattr(h, "family") else (h or {}).get("family")) if h is not None else None
    specialist = (h.specialist if h is not None and hasattr(h, "specialist") else (h or {}).get("specialist")) if h is not None else None
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
        index=index, dataset=dataset, question=question, family=family, specialist=specialist,
        status=sr.get("status") or ("no_handoff" if h is None else "unknown"), run_dir=run_dir,
        effect=est.get("value") if est else None, ci_low=est.get("ci_low") if est else None, ci_high=est.get("ci_high") if est else None,
        estimator=design.get("estimator") if isinstance(design.get("estimator"), str) else (est.get("method") if est else None),
        decision_record=out.get("decision_record") or "", decision=_decision(out), specialist_result=sr, artifacts=artifacts,
    )
