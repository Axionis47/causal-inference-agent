"""Snapshot seals and the real subprocess intake boundaries of all three lanes."""

import json
from pathlib import Path

import pytest

from causal_agent.common.contracts import Handoff
from causal_agent.desk import pipeline
from causal_agent.memory.context.snapshot import publish, resolve

ROOT = Path(__file__).resolve().parents[4]
CASES = [
    ("adjustment", "gov_transfers_forced.json"),
    ("diff_in_diff", "card_krueger_forced.json"),
    ("discontinuity", "senate_context.json"),
]


@pytest.mark.parametrize("family,filename", CASES)
def test_snapshot_is_exact_and_three_real_lanes_reject_changed_bytes(tmp_path, family, filename):
    handoff = Handoff.model_validate_json((ROOT / "causal_agent/families" / family / "evals/handoffs" / filename).read_text())
    # Do not invent study assumptions. An invalid dataset identity must stop at intake,
    # before any family reasoning or model call, in the actual isolated child process.
    handoff.dataset_identity = "0" * 64
    path = publish(tmp_path, handoff, "selected-revision")
    assert resolve(path).model_dump() == handoff.model_dump()
    record = pipeline.run(path, 1, handoff.pack_name, handoff.question, snapshot=True)
    assert record.status == "context_deficient", record
    assert record.specialist_result["context_deficiency"]["addresses"] == ["dataset.identity"]
    assert not record.specialist_result.get("estimates")


def test_seal_and_revision_cannot_be_replaced(tmp_path):
    family, filename = CASES[0]
    handoff = Handoff.model_validate_json((ROOT / "causal_agent/families" / family / "evals/handoffs" / filename).read_text())
    path = publish(tmp_path, handoff, "original-revision")
    with pytest.raises(ValueError, match="authorization cannot"):
        publish(tmp_path, handoff, "newer-revision")
    content = json.loads((tmp_path / "snapshot.json").read_text())
    content["question"] = "A silently substituted question"
    (tmp_path / "snapshot.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="integrity"):
        resolve(path)
