"""Supplied resources retain exact bytes and identity; there is no discovery or projection."""

import json

import pytest

from causal_agent.explore.contracts import ArtifactInput
from causal_agent.explore.inputs import digest, prepare
from causal_agent.explore.inspection import ExplorationTools
from causal_agent.explore.tests.test_agent import request
from causal_agent.lane.tools import Refused


def test_whole_dataset_and_explicit_result_are_staged_without_changes(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"estimate": 2.5, "interval": [1.0, 4.0], "target": "ATE"}))
    req = request(tmp_path)
    req.artifact_refs = [ArtifactInput(id="result:A", path=result, sha256=digest(result))]
    csv, root, manifest = prepare(req)
    assert csv.read_bytes() == req.dataset_version.path.read_bytes()
    assert (root / "artifact_1.json").read_bytes() == result.read_bytes()
    assert manifest["artifact_1.json"]["id"] == "result:A"
    assert not (root / "unselected.csv").exists()


def test_stale_checksum_fails_instead_of_silently_using_new_bytes(tmp_path):
    req = request(tmp_path)
    req.dataset_version.sha256 = "0" * 64
    with pytest.raises(ValueError, match="checksum changed"):
        prepare(req)


def test_unsupported_artifact_format_is_rejected(tmp_path):
    result = tmp_path / "result.pkl"
    result.write_bytes(b"never deserialize")
    req = request(tmp_path)
    req.artifact_refs = [ArtifactInput(id="result", path=result, sha256=digest(result))]
    with pytest.raises(ValueError, match="unsupported"):
        prepare(req)


def test_inspection_cannot_open_an_unsupplied_or_parent_path(tmp_path):
    req = request(tmp_path)
    csv, root, manifest = prepare(req)
    tool = ExplorationTools(csv=csv, inputs=root, manifest=manifest, out_dir=req.out_dir, timeout=60, stdout_tail=4000)
    with pytest.raises(ValueError, match="not a supplied"):
        tool.inspect("../request.json", output=False)
    with pytest.raises(ValueError):
        tool.inspect("step_1/../../outside.txt", output=True)


def test_duplicate_resource_identity_is_rejected(tmp_path):
    req = request(tmp_path)
    req.artifact_refs = [ArtifactInput(id=req.dataset_version.id, path=req.dataset_version.path, sha256=req.dataset_version.sha256)]
    with pytest.raises(ValueError, match="duplicate"):
        prepare(req)


def test_wrong_inspection_action_names_the_supplied_input_correction(tmp_path):
    req = request(tmp_path)
    csv, root, manifest = prepare(req)
    tool = ExplorationTools(csv=csv, inputs=root, manifest=manifest, out_dir=req.out_dir, timeout=60, stdout_tail=4000)
    with pytest.raises(Refused, match="action='inspect_inputs', reference='dataset.csv'"):
        tool.call("explore_action", {"action": "inspect_output", "reference": "dataset.csv"})
    result = tool.call("explore_action", {"action": "inspect_inputs", "reference": "dataset.csv"})
    assert "Preview only" in result.text and tool.inspections == 2 and not tool.steps
