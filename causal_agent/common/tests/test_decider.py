"""The decision model's wire and its configuration, without loading a checkpoint."""

from __future__ import annotations

from causal_agent.common import config, decider


def test_parse_reads_laya_and_jev_confidences_alike():
    raw = {
        "answers": {"kind": {"choice": "run", "answer_confidence": 0.9, "probabilities": {"run": 0.9, "ask": 0.1}}, "about": {"noul": 0.2, "confidence": 0.7}}
    }
    out = decider.parse(raw, model="x")
    assert out.model == "x"
    assert out.answers["kind"].choice == "run" and out.answers["kind"].confidence == 0.9 and out.answers["kind"].probabilities["ask"] == 0.1
    assert out.answers["about"].noul == 0.2 and out.answers["about"].confidence == 0.7
    assert decider.parse({}, model="x").answers == {}


def test_a_fine_tune_stands_in_for_the_english_checkpoint():
    assert decider.laya_models(None) is None
    assert decider.laya_models("/ckpt/turns") == {"english": "/ckpt/turns"}


def test_the_checkpoint_is_read_relative_to_the_root(tmp_path):
    d = config.load({"DECIDER": "laya"}, root=tmp_path).decider
    assert (d.kind, d.checkpoint) == ("laya", None)
    d = config.load({"DECIDER_CHECKPOINT": ".artifacts/decider/turns"}, root=tmp_path).decider
    assert d.checkpoint == str(tmp_path / ".artifacts" / "decider" / "turns")
    d = config.load({"DECIDER_CHECKPOINT": "/elsewhere/turns"}, root=tmp_path).decider
    assert d.checkpoint == "/elsewhere/turns"
