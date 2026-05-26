"""DecisionLog + record_decision + AnalysisDecision shape."""
from __future__ import annotations

import pytest

from src.analysis.substrate.events import (
    AnalysisDecision,
    DecisionLog,
    record_decision,
)


def test_record_decision_appends_and_returns_typed_entry():
    log = DecisionLog()
    decision = record_decision(
        log,
        agent="effect_estimator",
        decision_type="method_selected",
        choice="DoubleML",
        reason="overlap is poor for IPW",
    )

    assert len(log) == 1
    assert log.view()[0] is decision
    assert decision.choice == "DoubleML"


def test_alternatives_default_to_empty_list_not_none():
    # Notebook section iterates .alternatives; None would break it.
    log = DecisionLog()
    d = record_decision(
        log,
        agent="a",
        decision_type="method_selected",
        choice="IPW",
        reason="overlap is ok",
    )
    assert d.alternatives == []


def test_alternatives_carry_through():
    log = DecisionLog()
    alts = [{"option": "IPW", "reason": "poor overlap"}]
    d = record_decision(
        log,
        agent="a",
        decision_type="method_selected",
        choice="DoubleML",
        reason="DoubleML handles non-linearities",
        alternatives=alts,
    )
    assert d.alternatives == alts


def test_timestamp_auto_populated_when_omitted():
    d = AnalysisDecision(
        agent="a",
        decision_type="x",
        choice="y",
        reason="z",
    )
    assert d.timestamp  # non-empty
    assert "T" in d.timestamp  # ISO 8601


def test_timestamp_respected_when_caller_provides_it():
    # model_post_init only fills when empty; explicit value wins.
    d = AnalysisDecision(
        agent="a",
        decision_type="x",
        choice="y",
        reason="z",
        timestamp="2025-01-15T12:00:00+00:00",
    )
    assert d.timestamp == "2025-01-15T12:00:00+00:00"


def test_fifo_trim_keeps_last_n_when_over_cap():
    log = DecisionLog(max_decisions=3)
    for i in range(5):
        record_decision(
            log,
            agent="a",
            decision_type="x",
            choice=f"c{i}",
            reason="r",
        )

    assert len(log) == 3
    assert [d.choice for d in log.view()] == ["c2", "c3", "c4"]


def test_view_returns_a_copy():
    log = DecisionLog()
    record_decision(log, agent="a", decision_type="x", choice="y", reason="z")
    snapshot = log.view()
    snapshot.clear()
    assert len(log) == 1


def test_max_decisions_below_one_rejected():
    with pytest.raises(ValueError, match=">= 1"):
        DecisionLog(max_decisions=0)


def test_max_decisions_property_exposed():
    log = DecisionLog(max_decisions=50)
    assert log.max_decisions == 50


def test_decision_extra_fields_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnalysisDecision(
            agent="a",
            decision_type="x",
            choice="y",
            reason="z",
            confidence=0.9,  # type: ignore[call-arg]
        )


def test_persistence_shape_round_trips():
    # storage/firestore.py and storage/local_storage.py serialise
    # decisions via model_dump. The round-trip must preserve every field.
    d = record_decision(
        DecisionLog(),
        agent="dag_expert",
        decision_type="dag_edge_added",
        choice="age -> outcome",
        reason="confounder per domain knowledge",
        alternatives=[{"option": "skip", "reason": "no domain support"}],
    )
    dumped = d.model_dump()
    revived = AnalysisDecision.model_validate(dumped)
    assert revived == d
