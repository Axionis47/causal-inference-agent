"""Ledger revisions preserve source accounts and all accepted history."""

from types import SimpleNamespace

from causal_agent.common.contracts.pack import Fact as ContextFact
from causal_agent.lane.episode import EpisodeLog, Fact
from causal_agent.lane.ledger import Entry, Evidence, Ledger, apply, seed
from causal_agent.lane.state import merge_episode_visits


def test_model_evidence_cannot_delete_a_confirmed_hidden_cause():
    reported = Evidence(source="person", address="user:turn:1", category="study_account")
    entry = Entry(
        id="rel:hidden", kind="hidden", src="hidden:income", value=True, status="confirmed", evidence=[reported], reason="person supplied hidden cause"
    )
    ledger, errors = apply(Ledger(), [entry], by="seed")
    assert not errors
    proposal = entry.model_copy(
        update={
            "value": False,
            "status": "supported",
            "evidence": [Evidence(source="test", address="check:graph", category="diagnostic")],
            "reason": "graph diagnostic",
        }
    )
    unchanged, errors = apply(ledger, [proposal], by="model")
    assert errors and unchanged == ledger and len(unchanged.history) == 1


def test_revision_retains_before_and_after_and_revisits_do_not_overwrite_facts():
    entry = Entry(id="rel:edge", kind="edge", src="a", dst="b", reason="plausible mechanism")
    ledger, errors = apply(Ledger(), [entry], by="relate")
    assert not errors
    amended = entry.model_copy(update={"status": "contested", "reason": "conflicting account"})
    revised, errors = apply(ledger, [amended], by="probe")
    assert not errors and revised.revision == 2 and revised.history[1].before == ledger.entries[entry.id]
    first, second = EpisodeLog(node="probe"), EpisodeLog(node="probe")
    first.facts.append(Fact(node="probe", visit=first.visit, n=1, tool="support", text="first"))
    second.facts.append(Fact(node="probe", visit=second.visit, n=1, tool="support", text="revisit"))
    visits = merge_episode_visits(merge_episode_visits({}, {"probe": first}), {"probe": second})
    assert len(visits) == 2 and first.facts[0].address != second.facts[0].address


def test_mixing_association_with_judgement_cannot_promote_causal_support():
    diagnostic = Evidence(source="data", address="probe:association", category="association")
    assumption = Evidence(source="judgement", address="pair.target", category="assumption")
    timing = Evidence(source="data", address="probe:clock", category="measurement_timing")
    for kind in (
        "edge",
        "hidden",
        "exclusion",
        "monotonicity",
        "continuity",
        "parallel_trends",
        "selection",
        "anticipation",
        "spillover",
        "other_shock",
        "exchangeability",
        "modifies",
    ):
        for evidence in ([diagnostic, assumption], [timing], [diagnostic, timing, assumption]):
            entry = Entry(id="rel:test", kind=kind, src="t", dst="y", status="supported", evidence=evidence, reason="candidate causal assumption")
            unchanged, errors = apply(Ledger(), [entry], by="model", resolves=lambda _: True)
            assert errors and not unchanged.entries and not unchanged.history
            proposed = entry.model_copy(update={"status": "proposed"})
            accepted, errors = apply(Ledger(), [proposed], by="model", resolves=lambda _: True)
            assert not errors and accepted.entries[entry.id].status == "proposed"
    source = Evidence(source="person", address="user:turn:1", category="study_account")
    supported = Entry(
        id="rel:test",
        kind="edge",
        src="t",
        dst="y",
        status="supported",
        evidence=[source, diagnostic, assumption],
        reason="the person supplies the causal mechanism",
    )
    accepted, errors = apply(Ledger(), [supported], by="model", resolves=lambda _: True)
    assert not errors and accepted.entries[supported.id].status == "supported"
    absence = supported.model_copy(
        update={"kind": "no_edge", "evidence": [timing], "reason": "a recorded later intervention cannot directly cause an earlier measurement"}
    )
    accepted, errors = apply(Ledger(), [absence], by="code", resolves=lambda _: True)
    assert not errors and accepted.entries[absence.id].status == "supported"
    confirmed = absence.model_copy(update={"status": "confirmed"})
    unchanged, errors = apply(Ledger(), [confirmed], by="code", resolves=lambda _: True)
    assert errors and not unchanged.entries


def test_seed_preserves_data_and_model_provenance_without_laundering_causal_support():
    facts = {
        f"claim:source_{name}.value": ContextFact(value=True, status="confirmed", source=source, said=f"original {name} evidence")
        for name, source in {
            "person": "user:turn:1",
            "document": "doc:protocol",
            "fixture": "fixture:protocol",
            "eval": "eval:protocol",
            "data": "data",
            "code": "code:row_count",
            "model": "model:infer",
            "unsourced": None,
        }.items()
    }
    handoff = SimpleNamespace(facts=facts, resolve=lambda address: address in facts)
    ledger = seed(handoff)
    for name in ("person", "document", "fixture", "eval"):
        entry = ledger.entries[f"rel:context.claim:source_{name}.value"]
        assert entry.status == "confirmed"
        assert entry.evidence[0].source == "pack" and entry.evidence[0].category == "study_account"
    for name in ("data", "code"):
        entry = ledger.entries[f"rel:context.claim:source_{name}.value"]
        assert entry.status == "confirmed" and entry.kind == "reported_field"
        assert entry.evidence[0].source == "data" and entry.evidence[0].category == "data_fact"
        assert entry.evidence[0].note == f"original {name} evidence"
        for status in ("supported", "confirmed"):
            causal = Entry(
                id="rel:direct", kind="edge", src="t", dst="y", status=status, evidence=entry.evidence, reason="a data fact alone cannot establish causation"
            )
            unchanged, errors = apply(ledger, [causal], by="model", resolves=handoff.resolve)
            assert errors and unchanged == ledger
    for name in ("model", "unsourced"):
        entry = ledger.entries[f"rel:context.claim:source_{name}.value"]
        assert entry.status == "proposed"
        assert entry.evidence[0].source == "judgement" and entry.evidence[0].category == "assumption"
    assert ledger.revision == len(facts) and len(ledger.history) == len(facts)
