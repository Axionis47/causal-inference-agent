"""The pack weighed by code: provenance to weight, beliefs to flags, and the yaml's decisions (stop, ask, soften, caveat)."""

from __future__ import annotations

import yaml

from causal_agent.common.contracts import Belief, CheckResult, ColumnBrief, Provenance, Said
from causal_agent.desk.handoff import forced
from causal_agent.lane import case as C
from causal_agent.memory import store

RULES = yaml.safe_load("""
beliefs:
  trend_continues:
    value_field: believed
    by_status:
      confirmed_true:  {level: pass}
      confirmed_false: {level: hard, caveat: "the person says the treated group would have moved differently apart from the change"}
      unknown:         {level: soft, caveat: "the person could not say whether the groups would have kept moving together"}
      empty:           {level: soft, caveat: "not asked", ask: {question: "Would {treatment} have kept moving with the others?", options: [yes, no]}}
    with_check:
      pre_trends:
        - {belief: confirmed_false, check: hard, then: stop, reason: "the test and the person agree: the groups were already moving apart"}
        - {belief: confirmed_true, check: hard, then: ask, once: true, question: "The groups were already moving apart before the change (p = {value}). You said: \\"{said}\\". Is there a reason?", options: [yes, no]}
        - {belief: confirmed_true, check: hard, asked: true, then: soften, caveat: "the person kept the belief after seeing the test: \\"{said}\\""}
        - {belief: confirmed_true, check: soft, then: caveat, caveat: "the person believes the paths would have stayed together"}
  spillover:
    value_field: possible
    by_status:
      confirmed_true: {level: soft, caveat: "treated units could reach the comparison units"}
  movable:
    from: assignment.movable
    by_status:
      confirmed_true: {level: soft, caveat: "a unit could move its score"}
    with_check:
      density:
        - {belief: confirmed_true, check: soft, then: harden, unless_fact: "col:{score}.set_by", caveat: "the score could be moved and bunches at the line"}
unknowns:
  "col:*.when": {level: soft, caveat: "when {column} was set is not known"}
  "claim:change.period_value": {level: hard, ask: {question: "Which period is the first after the change?"}}
contradictions:
  "col:*.when": {level: soft, caveat: "the file and the person disagree on when {column} was set"}
""")


def cigar():
    return forced("cigar", "q", "diff_in_diff", "sales", "state", ["state", "year", "sales", "price", "pop"], memory=store.migrate("cigar", write=False))


def brief(key: str, **fields) -> ColumnBrief:
    prov = fields.pop("provenance", {})
    return ColumnBrief(name=key, key=key, provenance={k: Provenance(**v) for k, v in prov.items()}, **fields)


# ------------------------------------------------------------------ weights


def test_provenance_to_weight():
    assert C.weigh_provenance("confirmed", "user:turn:3") == "fact" and C.weigh_provenance("confirmed", "doc:note") == "fact" and C.weigh_provenance("confirmed", "data") == "fact"
    assert C.weigh_provenance("drafted", "model:infer") == "draft" and C.weigh_provenance("refuted", "user:turn:2") == "contested" and C.weigh_provenance("contradiction", None) == "contested"
    assert C.weigh_provenance("unknown", "user:turn:2") == "open" and C.weigh_provenance("empty", None) == "open"
    b = brief("lunch", when="before", moved_by_change=False, provenance={"when": {"status": "confirmed", "source": "user:turn:4"}, "moved_by_change": {"status": "drafted", "source": "model:infer"}})
    assert C.settled(b, "when") == ("fact", "before") and C.settled(b, "moved_by_change") == ("draft", False) and C.settled(b, "measures_outcome") == ("open", None)
    b = brief("lunch", when="unknown", provenance={"when": {"status": "refuted", "source": "user:turn:4"}})
    assert C.settled(b, "when") == ("contested", None)


def test_belief_status_keys():
    assert C.belief_status(None) == "empty"
    assert C.belief_status(Belief(kind="x", value=True, status="confirmed")) == "confirmed_true"
    assert C.belief_status(Belief(kind="x", value=False, status="confirmed")) == "confirmed_false"
    assert C.belief_status(Belief(kind="x", value=True, status="drafted")) == "drafted"
    assert C.belief_status(Belief(kind="x", status="unknown")) == "unknown"
    assert C.belief_status(Belief(kind="x", value=True, status="contradiction")) == "contradiction"


# ------------------------------------------------------------------ weigh


def test_weigh_sorts_fields_and_makes_flags_from_beliefs_unknowns_and_contradictions():
    h = cigar()
    h.columns = [brief("sales", role="outcome"), brief("state", role="treatment"),
                 brief("price", when="after", moved_by_change=True, provenance={"when": {"status": "confirmed", "source": "user:turn:2", "said": "the tax moved it"}, "moved_by_change": {"status": "confirmed", "source": "user:turn:2"}}),
                 brief("pop", when="before", provenance={"when": {"status": "drafted", "source": "model:infer"}}),
                 brief("ndi")]
    h.beliefs = {"trend_continues": Belief(kind="trend_continues", value=False, status="confirmed", source="user:turn:5", said="the state was already falling"),
                 "spillover": Belief(kind="spillover", value=True, status="confirmed", source="user:turn:6")}
    h.unknowns = ["col:ndi.when", "claim:change.period_value"]
    h.contradictions = ["col:pop.when"]
    case = C.weigh(h, RULES)
    assert case.facts["col:price.when"] == "after" and case.facts["col:price.moved_by_change"] is True and case.facts["claim:trend_continues.believed"] is False
    assert case.drafts == {"col:pop.when": "before"} and "col:ndi.when" in case.open and "col:ndi.moved_by_change" in case.open
    assert case.beliefs == {"trend_continues": "confirmed_false", "spillover": "confirmed_true", "movable": "empty"}
    names = {f.name: f for f in case.flags}
    assert names["belief.trend_continues"].level == "hard" and "moved differently" in names["belief.trend_continues"].caveat
    assert names["belief.spillover"].level == "soft"
    assert names["unknown.col:ndi.when"].caveat == "when ndi was set is not known" and names["unknown.col:ndi.when"].ask is None
    assert names["unknown.claim:change.period_value"].level == "hard" and names["unknown.claim:change.period_value"].ask.address == "claim:change.period_value"
    assert names["contradiction.col:pop.when"].caveat == "the file and the person disagree on when pop was set"
    text = case.render()
    assert "SETTLED BY THE PACK" in text and "[col:price.when] after" in text and "OPEN" in text and "FLAGS" in text
    checks = C.as_checks(case)
    assert {c.address for c in checks} >= {"check:all.belief.trend_continues", "check:all.belief.spillover", "check:all.unknown.claim:change.period_value"}
    assert all(c.level in ("soft", "hard") for c in checks)


def test_a_dataset_field_read_as_a_belief():
    h = cigar()
    h.assignment = {"kind": "cutoff_rule", "movable": True}
    h.claims = {"assignment": {"kind": "assignment", "fields": h.assignment, "status": "confirmed", "source": "user:turn:3"}}
    case = C.weigh(h, RULES)
    assert case.beliefs["movable"] == "confirmed_true" and case.facts["claim:assignment.movable"] is True
    h.unknowns = ["claim:assignment.movable"]
    assert C.weigh(h, RULES).beliefs["movable"] == "unknown"


# ------------------------------------------------------------------ decide_by_code


def _checks(level: str, name: str = "pre_trends", value: float = 0.004) -> list[CheckResult]:
    return [CheckResult(contrast="c", name=name, level=level, value=value, detail="joint test on the leads")]


def test_trend_false_and_a_hard_pre_trends_check_stops_by_code():
    h = cigar()
    h.beliefs = {"trend_continues": Belief(kind="trend_continues", value=False, status="confirmed", said="it was already falling")}
    case = C.weigh(h, RULES)
    action, payload, checks = C.decide_by_code(case, _checks("hard"), RULES, h)
    assert action == "stop" and "agree" in payload["reason"] and payload["flag"] == "belief.trend_continues"
    assert any("[check:c.pre_trends] hard" in f for f in payload["facts"]) and any('said "it was already falling"' in f for f in payload["facts"])


def test_trend_true_and_a_hard_check_asks_once_then_softens():
    h = cigar()
    h.beliefs = {"trend_continues": Belief(kind="trend_continues", value=True, status="confirmed", said="they moved together")}
    case = C.weigh(h, RULES)
    action, ask, checks = C.decide_by_code(case, _checks("hard"), RULES, h)
    assert action == "ask" and ask.address == "claim:trend_continues.believed" and ask.options == ["yes", "no"] and ask.evidence == ["check:c.pre_trends"]
    assert "p = 0.004" in ask.question and 'You said: "they moved together"' in ask.question and ask.because.startswith("[check:c.pre_trends]")
    h.said.append(Said(turn=7, about="lane:claim:trend_continues.believed", text="yes, the tax was announced early"))
    h.beliefs["trend_continues"].said = "yes, the tax was announced early"
    case = C.weigh(h, RULES)
    action, payload, checks = C.decide_by_code(case, _checks("hard"), RULES, h)
    assert action == "proceed" and payload is None and checks[0].level == "soft" and "kept the belief" in checks[0].detail and "announced early" in checks[0].detail
    action, _, checks = C.decide_by_code(case, _checks("soft"), RULES, h)
    assert action == "proceed" and checks[0].level == "soft" and "stayed together" in checks[0].detail


def test_an_unasked_belief_with_an_ask_opens_it_once():
    h = cigar()
    case = C.weigh(h, RULES)
    action, ask, _ = C.decide_by_code(case, [], RULES, h)
    assert action == "ask" and ask.address == "claim:trend_continues.believed" and "state" in ask.question
    h.said.append(Said(turn=3, about="lane:claim:trend_continues.believed, lane:claim:spillover.possible", text="I don't know"))
    assert C.already_asked(h, "claim:trend_continues.believed")
    action, _, _ = C.decide_by_code(C.weigh(h, RULES), [], RULES, h)
    assert action == "proceed"


def test_harden_unless_a_fact_stands():
    from causal_agent.common.contracts import RdDesign

    h = cigar()
    h.design = RdDesign(score="pop", cutoff=1.0, treated_side="above")
    h.beliefs = {"trend_continues": Belief(kind="trend_continues", value=True, status="confirmed")}  # settled, so no ask stands in the way
    h.assignment = {"kind": "cutoff_rule", "movable": True}
    h.claims = {"assignment": {"kind": "assignment", "fields": h.assignment, "status": "confirmed"}}
    case = C.weigh(h, RULES)
    action, _, checks = C.decide_by_code(case, _checks("soft", "density", 0.04), RULES, h)
    assert action == "proceed" and checks[0].level == "hard" and "bunches" in checks[0].detail
    case.facts["col:pop.set_by"] = "the census"
    action, _, checks = C.decide_by_code(case, _checks("soft", "density", 0.04), RULES, h)
    assert checks[0].level == "soft"


def test_a_stop_level_flag_stops_before_any_ask():
    h = cigar()
    rules = {"beliefs": {"cutoff_only": {"value_field": "believed", "by_status": {"confirmed_false": {"level": "stop", "caveat": "something else switches at the cutoff"},
                                                                                     "empty": {"level": "soft", "ask": {"question": "q"}}}}}}
    h.beliefs = {"cutoff_only": Belief(kind="cutoff_only", value=False, status="confirmed")}
    case = C.weigh(h, rules)
    assert C.as_checks(case)[0].level == "hard"
    action, payload, _ = C.decide_by_code(case, [], rules, h)
    assert action == "stop" and payload["reason"] == "something else switches at the cutoff"
