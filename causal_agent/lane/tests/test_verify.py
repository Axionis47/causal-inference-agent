"""A model answer against the pack: a contradiction of a settled fact needs a cite to a contested address."""

from __future__ import annotations

from causal_agent.common.contracts import ColumnBrief, Provenance
from causal_agent.desk.handoff import forced
from causal_agent.lane import case as C
from causal_agent.lane import verify as V
from causal_agent.memory import store

RULES: list[V.Rule] = [
    ("affected_by_treatment", True, "when", ("before",)),
    ("usable_as_control", True, "moved_by_change", (True,)),
]


def pack():
    h = forced("cigar", "q", "diff_in_diff", "sales", "state", ["state", "year", "sales", "pop"], memory=store.migrate("cigar", write=False))
    h.columns = [
        ColumnBrief(
            name="pop",
            key="pop",
            when="before",
            moved_by_change=True,
            provenance={"when": Provenance(status="confirmed", source="user:turn:2"), "moved_by_change": Provenance(status="confirmed", source="user:turn:2")},
        )
    ]
    return h


def test_a_contradicting_claim_is_rejected_without_a_cite_and_accepted_with_one():
    h = pack()
    case = C.weigh(h, {})
    errs = V.contradictions({"affected_by_treatment": True, "usable_as_control": False}, "pop", case, RULES, ["col:pop.note"], h)
    assert len(errs) == 1 and "[col:pop.when] = 'before'" in errs[0]
    assert V.contradictions({"affected_by_treatment": False, "usable_as_control": False}, "pop", case, RULES, [], h) == []
    # the pack itself marks the field contested: citing it is enough
    h.contradictions = ["col:pop.when"]
    assert V.contradictions({"affected_by_treatment": True}, "pop", case, RULES, ["col:pop.when"], h) == []
    assert len(V.contradictions({"affected_by_treatment": True}, "pop", case, RULES, ["col:pop.note"], h)) == 1
    # a drafted field is not a fact and cannot be contradicted
    h.columns[0].provenance["when"].status = "drafted"
    assert V.contradictions({"affected_by_treatment": True}, "pop", C.weigh(h, {}), RULES, [], h) == []


def test_cites_must_resolve():
    h = pack()
    assert V.cites_resolve(["col:pop.note", "col:nope.note"], h) == ["citation 'col:nope.note' does not resolve in the pack"]


def test_a_departure_from_the_last_reading_needs_a_named_departure_that_cites():
    from causal_agent.common.contracts import Departure

    h = pack()
    drafted = {"affects_treatment": True}
    claims = {"affects_treatment": False, "affects_outcome": True}
    assert V.departures(claims, drafted, [], h) == [
        "affects_treatment = False departs from the last reading True with no departure named; keep the reading or list "
        "affects_treatment under departures with the reason and the cite that changed it"
    ]
    assert V.departures(claims, drafted, [Departure(claim="affects_treatment", reason="the note says the offer never saw it", cites=["col:pop.note"])], h) == []
    assert V.departures(claims, drafted, [Departure(claim="affects_treatment", reason="r", cites=[])], h) == [
        "the departure for affects_treatment cites nothing; cite what changed the reading"
    ]
    assert V.departures(claims, drafted, [Departure(claim="affects_treatment", reason="r", cites=["col:nope.note"])], h) == [
        "citation 'col:nope.note' does not resolve in the pack"
    ]
    # a departure named for another claim does not cover this one; an address this run made resolves through `also`
    assert len(V.departures(claims, drafted, [Departure(claim="affects_outcome", reason="r", cites=["col:pop.note"])], h)) == 1
    assert (
        V.departures(claims, drafted, [Departure(claim="affects_treatment", reason="r", cites=["probe:roles.1"])], h, also=lambda a: a == "probe:roles.1") == []
    )
    # keeping the reading needs nothing
    assert V.departures({"affects_treatment": True}, drafted, [], h) == []


def test_a_cite_copied_with_its_brackets_or_the_words_after_a_said_tag_still_resolves():
    from causal_agent.common.addresses import norm_address

    assert norm_address("[col:Margin.when]") == "col:margin.when" and norm_address("[claim:assignment.kind]") == "claim:assignment.kind"
    assert norm_address("said:1 about claim:assignment.kind") == "said:1" and norm_address("[said:3]") == "said:3"
    h = pack()
    assert V.cites_resolve(["[col:pop.when]"], h) == [] and V.cites_resolve(["col:nope.when"], h)


def test_resolves_reads_the_checks_the_pack_and_what_the_run_made():
    h = pack()
    made = lambda a: a == "ladder:trends.leads"  # noqa: E731
    assert V.resolves("check:c.pre_trends", h, checks=["check:c.pre_trends"])
    assert V.resolves("[check:c.pre_trends]", h, checks=["check:c.pre_trends"])
    assert V.resolves("col:pop.when", h) and V.resolves("ladder:trends.leads", h, also=made)
    assert not V.resolves("ladder:trends.gap_slope", h, also=made) and not V.resolves("STORY", h, checks=["check:c.x"], also=made)
