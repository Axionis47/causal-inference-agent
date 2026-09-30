"""An episode: the model looks with the tools, every fact gets an address, the budget and the outcome rule hold, the gate
re-prompts with the log kept, and a fake drives it all."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from causal_agent.common.llm import set_llm
from causal_agent.lane import episode as E
from causal_agent.lane import tools as TL
from causal_agent.lane.tests.test_tools import table


class Record(BaseModel):
    ok: bool = True
    cites: list[str] = Field(default_factory=list)


class Fake:
    """Scripted: each entry of `looks` is the tool calls of one round of looking (an empty list stops looking); `answers` are
    the records, in order."""

    def __init__(self, looks, answers):
        self.looks = list(looks)
        self.answers = list(answers)
        self.bound: list[str] = []
        self.rounds: list[list] = []
        self.humans: list[str] = []
        self.systems: list[str] = []

    def bind_tools(self, tools):
        self.bound = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.rounds.append(list(messages))
        calls = self.looks.pop(0) if self.looks else []
        return AIMessage(
            content=[{"type": "thinking", "thinking": "looking"}] if calls else "",
            tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(calls)],
        )

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                fake.systems.append(messages[0][1])
                fake.humans.append(messages[-1][1])
                return {"raw": AIMessage(content=""), "parsed": fake.answers.pop(0), "parsing_error": None}

        return R()


def tools() -> TL.Tools:
    df = table()
    return TL.Tools(df, outcome="Math Score", treatment="test preparation course", treated=df["test preparation course"] == "completed")


def gate_cites(record: Record, log: E.EpisodeLog) -> list[str]:
    errs = [f"citation {c!r} does not resolve" for c in record.cites if not log.resolve(c)]
    return errs + ([] if record.ok else ["not ok"])


def run(fake, **kw):
    set_llm(fake)
    try:
        return E.run_episode(Record, "SYSTEM", "USER", tools=tools(), budget=kw.pop("budget", 4), gate=gate_cites, node="roles", **kw)
    finally:
        set_llm(None)


def test_the_model_looks_then_answers_and_every_fact_has_an_address():
    fake = Fake([[("by_arm", {"column": "lunch"}), ("timing", {})], []], [Record(cites=["probe:roles.1", "probe:roles.2"])])
    rec, log, thoughts, errors = run(fake)
    assert rec is not None and errors == [] and log.tries == 1 and log.calls == 2
    assert [f.address for f in log.facts] == ["probe:roles.1", "probe:roles.2"] and log.facts[0].tool == "by_arm"
    assert fake.bound == list(TL.NAMES)
    # the tool phase: the rule is in the system message; each result went back as a tool message with its address
    first = fake.rounds[0]
    assert "at most 4 calls" in first[0].content and "[probe:roles.<n>]" in first[0].content and first[1].content == "USER"
    second = fake.rounds[1]
    tool_msgs = [m for m in second if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["c0", "c1"] and tool_msgs[0].content.startswith("[probe:roles.1] by_arm(column='lunch'): 'lunch' by arm")
    # the answer: a fresh structured call whose material carries the facts
    assert fake.humans == [fake.humans[0]] and "FACTS YOU ASKED FOR\n[probe:roles.1]" in fake.humans[0] and "[probe:roles.2] timing()" in fake.humans[0]
    assert fake.systems == ["SYSTEM"]
    assert [t.node for t in thoughts] == ["roles:look", "roles"]


def test_the_outcome_rule_reaches_the_model_and_leaves_no_fact():
    fake = Fake([[("by_arm", {"column": "Math Score"}), ("association", {"a": "test preparation course", "b": "Math Score"})], []], [Record()])
    rec, log, _, _ = run(fake)
    assert rec is not None and log.facts == [] and len(log.refusals) == 2 and log.calls == 2
    tool_msgs = [m for m in fake.rounds[1] if isinstance(m, ToolMessage)]
    assert all(m.content.startswith("refused: that would join the outcome 'Math Score' with the treatment") for m in tool_msgs)
    assert "FACTS YOU ASKED FOR" not in fake.humans[0] and "Math Score" not in fake.humans[0]


def test_the_budget_bounds_the_looking():
    many = [("describe", {"column": c}) for c in ("lunch", "district", "school", "reading score", "parental level of education")]
    fake = Fake([many, many], [Record()])
    rec, log, _, _ = run(fake, budget=2)
    assert rec is not None and log.calls == 2 and len(log.facts) == 2
    assert len(fake.rounds) == 1 and fake.looks == [many]  # the budget was spent in the first round; no second was asked for
    # a call past the budget is answered with the budget, not run
    assert E._run(tools(), "describe", {"column": "school"}, 2, log).startswith("refused: the budget of 2 calls is spent") and log.calls == 2


def test_a_refused_answer_goes_back_with_the_log_kept_and_the_model_may_look_again():
    fake = Fake(
        [[("describe", {"column": "lunch"})], [], [("describe", {"column": "district"})], []],
        [Record(cites=["probe:roles.9"]), Record(cites=["probe:roles.1", "probe:roles.2"])],
    )
    rec, log, thoughts, errors = run(fake)
    assert rec is not None and errors == [] and log.tries == 2 and [f.address for f in log.facts] == ["probe:roles.1", "probe:roles.2"]
    assert "PREVIOUS ANSWER WAS REJECTED" in fake.humans[1] and "'probe:roles.9' does not resolve" in fake.humans[1]
    assert "[probe:roles.1]" in fake.humans[1] and "[probe:roles.2]" in fake.humans[1]
    assert "PREVIOUS ANSWER WAS REJECTED" in fake.rounds[2][1].content  # the second look saw the errors too


def test_three_refusals_return_no_record_and_the_errors():
    fake = Fake([[]] * 3, [Record(ok=False)] * 3)
    rec, log, _, errors = run(fake)
    assert rec is None and log.tries == 3 and errors == ["not ok"]


def test_no_tools_means_one_structured_call_and_no_binding():
    fake = Fake([], [Record()])
    set_llm(fake)
    try:
        rec, log, thoughts, _ = E.run_episode(Record, "S", "U", tools=None, budget=3, gate=gate_cites, node="road")
    finally:
        set_llm(None)
    assert rec is not None and fake.bound == [] and fake.rounds == [] and fake.humans == ["U"] and [t.node for t in thoughts] == ["road"]


def test_a_wrong_tool_name_or_arguments_is_a_refusal_not_a_crash():
    fake = Fake([[("drop_table", {}), ("describe", {"col": "lunch"})], []], [Record()])
    rec, log, _, _ = run(fake)
    assert rec is not None and log.facts == [] and [r.reason[:8] for r in log.refusals] == ["refused:", "refused:"]
    assert "no tool named 'drop_table'" in log.refusals[0].reason and "could not run" in log.refusals[1].reason
