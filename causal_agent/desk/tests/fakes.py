"""A scripted model for the desk: answers the frame, the Reader over a note or a message, the Explainer before and after the
run, and the Designer from the person's words alone, so a conversation can be driven without Vertex. Shared by the desk and
the server tests."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Candidate, Cited, DecisionMade, DesignBrief, FamilyDecision, PrefilterVote, QuestionFrame, Rejection, Scope
from causal_agent.desk.contracts import AfterReply, FieldUpdate, Reading
from causal_agent.families import registry as R
from causal_agent.viz.draw import DrawCode

FAMILIES = [f.name for f in R.knowledge()]
QUESTION = "Did completing the prep course raise math scores?"
STORY_ANSWER = (
    "Each row is one student's exam results. The change was a six-week test preparation course, offered to students at the school "
    "in the six weeks before the May 2026 exam. Places were offered first by lunch status and parental education, then open to "
    "anyone who asked; the course column says who completed it. Lunch and parental level of education were recorded at enrolment; "
    "the math score is from the exam sat after the course."
)

# a script the drawing tool can run on the students file: mean math score by lunch, two numbers kept
DRAW_SCRIPT = """
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(os.environ["VIZ_CSV"])
means = df.groupby("lunch")["math score"].mean()
fig, ax = plt.subplots()
ax.bar(means.index, means.values)
ax.set_ylabel("mean math score")
fig.savefig("figure.png")
json.dump({"mean_standard": float(means["standard"]), "mean_free_reduced": float(means["free/reduced"])}, open("facts.json", "w"))
"""
DRAW = DrawCode(code=DRAW_SCRIPT, caption="Mean math score by lunch, standard against free or reduced.", facts=["mean_standard", "mean_free_reduced"])


def U(address: str, value, said: str = "scripted") -> FieldUpdate:
    return FieldUpdate(address=address, value=str(value), said=said, reason="scripted")


def students_doc_updates() -> list[FieldUpdate]:
    """What a careful reading of the students note yields: the dataset kinds and one measured claim per column, each with the
    sentence it rests on."""
    ups = [
        U("claim:grain.row_is", "one student's exam results", "one row per student"),
        U("claim:grain.panel", "false", "one row per student"),
        U("claim:sampling.how", "whole", "every student who sat the exam"),
        U("claim:sampling.detail", "every student who sat the exam", "every student who sat the exam"),
        U("claim:change.what", "a six-week test preparation course", "a six-week test preparation course"),
        U("claim:change.to_whom", "students at the school", "offered to students at the school"),
        U("claim:change.when", "the six weeks before the May 2026 exam", "in the six weeks before the May 2026 exam"),
        U("claim:assignment.kind", "own_choice", "then open to anyone who asked"),
        U("claim:assignment.rule", "offered first by lunch status and parental education, then open to anyone who asked", "offered first by lunch status"),
        U("claim:assignment.depends_on", "lunch, parental level of education", "offered first by lunch status and parental education"),
        U("claim:assignment.treatment_column", "test preparation course", "the course column says who completed it"),
        U("claim:assignment.treated_level", "completed", "the course column says who completed it"),
        U("claim:unobserved.exists", "false", "nothing else mattered"),  # a description never sets a belief: the Reader drops this one
    ]
    cols = {
        "gender": "before",
        "race/ethnicity": "before",
        "parental level of education": "before",
        "lunch": "before",
        "test preparation course": "at",
        "math score": "after",
        "reading score": "after",
        "writing score": "after",
    }
    for c, when in cols.items():
        key = re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()
        ups.append(U(f"col:{key}.meaning", f"{c} as recorded", f"{c} as recorded"))
        ups.append(U(f"col:{key}.when", when, f"{c} was recorded {'at enrolment' if when == 'before' else 'at the course' if when == 'at' else 'at the exam'}"))
    return ups


def students_story_updates() -> list[FieldUpdate]:
    """What the Reader makes of the scripted story: the assignment, the change, the grain and the in-play columns, with the
    sentence each rests on."""
    return [
        U("claim:grain.row_is", "one student's exam results", "Each row is one student's exam results."),
        U("claim:grain.panel", "false", "Each row is one student's exam results."),
        U("claim:change.what", "a six-week test preparation course", "The change was a six-week test preparation course"),
        U("claim:change.to_whom", "students at the school", "offered to students at the school"),
        U("claim:change.when", "the six weeks before the May 2026 exam", "in the six weeks before the May 2026 exam"),
        U("claim:assignment.kind", "own_choice", "then open to anyone who asked"),
        U("claim:assignment.rule", "offered first by lunch status and parental education, then open to anyone who asked", "Places were offered first"),
        U("claim:assignment.depends_on", "lunch, parental level of education", "offered first by lunch status and parental education"),
        U("claim:assignment.treatment_column", "test preparation course", "the course column says who completed it"),
        U("claim:assignment.treated_level", "completed", "the course column says who completed it"),
        U("col:lunch.meaning", "lunch status at enrolment", "Lunch and parental level of education were recorded at enrolment"),
        U("col:lunch.when", "before", "Lunch and parental level of education were recorded at enrolment"),
        U("col:parental_level_of_education.meaning", "the parents' education at enrolment", "recorded at enrolment"),
        U("col:parental_level_of_education.when", "before", "recorded at enrolment"),
        U("col:test_preparation_course.meaning", "whether the student completed the course", "the course column says who completed it"),
        U("col:test_preparation_course.when", "at", "the course column says who completed it"),
        U("col:math_score.meaning", "the math mark on the May exam", "the math score is from the exam sat after the course"),
        U("col:math_score.when", "after", "the math score is from the exam sat after the course"),
    ]


def students_frame() -> QuestionFrame:
    c = lambda col, why, a: Candidate(column=col, reason=why, cites=[a])  # noqa: E731
    return QuestionFrame(
        intent="effect_of_change",
        decision_served="whether to keep running the course",
        outcome_candidates=[c("math score", "the question asks about math scores", "col:math_score.note")],
        cause_candidates=[c("test preparation course", "a decision by the counsellor", "col:test_preparation_course.note")],
        scope=Scope(),
        relevant_columns=[
            c("math score", "outcome", "col:math_score.note"),
            c("test preparation course", "cause", "col:test_preparation_course.note"),
            c("lunch", "the course note says places depended on it", "col:test_preparation_course.note"),
            c("parental level of education", "same", "col:test_preparation_course.note"),
        ],
        reasons=[],
    )


def brief_by_rule(human: str) -> DesignBrief:
    """A brief from the DESIGN prompt alone: the family and its decision names from the family block, the back door for a family that
    lists a road, one cite every memory holds once the assignment is settled."""
    block = human.split("THE FAMILY (its decisions are the ones to fill, by name)\n")[1].split("\n\nTHE MEMORY")[0]
    family = re.search(r"family: (\S+)", block).group(1)
    names = re.findall(r"^\s+- ([a-z_]+): ", block, flags=re.M)
    cite = "claim:assignment.kind"
    return DesignBrief(
        family=family,
        road="backdoor" if "road" in names else None,
        target="average",
        decisions=[DecisionMade(name=n, choice=f"{n} decided from the file", rests_on=[cite], reason="scripted") for n in names],
        threats=[Cited(reason="something outside the file could have driven both", cites=[cite])],
        checks=["the balance of each adjustment column across the arms"],
        bets_on="nothing beyond lunch and parents' education drove both the course and the score",
    )


def message_of(human: str) -> str:
    """The material from a READ prompt, without the [source] tag: the person's words this turn."""
    return human.split("THE MATERIAL\n")[1].split("\n")[0].split("] ", 1)[-1].strip()


def source_of(human: str) -> str:
    """The source tag of a READ prompt's material: user:turn:<n> or doc:<name>."""
    return human.split("THE MATERIAL\n[")[1].split("]", 1)[0]


def asked_addresses(human: str) -> list[str]:
    m = re.search(r"\(settles: ([^)]*)\)", human)
    return [a.strip() for a in m.group(1).split(",")] if m else []


def before_the_run(human: str) -> bool:
    """Whether an ANSWER prompt is the Explainer before the run."""
    return "THE PHASE\nbefore the run" in human


class DeskFake:
    """Answers by rule from the person's words. `after` is a queue of AfterReply objects for the chat after a run and `explain` one
    for the Explainer before it; `cites` is what the family decision cites, for the gate; `brief` is a queue of design briefs, else
    one by rule."""

    def __init__(
        self,
        *,
        after: list[AfterReply] | None = None,
        infer=None,
        explain: list[AfterReply] | None = None,
        cites: list[str] | None = None,
        draw: list[DrawCode] | None = None,
        brief: list[DesignBrief] | None = None,
    ):
        self.after = list(after or [])
        self.brief = list(brief or [])  # a queue of design briefs; empty means one by rule from the prompt
        self.draw = list(draw or [])  # a queue of scripts for the drawing tool; empty means the one that draws math score by lunch
        self.cites = list(cites or [])
        self.infer = infer  # optional callable(message, asked_addresses, human) -> Reading | None, for the Reader over a message
        self.explain = list(explain or [])  # a queue of AfterReply for questions asked before the run
        self.calls: list[str] = []
        self.humans: dict[str, list[str]] = {}

    def reads(self, source: str = "user:") -> list[str]:
        """The READ prompts whose material carried this source prefix."""
        return [h for h in self.humans.get("Reading", []) if source_of(h).startswith(source)]

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                human = messages[-1][1]
                parsed = fake.answer(schema, human)
                raw = AIMessage(
                    content=[{"type": "thinking", "thinking": f"thinking about {schema.__name__}"}, "{}"],
                    usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "output_token_details": {"reasoning": 7}},
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return R()

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        self.humans.setdefault(schema.__name__, []).append(human)
        if schema is Reading and source_of(human).startswith("doc:"):
            return Reading(updates=students_doc_updates())
        if schema is QuestionFrame:
            q = human.split("QUESTION\n")[1].split("\n")[0].lower()
            if "how many" in q or "what share" in q:
                return QuestionFrame(
                    intent="not_causal", decision_served="a count", outcome_candidates=[], cause_candidates=[], scope=Scope(), relevant_columns=[], reasons=[]
                )
            if "why" in q and "raise" not in q:
                return QuestionFrame(
                    intent="root_cause",
                    decision_served="why",
                    outcome_candidates=[Candidate(column="math score", reason="r", cites=["col:math_score.note"])],
                    cause_candidates=[],
                    scope=Scope(),
                    relevant_columns=[],
                    reasons=[],
                )
            if "gender" in q:  # a question whose outcome is one the file cannot move: the change is the outcome
                fr = students_frame()
                fr.outcome_candidates = [Candidate(column="test preparation course", reason="r", cites=["col:test_preparation_course.note"])]
                return fr
            return students_frame()
        if schema is Reading:
            msg, addrs = message_of(human), asked_addresses(human)
            if self.infer is not None:
                out = self.infer(msg, addrs, human)
                if out is not None:
                    return out
            return read_by_rule(msg, addrs)
        if schema is PrefilterVote:
            return PrefilterVote(column="x", relevant=True, reason="r", cites=["dataset.note"])
        if schema is FamilyDecision:
            return FamilyDecision(
                admissible=["adjustment"],
                chosen="adjustment",
                chosen_assumption="nothing else drove both",
                why_over_alternatives="scripted",
                rejected=[Rejection(family=f, reason="a need is unmet", cites=self.cites) for f in FAMILIES if f != "adjustment"],
                cites=self.cites,
            )
        if schema is DesignBrief:
            return self.brief.pop(0) if self.brief else brief_by_rule(human)
        if schema is AfterReply and before_the_run(human):
            return self.explain.pop(0) if self.explain else AfterReply(kind="answer", text="A scripted answer.", cites=["family:adjustment"])
        if schema is AfterReply:
            return self.after.pop(0)
        if schema is DrawCode:
            return self.draw.pop(0) if self.draw else DRAW
        raise AssertionError(schema)


def read_by_rule(msg: str, addrs: list[str]) -> Reading:
    """The Reader by rule: the scripted story fills many fields at once; a yes confirms what was asked; a few phrases fill one
    field each; an address with '=' fills that field."""
    low = msg.lower()
    if msg == STORY_ANSWER:
        return Reading(updates=students_story_updates())
    if low.startswith("yes") or low.startswith("all right") or low.startswith("right"):
        return Reading(confirms=addrs)
    if "don't know" in low or "do not know" in low:
        return Reading(unknown=addrs)
    if low.startswith("none for all"):
        return Reading(updates=[FieldUpdate(address=a, value="none", said=msg) for a in addrs])
    if low.startswith("not that i know"):
        return Reading(updates=[FieldUpdate(address=a, value="false", said=msg) for a in addrs])
    if "nothing hidden" in low:
        return Reading(updates=[FieldUpdate(address="claim:unobserved.exists", value="false", said=msg)])
    if "sit alone" in low or "no spillover" in low:
        return Reading(updates=[FieldUpdate(address="claim:spillover.possible", value="false", said=msg)])
    if "no nudge" in low:
        return Reading(updates=[FieldUpdate(address="claim:exclusion.exists", value="false", said=msg)])
    if "lunch was after" in low or "lunch: after" in low:
        return Reading(updates=[FieldUpdate(address="col:lunch.when", value="after", said=msg)])
    if "lunch was before" in low:
        return Reading(updates=[FieldUpdate(address="col:lunch.when", value="before", said=msg)])
    m = re.match(r"([a-z_:.]+) = (.+)", msg)  # "claim:sampling.how = whole"
    if m:
        return Reading(updates=[FieldUpdate(address=m.group(1), value=m.group(2), said=msg)])
    return Reading()


def answer_ask(payload: dict) -> str:
    """A person who tells the scripted story when asked for it, confirms every draft, gives the beliefs the desk asks for, and
    knows nothing else."""
    a = payload.get("ask") or {}
    addrs = a.get("addresses") or []
    if a.get("kind") == "story":
        return STORY_ANSWER
    if a.get("kind") == "confirm":
        return "yes, all right"
    if addrs and "unobserved" in addrs[0]:
        return "nothing hidden"
    if addrs and "spillover" in addrs[0]:
        return "they sit alone"
    if addrs and "exclusion" in addrs[0]:
        return "no nudge"
    if addrs and all(a.endswith((".offer_column", ".uptake_column", ".same_as", ".nested_in")) for a in addrs):
        return "none for all"
    if addrs and all(a.endswith(".may_modify") for a in addrs):
        return "not that I know"
    return "I don't know"
