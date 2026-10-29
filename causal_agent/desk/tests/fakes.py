"""A scripted model for the desk: answers the frame, the Reader over a note or a message, the Explainer before and after the
run, and the Designer from the person's words alone, so a conversation can be driven without Vertex. Shared by the desk and
the server tests."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Candidate, Cited, DecisionMade, DesignBrief, FamilyDecision, PrefilterVote, QuestionFrame, Rejection, Scope
from causal_agent.desk.contracts import AfterReply, FieldUpdate, Reading
from causal_agent.explore.contracts import Exploration
from causal_agent.explore.inspection import ExploreAction
from causal_agent.explore.tools import RunCodeStep
from causal_agent.families import registry as R

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

df = pd.read_csv(os.environ["DATA_CSV"])
means = df.groupby("lunch")["math score"].mean()
fig, ax = plt.subplots()
ax.bar(means.index, means.values)
ax.set_ylabel("mean math score")
fig.savefig("figure.png")
json.dump({"mean_standard": float(means["standard"]), "mean_free_reduced": float(means["free/reduced"])}, open("facts.json", "w"))
"""
EXPLORED = Exploration(
    summary="Mean math score by lunch, standard against free or reduced.",
    step=1,
    facts=["mean_standard", "mean_free_reduced"],
    figure=True,
    method="Compare observed group means.",
    findings=["The plot shows group means [figure]."],
    limitations=["Descriptive, not causal."],
)
BASELINE_DRAW_SCRIPT = """
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(os.environ["DATA_CSV"])
counts = df["lunch"].value_counts()
fig, ax = plt.subplots()
ax.bar(counts.index, counts.values)
ax.set_ylabel("students")
fig.savefig("figure.png")
json.dump({"count_standard": int(counts["standard"]), "count_free_reduced": int(counts["free/reduced"])}, open("facts.json", "w"))
"""
BASELINE_EXPLORED = Exploration(
    summary="Student counts by lunch, standard against free or reduced.",
    step=1,
    facts=["count_standard", "count_free_reduced"],
    figure=True,
    method="Count students in each lunch group.",
    findings=["The plot shows group sizes [figure]."],
    limitations=["Counts only."],
)
TABLE_SCRIPT = """
import json, os
import pandas as pd

df = pd.read_csv(os.environ["DATA_CSV"])
df["high reading"] = (df["reading score"] >= 70).astype(int)
df.to_csv("table.csv", index=False)
json.dump({"rows": int(len(df)), "columns": int(df.shape[1])}, open("facts.json", "w"))
"""
TRANSFORMED = Exploration(summary="Added the high reading flag; every row kept its place.", step=1, facts=["rows"], table="table.csv")
TRANSFORM_WORDS = ("add a column", "drop the", "keep only", "recode", "rename")


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
    names = re.findall(r"^\s+- ([a-z_.]+): ", block, flags=re.M)
    cite = "claim:assignment.kind"
    return DesignBrief(
        family=family,
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
        explore: list[Exploration] | None = None,
        brief: list[DesignBrief] | None = None,
    ):
        self.after = list(after or [])
        self.brief = list(brief or [])  # a queue of design briefs; empty means one by rule from the prompt
        self.explore = list(explore or [])  # a queue of the exploration agent's records; empty means the one that drew math score by lunch
        self.cites = list(cites or [])
        self.infer = infer  # optional callable(message, asked_addresses, human) -> Reading | None, for the Reader over a message
        self.explain = list(explain or [])  # a queue of AfterReply for questions asked before the run
        self.calls: list[str] = []
        self.humans: dict[str, list[str]] = {}

    name = "fake"

    def predict(self, state, questions) -> dict:
        """The decision model by rule, confident on the same cues `read_by_rule` reads and unsure on anything else: a run word is
        run, plot/show/draw is explore, a question mark is ask, a known phrase about the data is update."""
        msg = str(state.get("message", "") if isinstance(state, dict) else state)
        low = msg.lower().strip()
        self.calls.append("decide")

        def sure(kind, about):
            return {"answers": {"kind": {"choice": kind, "answer_confidence": 0.95}, "about_data": {"noul": about, "answer_confidence": 0.95}}}

        if low in {"run", "go", "run it", "run the analysis", "yes, run it", "let's run it"} or "run the analysis" in low:
            return sure("run", 0.05)
        if low.startswith(("plot", "show", "draw")):
            return sure("explore", 0.05)
        if low.startswith(TRANSFORM_WORDS):
            return sure("transform", 0.05)
        if msg == STORY_ANSWER or low.startswith(("yes", "all right", "right", "none for all", "not that i know")) or "don't know" in low or " = " in msg:
            return sure("update", 0.95)
        if any(w in low for w in ("nothing hidden", "sit alone", "no spillover", "no nudge", "lunch was", "lunch:")):
            return sure("update", 0.95)
        if low.endswith("?"):
            return sure("ask", 0.05)
        return {
            "answers": {"kind": {"choice": "update", "answer_confidence": 0.3}, "about_data": {"noul": 0.5, "answer_confidence": 0.3}}
        }  # unsure: the Reader decides

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
        if isinstance(human, list):
            human = "\n".join(part.get("text", "[image]") for part in human)
        self.calls.append(schema.__name__)
        self.humans.setdefault(schema.__name__, []).append(human)
        if schema is Reading and source_of(human).startswith("doc:"):
            return Reading(kind="update", updates=students_doc_updates())
        if schema.__name__ == "OpeningIntent":
            return schema(kind="causal")
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
        if schema is ExploreAction:
            if "Image pixels attached" in human:
                return ExploreAction(done=True)
            if "step 1 ran" in human:
                return ExploreAction(action="inspect_output", reference="step_1/figure.png")
            return ExploreAction(action="run_code", code=BASELINE_DRAW_SCRIPT if "student counts" in human else DRAW_SCRIPT)
        if schema is RunCodeStep:  # the agent's look: one script (a picture, or a new table when one is asked for), then done
            ran = "WHAT YOU RAN SO FAR" in human
            table = "new version of the whole table" in human
            fenced = "is not in the table you were given" in human
            return RunCodeStep(done=True) if ran else RunCodeStep(code=TABLE_SCRIPT if table else BASELINE_DRAW_SCRIPT if fenced else DRAW_SCRIPT)
        if schema is Exploration:
            if self.explore:
                return self.explore.pop(0)
            if "new version of the whole table" in human:
                return TRANSFORMED
            return BASELINE_EXPLORED if "is not in the table you were given" in human or "student counts" in human else EXPLORED
        raise AssertionError(schema)


def read_by_rule(msg: str, addrs: list[str]) -> Reading:
    """The Reader by rule: the scripted story fills many fields at once; a yes confirms what was asked; a few phrases fill one
    field each; an address with '=' fills that field. The kind by cue: a run word is run, a message that starts with plot, show
    or draw is explore, one that ends with a question mark and fills nothing is ask, everything else is update."""
    low = msg.lower()
    if low in {"run", "go", "run it", "run the analysis", "yes, run it", "let's run it"} or "run the analysis" in low:
        return Reading(kind="run")
    if low.startswith(("plot", "show", "draw")):
        return Reading(kind="explore", explore=msg)
    if low.startswith(TRANSFORM_WORDS):
        return Reading(kind="transform", transform=msg)
    if msg == STORY_ANSWER:
        return Reading(kind="update", updates=students_story_updates())
    if low.startswith("yes") or low.startswith("all right") or low.startswith("right"):
        return Reading(kind="update", confirms=addrs)
    if "don't know" in low or "do not know" in low:
        return Reading(kind="update", unknown=addrs)
    if low.startswith("none for all"):
        return Reading(kind="update", updates=[FieldUpdate(address=a, value="none", said=msg) for a in addrs])
    if low.startswith("not that i know"):
        return Reading(kind="update", updates=[FieldUpdate(address=a, value="false", said=msg) for a in addrs])
    if "nothing hidden" in low:
        return Reading(kind="update", updates=[FieldUpdate(address="claim:unobserved.exists", value="false", said=msg)])
    if "sit alone" in low or "no spillover" in low:
        return Reading(kind="update", updates=[FieldUpdate(address="claim:spillover.possible", value="false", said=msg)])
    if "no nudge" in low:
        return Reading(kind="update", updates=[FieldUpdate(address="claim:exclusion.exists", value="false", said=msg)])
    if "lunch was after" in low or "lunch: after" in low:
        return Reading(kind="update", updates=[FieldUpdate(address="col:lunch.when", value="after", said=msg)])
    if "lunch was before" in low:
        return Reading(kind="update", updates=[FieldUpdate(address="col:lunch.when", value="before", said=msg)])
    matches = list(
        re.finditer(r"((?:claim|col):[a-z_.]+) = ([^\n]+)", msg)
    )  # "I'd say claim:sampling.how = whole" (a bare `address = value` line never reaches a reader)
    if matches:
        return Reading(kind="update", updates=[FieldUpdate(address=m.group(1), value=m.group(2), said=msg) for m in matches])
    if msg.rstrip().endswith("?"):
        return Reading(kind="ask", question=msg)
    return Reading(kind="update")


def answer_ask(payload: dict) -> str:
    """A person who tells the scripted story when asked for it, confirms every draft, gives the beliefs the desk asks for, and
    knows nothing else."""
    a = payload.get("ask") or {}
    addrs = a.get("addresses") or []
    if payload.get("kind") == "ready":
        return "yes, run it"
    if payload.get("kind") == "chat":
        return "let's run it"
    if a.get("kind") == "story":
        return STORY_ANSWER
    if a.get("kind") == "confirm":
        return "yes, all right"
    known = {"claim:sampling.how": "whole", "claim:unobserved.exists": "false", "claim:spillover.possible": "false"}
    if len(addrs) > 1 and all(addr in known for addr in addrs):
        return "\n".join(f"{addr} = {known[addr]}" for addr in addrs)
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
