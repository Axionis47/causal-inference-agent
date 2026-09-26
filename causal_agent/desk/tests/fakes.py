"""A scripted model for the desk: answers the frame, the mining, the inference, and the after-phase turn from the person's
words alone, so a conversation can be driven without Vertex. Shared by the desk and the server tests."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Candidate, FamilyDecision, QuestionFrame, Rejection, Scope
from causal_agent.desk.contracts import AfterReply, DeskAnswer, FieldUpdate, Inference
from causal_agent.families import registry as R
from causal_agent.memory.claims import ClaimUpdate, Extraction, FieldValue

FAMILIES = [f.name for f in R.knowledge()]
QUESTION = "Did completing the prep course raise math scores?"


def U(claim_kind, cites, column=None, **values):
    return ClaimUpdate(
        kind=claim_kind, column=column, reason="scripted", cites=list(cites), values=[FieldValue(name=k, value=str(v)) for k, v in values.items()]
    )


def students_doc_updates(cite="doc:context"):
    """What a careful reading of the students note yields: the dataset kinds and one measured claim per column."""
    ups = [
        U("grain", [cite], row_is="one student's exam results", panel="false"),
        U("sampling", [cite], how="whole", detail="every student who sat the exam"),
        U("change", [cite], what="a six-week test preparation course", to_whom="students at the school", when="the six weeks before the May 2026 exam"),
        U(
            "assignment",
            [cite],
            kind="own_choice",
            rule="offered first by lunch status and parental education, then open to anyone who asked",
            depends_on="lunch, parental level of education",
            treatment_column="test preparation course",
            treated_level="completed",
        ),
        U("unobserved", [cite], exists="false"),  # a description never sets a belief: this one must be dropped by mine
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
        ups.append(U("measured", [cite], column=c, meaning=f"{c} as recorded", when=when))
    return ups


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


def message_of(human: str) -> str:
    """The person's words from an INFER prompt, without the [user:turn:n] tag."""
    return human.split("THE PERSON SAYS\n")[1].split("\n")[0].split("] ", 1)[-1].strip()


def asked_addresses(human: str) -> list[str]:
    m = re.search(r"\(settles: ([^)]*)\)", human)
    return [a.strip() for a in m.group(1).split(",")] if m else []


class DeskFake:
    """Answers by rule from the person's words. `after` is a queue of AfterReply objects for the chat after a run."""

    def __init__(self, *, after: list[AfterReply] | None = None, infer=None, explain: list[DeskAnswer] | None = None):
        self.after = list(after or [])
        self.infer = infer  # optional callable(message, asked_addresses, human) -> Inference | None
        self.explain = list(explain or [])  # a queue of DeskAnswer for questions asked before the run
        self.calls: list[str] = []
        self.humans: dict[str, list[str]] = {}

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
        if schema is Extraction:
            return Extraction(updates=students_doc_updates(), confirmed=[], notes=[])
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
        if schema is Inference:
            msg, addrs = message_of(human), asked_addresses(human)
            if self.infer is not None:
                out = self.infer(msg, addrs, human)
                if out is not None:
                    return out
            return infer_by_rule(msg, addrs)
        if schema is FamilyDecision:
            return FamilyDecision(
                admissible=["adjustment"],
                chosen="adjustment",
                chosen_assumption="nothing else drove both",
                why_over_alternatives="scripted",
                rejected=[Rejection(family=f, reason="a need is unmet", cites=[]) for f in FAMILIES if f != "adjustment"],
                cites=[],
            )
        if schema is AfterReply:
            return self.after.pop(0)
        if schema is DeskAnswer:
            return self.explain.pop(0) if self.explain else DeskAnswer(text="A scripted answer.", cites=["adjustment"])
        raise AssertionError(schema)


def infer_by_rule(msg: str, addrs: list[str]) -> Inference:
    low = msg.lower()
    if low.startswith("yes") or low.startswith("all right") or low.startswith("right"):
        return Inference(confirms=addrs)
    if "don't know" in low or "do not know" in low:
        return Inference(unknown=addrs[:1])
    if "nothing hidden" in low:
        return Inference(updates=[FieldUpdate(address="claim:unobserved.exists", value="false", said=msg)])
    if "sit alone" in low or "no spillover" in low:
        return Inference(updates=[FieldUpdate(address="claim:spillover.possible", value="false", said=msg)])
    if "no nudge" in low:
        return Inference(updates=[FieldUpdate(address="claim:exclusion.exists", value="false", said=msg)])
    if "lunch was after" in low or "lunch: after" in low:
        return Inference(updates=[FieldUpdate(address="col:lunch.when", value="after", said=msg)])
    if "lunch was before" in low:
        return Inference(updates=[FieldUpdate(address="col:lunch.when", value="before", said=msg)])
    m = re.match(r"([a-z_:.]+) = (.+)", msg)  # "claim:sampling.how = whole"
    if m:
        return Inference(updates=[FieldUpdate(address=m.group(1), value=m.group(2), said=msg)])
    return Inference()


def answer_ask(payload: dict) -> str:
    """A person who confirms every draft, gives the beliefs the desk asks for, and knows nothing else."""
    a = payload.get("ask") or {}
    addrs = a.get("addresses") or []
    if a.get("kind") == "confirm":
        return "yes, all right"
    if addrs and "unobserved" in addrs[0]:
        return "nothing hidden"
    if addrs and "spillover" in addrs[0]:
        return "they sit alone"
    if addrs and "exclusion" in addrs[0]:
        return "no nudge"
    return "I don't know"
