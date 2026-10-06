"""The decision model: a small encoder that answers typed questions about a text in one forward pass, with a calibrated
confidence, and no words. Two backends on one wire, the one Jev defined and Laya serves: `laya` runs the open checkpoint in
this process; `jev` posts to a hosted endpoint. The chokepoint that calls either is `common/llm.py::decide`; nothing else
invokes a decision model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from causal_agent.common import config


class Answer(BaseModel):
    """One question's answer: the choice, the probability or the score, and how sure the model is of it."""

    choice: str | None = None
    noul: float | None = Field(default=None, description="for a yes/no question: the probability that it is true")
    score: float | None = None
    confidence: float = Field(default=0.0, description="the calibrated confidence in the answer, 0 to 1")
    probabilities: dict[str, float] = Field(default_factory=dict, description="for a choice: every option's probability")


class Decisions(BaseModel):
    answers: dict[str, Answer]
    model: str = Field(description="which backend and checkpoint answered")


class Decider(Protocol):
    name: str

    def predict(self, state: dict[str, Any] | str, questions: dict[str, Any]) -> dict[str, Any]: ...


def parse(raw: dict[str, Any], model: str) -> Decisions:
    """The wire's answer as a record. Laya and Jev agree on `answers`, `choice`, `noul`, `score` and `probabilities`; the
    calibrated confidence is `answer_confidence` on Laya and `confidence` elsewhere."""
    answers = {}
    for name, a in (raw.get("answers") or {}).items():
        a = a or {}
        conf = a.get("answer_confidence")
        if conf is None:
            conf = a.get("confidence")
        answers[name] = Answer(
            choice=a.get("choice"),
            noul=a.get("noul"),
            score=a.get("score"),
            confidence=float(conf or 0.0),
            probabilities={str(k): float(v) for k, v in (a.get("probabilities") or {}).items()},
        )
    return Decisions(answers=answers, model=model)


def laya_models(checkpoint: str | None) -> dict[str, str] | None:
    """What the router is told about checkpoints: nothing, or a fine-tuned directory standing in for the English one. The
    router keeps its name for it, so the desk's questions reach the fine-tune wherever they would have reached the open
    checkpoint."""
    return {"english": checkpoint} if checkpoint else None


class Laya:
    """The open checkpoint, in this process, or a fine-tune of it (`evals/decider.py` trains one on the labelled turns;
    `DECIDER_CHECKPOINT` names its directory). The router picks the checkpoint by the text's script; preloading is the
    caller's choice, since the first prediction otherwise pays the load."""

    def __init__(self, checkpoint: str | None = None, preload: bool = False) -> None:
        from laya import Router  # torch and the checkpoint come with it; imported only when this backend is used

        self.name = "laya" if checkpoint is None else f"laya:{Path(checkpoint).name}"
        self.router = Router(models=laya_models(checkpoint), preload=preload)

    def predict(self, state: dict[str, Any] | str, questions: dict[str, Any]) -> dict[str, Any]:
        return dict(self.router.predict(state, questions))


class Jev:
    """A hosted endpoint on the same wire: Jev itself, or a `laya-serve` elsewhere."""

    name = "jev"

    def __init__(self, url: str, key: str | None, model: str = "jev-latest") -> None:
        self.url, self.key, self.model = url.rstrip("/"), key, model

    def predict(self, state: dict[str, Any] | str, questions: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = httpx.post(self.url + "/v1/systemone", json={"model": self.model, "state": state, "questions": questions}, headers=headers, timeout=30)
        r.raise_for_status()
        return dict(r.json())


def build(cfg: config.Decider | None = None) -> Decider:
    """The backend the config names."""
    d = cfg or config.get().decider
    if d.kind == "laya":
        return Laya(checkpoint=d.checkpoint)
    if d.kind == "jev":
        if not d.url:
            raise ValueError("DECIDER=jev needs DECIDER_URL")
        return Jev(d.url, d.key)
    raise ValueError(f"DECIDER={d.kind!r} is not a decision model this desk knows (laya, jev)")
