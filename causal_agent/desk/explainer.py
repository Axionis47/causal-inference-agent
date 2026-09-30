"""The Explainer: a question and the material into a cited answer. One prompt before and after the run; one gate.

Before the run the material is the families' knowledge, the fit matrix, the probes, the steps of the conversation, the
person's own words and the memory, each line with an address (`before_material`). After the run it is what the run left
behind (`material.render`). The gate is code: every cite is an address in the material, every number stated matches its
address within one percent, and before the run only an answer or a drawing request is legal."""

from __future__ import annotations

import re
from typing import Literal

from causal_agent.common.contracts import Thought
from causal_agent.common.llm import structured
from causal_agent.desk import material as M
from causal_agent.desk.contracts import AfterReply, Exchange
from causal_agent.desk.nodes.shared import kinds_text
from causal_agent.desk.prompts import journey as P
from causal_agent.families import registry as R
from causal_agent.memory.journal import Step
from causal_agent.memory.matrix import Matrix
from causal_agent.memory.records import Memory

Phase = Literal["before", "after"]
TOL = 0.01
NUM_RE = re.compile(r"(?<![\w:.\-])-?\d+(?:\.\d+)?(?![\w.:\-]*[a-zA-Z_])")
BEFORE_KINDS = {"answer", "draw"}


# ------------------------------------------------------------------ the material before a run


def before_material(memory: Memory, matrix: Matrix | None, probes: list, steps: list[Step], asked: str | None = None) -> M.Material:
    """What the desk may cite before a run: `family:<name>` (the family's knowledge), `matrix:` cells, `probe:` results,
    `step:<n>`, `user:turn:<n>`, every `claim:`/`col:` field the memory holds, every picture drawn so far with its numbers, and
    what the desk was about to ask."""
    m = M.Material()
    M.add_artifacts(m, memory.name)
    for fam in R.knowledge():
        m.add(f"family:{fam.name}", fam.render())
    for line in matrix.render() if matrix is not None else []:
        address, _, text = line[1:].partition("] ")
        m.add(address, text)
    for p in probes:
        m.add(p.address, p.render().split("] ", 1)[-1], p.value)
    for s in steps:
        m.add(s.address, s.line())
    for said in memory.said:
        m.add(f"user:turn:{said.turn}", f'"{said.text}"' + (f" (about {said.about})" if said.about else ""))
    for c in memory.columns.values():
        m.add(c.address, f"column {c.name!r}")
    for address, f in memory.fields.items():
        if f.value is None and f.status == "empty":
            continue
        m.add(address, f.render(address).split("] ", 1)[-1], float(f.value) if isinstance(f.value, (int, float)) and not isinstance(f.value, bool) else None)
    for a in list(m.addresses):
        m.addresses.add(a.rsplit(".", 1)[0])
    if asked:
        m.add("desk.asked", asked)
    return m


# ------------------------------------------------------------------ the judgement


def exchanges_text(exchanges: list[Exchange]) -> str:
    return "\n".join(f"[{e.turn}] person: {e.user}\n[{e.turn}] you ({e.kind}): {e.assistant[:600]}" for e in exchanges[-6:]) or "(none yet)"


def answer_from(
    material: M.Material, memory: Memory | None, exchanges: list[Exchange], message: str, phase: Phase, errors: list[str]
) -> tuple[AfterReply, Thought]:
    """One reply from the material. `errors` are the gate's refusals of the last reply, shown so the model fixes them."""
    shown = ("\nPREVIOUS REPLY WAS REJECTED:\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else ""
    user = P.ANSWER_USER.format(
        material=material.text or "(nothing yet)",
        memory=(memory.render() if memory is not None else "") or "(nothing known)",
        kinds=kinds_text(),
        exchanges=exchanges_text(exchanges),
        phase="before the run: only answer and draw are legal" if phase == "before" else "after the run",
        message=message,
        errors=shown,
    )
    return structured(AfterReply, P.ANSWER_SYSTEM, user, node=f"explain:{phase}")


# ------------------------------------------------------------------ the gate


def _numbers_in(text: str) -> list[float]:
    out = []
    for tok in NUM_RE.findall(text):
        try:
            v = float(tok)
        except ValueError:
            continue
        if abs(v) >= 10 or "." in tok:
            out.append(v)
    return out


def _in_line(v: float, line: str) -> bool:
    for tok in NUM_RE.findall(line):
        try:
            x = float(tok)
        except ValueError:
            continue
        if abs(v - x) <= max(abs(x), 1e-9) * TOL or any(round(x, d) == v for d in range(5)):
            return True
    return False


def _grounded(v: float, mat: M.Material) -> bool:
    for x in mat.numbers.values():
        if abs(v - x) <= max(abs(x), 1e-9) * TOL:
            return True
        for d in (0, 1, 2, 3, 4):
            if round(x, d) == v:
                return True
    return f"{v:g}" in mat.text or str(v) in mat.text


def gate(reply: AfterReply, mat: M.Material, phase: Phase = "after") -> list[str]:
    """Why the reply is refused, or nothing. Cites and numbers are checked against the material; the kinds legal in this
    phase, and what each kind must carry."""
    errors: list[str] = []
    if phase == "before" and reply.kind not in BEFORE_KINDS:
        errors.append(f"{reply.kind} is not legal before the run; answer from the material, or draw")
        return errors
    if reply.kind == "answer":
        if not reply.cites and reply.figure:
            reply.cites = [reply.figure]
        if not reply.cites and reply.numbers:
            reply.cites = list(dict.fromkeys(n.address for n in reply.numbers))
        bad = [c for c in reply.cites if c not in mat.addresses]
        if bad:
            errors.append(f"cites not in the material: {bad}")
        if not reply.cites:
            errors.append(
                "an answer must cite at least one address; attach the addresses of the numbers you state, "
                "or say the material cannot answer and cite the nearest line"
            )
        for n in reply.numbers:
            if n.address not in mat.addresses:
                errors.append(f"number {n.value} attached to {n.address!r}, which is not in the material")
            elif (
                n.address in mat.numbers
                and abs(n.value - mat.numbers[n.address]) > max(abs(mat.numbers[n.address]), 1e-9) * TOL
                and not any(round(mat.numbers[n.address], d) == n.value for d in range(5))
                and not _in_line(n.value, mat.by_address.get(n.address, ""))
            ):
                errors.append(f"number {n.value} does not match {n.address} = {mat.numbers[n.address]:.6g}, and does not appear in that line's text")
        stated = {round(n.value, 6) for n in reply.numbers}
        for v in _numbers_in(reply.text):
            if round(v, 6) in stated or _grounded(v, mat):
                continue
            errors.append(f"the text states {v:g}, which is in no artifact; remove it or attach its address in numbers")
    if reply.figure and reply.figure not in mat.addresses:
        errors.append(f"figure {reply.figure!r} is not in the material; name one of the figure: addresses or leave it empty")
    if reply.kind in ("revise", "what_if"):
        if not reply.updates:
            errors.append(f"{reply.kind} needs at least one field update; if the person wants a design choice changed, answer with which field would change it")
    elif reply.kind == "requestion":
        if not (reply.question or "").strip():
            errors.append("requestion needs the new question in full")
    elif reply.kind == "draw":
        if not (reply.draw or "").strip():
            errors.append("draw needs what to draw, in the person's words")
    return errors
