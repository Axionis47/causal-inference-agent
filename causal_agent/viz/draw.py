"""The drawing tool. The model writes one Python script that draws one figure from the file; the script runs in the
sandbox; the picture and the numbers it shows are kept as an Artifact the chat can cite. The prompt names no method and
no column: the columns and what is known arrive as data from the caller. Up to three tries; a failed try's error goes
into the next prompt; three failures are a Decline and no folder."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Decline, Thought
from causal_agent.common.llm import structured
from causal_agent.viz import sandbox, store
from causal_agent.viz.store import Artifact, Moment

TRIES = 3
STDERR_TAIL = 1500

SYSTEM = """You write one complete Python script that draws one figure from a CSV file, and nothing else.

The script:
- reads the CSV from the path in the environment variable VIZ_CSV with pandas
- draws one matplotlib figure with the Agg backend and saves it as figure.png in the current directory
- writes facts.json in the current directory: a JSON object from fact name to number, with every number the picture shows
  (a count, a share, a mean, a difference), so the reader can quote each one
- uses no network, writes no other file, and reads nothing but VIZ_CSV
- names columns by the file's own names, exactly as listed

Answer with the script, a caption of one sentence in the person's words on what the picture shows, and the names of the
numbers the script writes to facts.json."""

USER = """What to draw, in the person's words:
{ask}

What is known about the data:
{context}

The file's columns (key -> the file's own column name, use the file's own name):
{columns}
{failure}"""

FAILED = "\nPREVIOUS ATTEMPT FAILED. Fix the script.\n{why}\n"


class DrawRequest(BaseModel):
    dataset: str
    moment: Moment
    design: int | None = None
    memory_version: int = 0
    ask: str = Field(description="what to draw, in the person's words")
    context: str = Field(description="what is known about the data, as text with addresses")
    csv: Path = Field(description="the file, read-only for the code")
    columns: dict[str, str] = Field(description="key -> the file's own column name")


class DrawCode(BaseModel):
    """The model's answer: the script, what the picture shows, and the numbers it writes."""

    code: str = Field(description="a complete Python script")
    caption: str = Field(description="one sentence, in the person's words")
    facts: list[str] = Field(description="the names of the numbers the script writes to facts.json")


def draw(req: DrawRequest) -> tuple[Artifact | None, Decline | None, list[Thought]]:
    thoughts: list[Thought] = []
    artifact_id = store.new_id()
    out = store.folder(req.dataset, req.moment, req.design, artifact_id)
    columns = "\n".join(f"- {k} -> {v}" for k, v in req.columns.items())
    why = ""
    for _ in range(TRIES):
        failure = FAILED.format(why=why) if why else ""
        answer, thought = structured(DrawCode, SYSTEM, USER.format(ask=req.ask, context=req.context, columns=columns, failure=failure), node="draw")
        thoughts.append(thought)
        if out.exists():
            shutil.rmtree(out)
        ran = sandbox.run(answer.code, req.csv, out)
        if not ran.ok:
            why = ran.stderr[-STDERR_TAIL:] or "the script failed with no error text"
            continue
        facts, why = _check(out, answer.facts)
        if why:
            continue
        assert facts is not None
        (out / "request.json").write_text(json.dumps(req.model_dump(mode="json", exclude={"context"}), indent=2))
        a = Artifact(
            id=artifact_id,
            dataset=req.dataset,
            moment=req.moment,
            design=req.design,
            memory_version=req.memory_version,
            ask=req.ask,
            caption=answer.caption,
            facts=facts,
            made_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            files=sorted({*ran.files, "request.json", store.FILE}),
        )
        store.save(a)
        return a, None, thoughts
    _remove(out, store.home() / req.dataset)
    return None, Decline(stage="draw", kind="declined", about="artifact", check="draw.failed", reason=why), thoughts


            sandbox=ran.kind,
def _check(out: Path, names: list[str]) -> tuple[dict[str, float] | None, str]:
    """The picture is there, facts.json is an object of numbers, and every promised name is in it. Else why not."""
    png = out / "figure.png"
    if not png.is_file() or png.stat().st_size == 0:
        return None, "figure.png was not written or is empty"
    try:
        doc = json.loads((out / "facts.json").read_text())
    except (OSError, ValueError) as e:
        return None, f"facts.json could not be read as JSON: {e}"
    if not isinstance(doc, dict):
        return None, "facts.json is not a JSON object"
    bad = [k for k, v in doc.items() if isinstance(v, bool) or not isinstance(v, int | float)]
    if bad:
        return None, "facts.json has values that are not numbers: " + ", ".join(bad)
    missing = [n for n in names if n not in doc]
    if missing:
        return None, "facts.json lacks the promised numbers: " + ", ".join(missing)
    return {k: float(v) for k, v in doc.items()}, ""


def _remove(out: Path, stop: Path) -> None:
    """Remove the folder and any parent it leaves empty, up to the dataset's home."""
    shutil.rmtree(out, ignore_errors=True)
    p = out.parent
    while p != stop and p.is_dir() and not any(p.iterdir()):
        p.rmdir()
        p = p.parent
