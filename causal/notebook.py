"""Write a notebook that redoes the analysis rather than describing it.

The point is independence. A notebook that pastes the app's numbers into
markdown proves nothing: if the app is wrong, the notebook is wrong in the same
way and agrees with itself. So every figure here is recomputed by calling the
same lane against the CSV, and the last cell asserts the recomputed estimate
matches what the app reported. Run it and you have checked the tool, not read
its opinion.

That assert is the whole design. If it ever fails, either the app or the
notebook is broken, and you find out by running it rather than by trusting it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .design import ASSUMPTION


def _fmt(value: Any) -> str:
    return json.dumps(value, indent=None, default=str)


def build(
    *,
    csv_path: str,
    question: str,
    context: str,
    lane: str,
    kwargs: dict,
    estimate: dict,
    strength: str,
    roles: dict | None = None,
    source: str = "",
) -> nbformat.NotebookNode:
    """A notebook that reproduces one analysis end to end."""
    roles = roles or {}
    kw = {k: v for k, v in kwargs.items() if not str(k).startswith("_")}
    value = estimate.get("value")

    cells = [
        new_markdown_cell(
            f"# {question}\n\n"
            f"Design: **{lane}** · source: `{source or Path(csv_path).name}`\n\n"
            f"> {ASSUMPTION.get(lane, 'assumption unstated')}\n\n"
            "Everything below is recomputed from the data. Nothing is pasted "
            "from the application, so running this notebook checks the result "
            "rather than restating it."
        ),
        new_code_cell(
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import pandas as pd\n"
            "from causal import lanes\n"
            "from causal.profile import profile\n\n"
            f"df = pd.read_csv({Path(csv_path).as_posix()!r})\n"
            "print(f'{len(df):,} rows, {len(df.columns)} columns')"
        ),
        new_markdown_cell(
            "## What the person analysing this said about the data\n\n"
            + (f"```\n{context.strip()}\n```" if context.strip()
               else "_no context was given_")
        ),
        new_markdown_cell("## The columns"),
        new_code_cell(
            "p = profile(df)\n"
            "pd.DataFrame([c.__dict__ for c in p.columns])[\n"
            "    ['name', 'dtype', 'n_unique', 'missing', 'numeric', 'binary', 'datelike']\n"
            "]"
        ),
    ]

    # Only claim an adjustment the estimator actually made. Reporting the role
    # reasoning as "adjusted for" would be false on lanes that take no
    # covariates at all, such as difference in differences.
    used = list(kw.get("covariates") or [])
    if roles and used:
        dropped = {
            n: j.get("role")
            for n, j in roles.items()
            if j.get("role") in ("mediator", "collider", "proxy_for_outcome")
        }
        cells += [
            new_markdown_cell(
                "## Which columns were adjusted for, and which were not\n\n"
                "Adjusting for a mediator removes part of the effect being "
                "measured. Adjusting for a collider invents one. Both look like "
                "ordinary correlations, so the split below is a judgement about "
                "how the world is wired, not something read off the data."
            ),
            new_code_cell(
                f"adjusted_for = {used!r}   # passed to the estimator\n"
                f"left_out = {dropped!r}   # column -> why it is not a control\n"
                "print('adjusted for:', adjusted_for)\n"
                "print('left out    :', left_out)"
            ),
        ]
    elif roles:
        cells.append(new_markdown_cell(
            f"## Covariates\n\nThe **{lane}** design takes no covariates, so "
            "nothing was adjusted for. Identification comes from the design "
            "itself, not from a control set."
        ))

    cells += [
        new_markdown_cell(
            f"## Recompute the {lane} estimate\n\n"
            "This calls the same function the application called, with the same "
            "arguments, against the same file."
        ),
        new_code_cell(
            f"kwargs = {_fmt(kw)}\n"
            "kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}\n"
            f"result = lanes.{lane}(df, **kwargs)\n"
            "print(result)\n"
            "for note in result.notes:\n"
            "    print(' -', note)"
        ),
        new_markdown_cell(
            "## Does it match what the application reported?\n\n"
            "If this cell raises, one of the two is wrong and you have just "
            "found out by running it."
        ),
        new_code_cell(
            f"reported = {value!r}\n"
            "assert abs(result.value - reported) < 1e-9, (\n"
            "    f'notebook got {result.value}, the app reported {reported}'\n"
            ")\n"
            "print(f'match: {result.value}')"
        ),
        new_markdown_cell(
            f"## How much weight this carries\n\n"
            f"Claim strength: **{strength}**\n\n"
            "Strength is set by the design, not by the size of the number or "
            "its p-value. An interval covering the null lowers it; nothing "
            "raises it above what the design can support."
        ),
        new_code_cell(
            "lo, hi = result.ci_low, result.ci_high\n"
            "print(f'estimand : {result.estimand}')\n"
            "print(f'estimate : {result.value:.6g}')\n"
            "print(f'interval : {lo:.6g} to {hi:.6g}' if lo is not None else 'interval : n/a')\n"
            f"print('strength : {strength}')"
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    return nb


def to_json(nb: nbformat.NotebookNode) -> str:
    return nbformat.writes(nb, version=4)
