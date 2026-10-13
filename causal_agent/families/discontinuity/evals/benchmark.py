"""Reproducible declared study accounts for real-model endpoint acceptance.

These generated tables are labelled evaluations, not empirical causal evidence. No scripted model is used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_agent.common import trace
from causal_agent.common.contracts import Belief, Candidate, ColumnBrief, ColumnFacts, Fact, Handoff, Scope
from causal_agent.families.discontinuity.design import RdDesign
from causal_agent.families.discontinuity.lane.knowledge import CATALOGUE
from causal_agent.lane.endpoints import admit


def generate():
    folder = Path(__file__).parent / "handoffs"
    for fuzzy in (False, True):
        name = "window_fuzzy" if fuzzy else "window_sharp"
        rng = np.random.default_rng(72)
        x = np.concatenate([-np.linspace(0.0025, 1.0, 400), np.linspace(0.0025, 1.0, 400)])
        receipt = ((x >= 0) & (rng.random(len(x)) < 0.65)) if fuzzy else x >= 0
        frame = pd.DataFrame({"score": x, "receipt": receipt.astype(int), "outcome": 2 + 2 * receipt + rng.normal(size=len(x))})
        csv = folder / f"{name}.csv"
        trace.write(csv, frame.to_csv(index=False))
        values = {
            "target.locality": "window" if fuzzy else "cutoff",
            "target.comparison": "receipt",
            "assignment.kind": "cutoff_rule",
            "assignment.score_column": "score",
            "assignment.cutoff": 0.0,
            "assignment.treated_side": "above",
            "assignment.cutoff_value_treated": True,
            "assignment.movable": False,
            "assignment.compliance": "fuzzy" if fuzzy else "sharp",
            "assignment.treatment_column": "receipt",
            "assignment.treated_level": "1",
            "cutoff_only.believed": True,
            "cutoff_only.why": "The declared generating process changes only eligibility and receipt at the threshold; there is no concurrent intervention.",
            "exclusion.why": "The generating equation changes the outcome solely through actual receipt; eligibility has no direct term.",
            "monotonicity.why": "Only units above the cutoff can receive treatment, and no unit loses receipt when becoming eligible.",
            "local_randomisation.why": "Potential-outcome errors are independent identically distributed draws independent of score, with constant effect two; units inside a score-only window are exchangeable under the declared complete-randomisation analysis premise.",
            "exclusion.believed": True,
            "monotonicity.believed": True,
            "local_randomisation.believed": True,
            "grain.row_is": "one independently generated evaluation unit",
            "sampling.how": "random sample",
        }
        facts = {
            f"claim:{a}": Fact(
                value=v,
                status="confirmed",
                source="eval:declared_assignment_account",
                said="Scores are fixed on a balanced symmetric support grid before assignment; potential-outcome noise is independent of score; a strict mechanical threshold establishes eligibility. The stated randomisation, exclusion and monotonicity assumptions are evaluation premises.",
            )
            for a, v in values.items()
        }
        facts["col:score.when"] = Fact(value="before", status="confirmed", source="eval:declared_assignment_account")
        columns = [
            ColumnBrief(
                name=k,
                key=k,
                when="after" if k == "outcome" else "before",
                meaning="assignment score fixed on a symmetric grid before the intervention"
                if k == "score"
                else "actual receipt"
                if k == "receipt"
                else "post-assignment numeric outcome",
                facts=ColumnFacts(kind="numeric", distinct=frame[k].nunique()),
            )
            for k in frame
        ]
        beliefs = {
            k: Belief(kind=k, value=True, status="confirmed", source="eval:declared_assignment_account")
            for k in ("cutoff_only", "exclusion", "monotonicity", "local_randomisation")
        }
        h = Handoff(
            family="discontinuity",
            specialist="rdrobust",
            supported_now=True,
            outcome="outcome",
            treatment="receipt",
            scope=Scope(),
            pack_name=name,
            csv=str(csv.resolve()),
            question="Did actual receipt of the intervention change the outcome inside a score window?"
            if fuzzy
            else "Did assignment to the intervention change the outcome at the cutoff?",
            chosen_assumption="the declared evaluation study assumptions",
            reasons=[],
            relevant_columns=[Candidate(column=k, reason="the assignment score, receipt and outcome", cites=[f"col:{k}"]) for k in frame],
            columns=columns,
            facts=facts,
            beliefs=beliefs,
            assignment={k.partition(".")[2]: v for k, v in values.items() if k.startswith("assignment.")},
            design=RdDesign(
                score="score",
                cutoff=0.0,
                treated_side="above",
                cutoff_value_treated=True,
                score_fixed_before=True,
                movable=False,
                takeup={"column": "receipt", "level": "1"},
                **beliefs,
            ),
        )
        h = admit(h, CATALOGUE)
        if not h.admission.ready:
            raise ValueError(h.admission.model_dump_json())
        trace.write(folder / f"{name}.json", h.model_dump_json(indent=2))


if __name__ == "__main__":
    generate()
