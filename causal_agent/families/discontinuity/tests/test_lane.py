"""The discontinuity lane with a fake model and real rdrobust and rddensity, on the real Uruguay file and on
synthetic cutoff data with a known jump. No Vertex calls."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Cited, Handoff, Scope
from causal_agent.common.llm import set_llm
from causal_agent.desk.handoff import forced
from causal_agent.families.discontinuity.lane import nodes as N
from causal_agent.families.discontinuity.lane.contracts import (
    CovariateRelation,
    CovariateRoles,
    DesignAssessment,
    EstimatorPick,
    Line,
    RDInterpretation,
    Risk,
    Score,
    WindowPick,
)
from causal_agent.families.discontinuity.lane.graph import compile_local
from causal_agent.lane.ladder import Heterogeneity, Modifier
from causal_agent.memory import store
from causal_agent.profile import datasets as DS
from causal_agent.profile.profiler import profile


def memory(pack: str, *, cutoff_only=True, cutoff_only_status="confirmed", said=None):
    """The claims file alone (a note mined on disk never moves a test), plus the belief the family asks for: by default the
    person says nothing else switches at the cutoff."""
    m = store.migrate(pack, write=False)
    m.set("claim:cutoff_only.believed", cutoff_only, status=cutoff_only_status, source="user:turn:1", said=said)
    return m


def handoff(pack: str, outcome: str, treatment: str | None, cols: list[str], cite: str, target: str = "average", memory_=None) -> Handoff:
    return forced(
        pack,
        "Did crossing the cutoff change the outcome?",
        "discontinuity",
        outcome,
        treatment,
        cols,
        scope=Scope(target=target),
        assumption="units just either side of the cutoff are alike",
        cite=cite,
        memory=memory_ or memory(pack),
    )


URUGUAY = dict(
    pack="gov_transfers",
    outcome="Support",
    treatment="Participation",
    cols=["Support", "Participation", "Income_Centered", "Education", "Age"],
    cite="col:income_centered.note",
)
URUGUAY_SCORE = Score(
    column="income_centered",
    cutoff=0.0,
    treated_side="below",
    cutoff_value_treated=False,
    takeup_column="participation",
    takeup_level="1",
    reason="the note says households below zero were eligible and every one of them received the transfer",
    cites=["col:income_centered.note", "change:1.note"],
)
URUGUAY_RELATIONS = {"education": dict(predetermined=True), "age": dict(predetermined=True)}


# ------------------------------------------------------------------ the fake model


class FakeLLM:
    """Scripted answers: a Score, relations per column, and defaults for the rest that read the material like a careful model would.
    `overrides` sets a column's claims in the covariates rung; `risks` are what the line rung names; `looks` scripts the tool calls
    of an episode by node name."""

    def __init__(
        self,
        score: Score,
        relations: dict,
        cite: str,
        *,
        bad_cites=False,
        assess_script=None,
        pick_script=None,
        score_script=None,
        interpret_bad_first=False,
        overrides=None,
        risks=None,
        looks=None,
        line_script=None,
        ignore_balance=False,
        window_script=None,
    ):
        self.score, self.relations, self.cite, self.bad_cites = score, relations, cite, bad_cites
        self.assess_script, self.pick_script, self.score_script = list(assess_script or []), list(pick_script or []), list(score_script or [])
        self.interpret_bad_first = interpret_bad_first
        self.overrides = dict(overrides or {})
        self.risks = list(risks or [])
        self.line_script = list(line_script or [])
        self.ignore_balance = ignore_balance  # a careless first answer, to see the gate re-prompt
        self.window_script = list(window_script or [])
        self.looks = {k: list(v) for k, v in (looks or {}).items()}
        self.calls: list[str] = []
        self.humans: list[tuple[str, str]] = []

    def humans_of(self, name: str) -> list[str]:
        return [h for n, h in self.humans if n == name]

    def asked(self, name: str) -> list[str]:
        return [c for h in self.humans_of(name) for c in re.findall(r"^COLUMN '([^']+)'", h, re.M)]

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        m = re.search(r"\[probe:([a-z_]+)\.<n>\]", messages[0].content)
        node = m.group(1) if m else ""
        rounds = self.looks.get(node) or []
        calls = rounds.pop(0) if rounds else []
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"{node}{i}"} for i, (n, a) in enumerate(calls)])

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                parsed = fake.answer(schema, messages[-1][1])
                raw = AIMessage(
                    content=[{"type": "thinking", "thinking": f"thinking about {schema.__name__}"}, "{}"],
                    usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "output_token_details": {"reasoning": 7}},
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return R()

    def covariates_answer(self, human: str) -> CovariateRoles:
        cite = "col:nope.note" if self.bad_cites else self.cite
        # a careful model reads the balance rung: a column it calls fixed before that differs at the line gets the balance line cited;
        # a careless one waits for the gate to say so
        must_cite = set(re.findall(r"^- (\S+) jumps at the line \[ladder:balance\.[^\]]+\]", human, re.M))
        if not self.ignore_balance:
            must_cite |= set(re.findall(r"^\[ladder:balance\.([^\]]+)\] .*; differs at the line$", human, re.M))
        items = []
        for col in re.findall(r"^COLUMN '([^']+)'", human.split("THE COLUMNS TO PLACE")[1], re.M):
            flags = dict(predetermined=False, affected_by_treatment=False, is_outcome_measure=False, modifier_candidate=False)
            flags.update(self.relations.get(col, {}))
            flags.update(self.overrides.get(col, {}))
            cites = [cite] + ([f"ladder:balance.{col}"] if col in must_cite else [])
            reasons = [Cited(reason=f"{col}: {k}", cites=cites) for k in ("predetermined", "affected_by_treatment", "is_outcome_measure") if flags[k]]
            items.append(CovariateRelation(column=col, reasons=reasons, **flags))
        return CovariateRoles(items=items)

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        self.humans.append((schema.__name__, human))
        cite = "col:nope.note" if self.bad_cites else self.cite
        if schema is Score:
            s = (self.score_script.pop(0) if self.score_script else self.score).model_copy()
            s.cites = [cite]
            return s
        if schema is Line:
            if self.line_script:
                return self.line_script.pop(0)
            # a careful model reads the density rung: it cites the test when the rows bunch, and names manipulation when told the person said the score could be moved
            risks, cites = list(self.risks), [cite]
            if "[ladder:density.test]" in human and "the rows bunch" in human:
                cites.append("ladder:density.test")
            if "name the manipulation risk" in human and not any(r.name == "manipulation" for r in risks):
                risks.append(
                    Risk(
                        name="manipulation",
                        reason="units bunch just on the treated side and the person says the score could be moved",
                        cites=["ladder:density.test"],
                    )
                )
            return Line(
                clean=not risks,
                why="the story says the score was set before the programme and nothing else switches there",
                risks=risks,
                cites=cites,
            )
        if schema is CovariateRoles:
            return self.covariates_answer(human)
        if schema is WindowPick:
            if self.window_script:
                return self.window_script.pop(0)
            default = re.search(r"THE WIDTHS ON OFFER \(default: (\w+)\)", human).group(1)
            return WindowPick(selector=default, why="nothing in the ladder argues for another width", cites=[self.cite])
        if schema is Heterogeneity:
            cands = re.findall(r"^\[col:([^\]]+)\]", human.split("CANDIDATES")[1], re.M)
            return Heterogeneity(
                modifiers=[Modifier(column=cands[0], reason="the story says the effect could differ by it", cites=[cite])],
                why="one trait the story backs",
                cites=[cite],
            )
        if schema is DesignAssessment:
            if self.assess_script:
                return self.assess_script.pop(0)
            flags = re.findall(r"\[(check:[^\]]+)\] (\w+):", human.split("FLAGGED CHECKS")[1].split("SCORE CARD")[0])
            if any(level == "HARD" for _, level in flags):
                return DesignAssessment(action="stop", reason="a hard flag stands", cites=[a for a, _ in flags])
            return DesignAssessment(
                action="proceed",
                reason="the note says the score was set before the programme and could not be moved",
                cites=[a for a, _ in flags] + [self.cite],
            )
        if schema is EstimatorPick:
            names = [n.strip() for n in re.search(r"NAMES YOU MAY PICK: (.*)", human).group(1).split(",")]
            return EstimatorPick(name=self.pick_script.pop(0) if self.pick_script else names[0], reason="ranked first", cites=[])
        if schema is RDInterpretation:
            c = re.search(r"COMPARISON: (\S+)", human).group(1)
            m = re.search(r"\[estimate:%s\.value\] ([-\d.eE+]+) \(primary: \w+, (\w+)\)" % re.escape(c), human)
            value, estimand = float(m.group(1)), m.group(2)
            lo, hi = (float(v) for v in re.search(r"\[estimate:%s\.ci\] 95%% robust interval ([-\d.eE+]+) to ([-\d.eE+]+)" % re.escape(c), human).groups())
            nl, nr = (int(v) for v in re.search(r"\[estimate:%s\.n\] (\d+) control-side and (\d+) treated-side" % re.escape(c), human).groups())
            hl, hr = (
                float(v)
                for v in re.search(
                    r"\[estimate:%s\.bandwidth\] h = ([-\d.eE+]+) on the control side, ([-\d.eE+]+) on the treated side" % re.escape(c), human
                ).groups()
            )
            required = [a for a in human.split("ADDRESSES YOU MUST CITE")[1].split("\n\n")[0].strip().splitlines()[1:] if a and a != "(none)"]
            bad = self.interpret_bad_first and "PREVIOUS ANSWER WAS REJECTED" not in human
            return RDInterpretation(
                contrast=c,
                answer=f"At the cutoff the effect is {value:.3g}, interval {lo:.3g} to {hi:.3g}; local to units at the cutoff.",
                effect_stated=value,
                caveats=["local to the cutoff"],
                cites=["nope:x"] if bad else required,
                estimand=estimand,
                bandwidth_left_stated=hl,
                bandwidth_right_stated=hr,
                n_left_stated=nl,
                n_right_stated=nr,
                ci_low_stated=lo,
                ci_high_stated=hi,
            )
        raise AssertionError(schema)


@pytest.fixture(autouse=True)
def _restore(tmp_path, monkeypatch):
    yield
    set_llm(None)


def _run(fake, h, question="Did crossing the cutoff change the outcome?"):
    set_llm(fake)
    g = compile_local()
    return g.invoke({"question": question, "handoff": h, "dataset": h.pack_name}, {"configurable": {"thread_id": str(uuid.uuid4())}})


# ------------------------------------------------------------------ synthetic packs


NOTE = """# {name}

## About the dataset
Each row is one unit. Synthetic data with a known jump. [synthetic]

## What changed
**A grant.** {rule} [synthetic]

## About each column
{columns}
"""


def make_pack(
    tmp_path, monkeypatch, name: str, df: pd.DataFrame, rule: str, columns: dict[str, str], *, entity: list[str] | None = None, sampled_by_side=False
):
    csv, md, prof = tmp_path / f"{name}.csv", tmp_path / f"{name}.md", tmp_path / f"{name}.json"
    df.to_csv(csv, index=False)
    md.write_text(NOTE.format(name=name, rule=rule, columns="\n\n".join(f"**{k}** — {v} [synthetic]" for k, v in columns.items())))
    prof.write_text(profile(csv, entity_columns=entity).model_dump_json())
    entry = {"csv": str(csv), "note": str(md), "profile": str(prof), "sampled_by_side": sampled_by_side}
    if entity:
        entry["entity"] = entity
    monkeypatch.setattr(N, "dataset_entries", lambda: {name: entry})
    monkeypatch.setattr(DS, "dataset_entries", lambda *a, **k: {name: entry})  # the builder and the memory store read the index and the root
    monkeypatch.setattr(DS, "ROOT", tmp_path)


def sharp_below(n=3000, jump=1.0, seed=1) -> pd.DataFrame:
    """Treated strictly below 50 on a score rounded to halves, so many units tie and some sit exactly on 50."""
    rng = np.random.default_rng(seed)
    score = np.round(rng.uniform(0, 100, n) * 2) / 2
    treated = (score < 50).astype(int)
    z = rng.normal(0, 1, n)
    y = 10 + 0.05 * score + jump * treated + 0.3 * z + rng.normal(0, 0.5, n)
    return pd.DataFrame({"score": score, "y": y, "got": treated, "z": z, "later": y + rng.normal(0, 0.1, n)})


def fuzzy_above(n=4000, takeup=0.7, effect=2.0, seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    eligible = (x >= 0).astype(int)
    d = (eligible * (rng.uniform(0, 1, n) < takeup)).astype(int)
    y = 1 + x + effect * d + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y, "eligible": eligible, "received": d})


def weak_first_stage(n=4000, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    eligible = (x >= 0).astype(int)
    d = (rng.uniform(0, 1, n) < 0.30 + 0.06 * eligible).astype(int)
    y = 1 + x + 0.5 * d + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y, "eligible": eligible, "received": d})


def no_first_stage(n=4000, seed=4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    d = (rng.uniform(0, 1, n) < 0.3).astype(int)
    y = 1 + x + 0.5 * d + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y, "received": d})


def discrete(n=3000, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 13, n)
    y = 0.2 * x + 1.0 * (x >= 7) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y})


def manipulated(n=6000, seed=6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    band = np.where((x > -0.3) & (x < 0))[0]
    move = rng.choice(band, int(0.6 * len(band)), replace=False)
    x[move] = -x[move]
    y = x + 0.5 * (x >= 0) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y})


SYNTH_COLS = {
    "score": "The score the rule was applied to, fixed before the grant.",
    "y": "The outcome, measured after the grant.",
    "got": "1 if the unit received the grant, 0 if not.",
    "z": "A characteristic fixed before the grant.",
    "later": "The outcome measured again a year later.",
}
FUZZY_COLS = {
    "x": "The score, centred on the cutoff; fixed before the offer.",
    "y": "The outcome, measured after.",
    "eligible": "1 if the score was at or above zero.",
    "received": "1 if the unit actually took the grant up.",
}


# ------------------------------------------------------------------ tests: the real file


def test_uruguay_happy_path():
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    s = out["shape"]
    assert s.kind == "sharp" and s.n_left == 821 and s.n_right == 1127 and s.rows_primary == 1948 and s.rows_at_cutoff == 0 and s.cutoff_shift == 0
    assert s.takeup_left == 0.0 and s.takeup_right == 1.0
    d = out["design"]
    assert d.estimator == "local_linear" and d.estimand == "effect_at_cutoff" and d.inference == "robust_bc"
    assert set(d.covariates.balance_tested) == {"education", "age"} and d.covariates.adjusted == d.covariates.balance_tested
    levels = {c.name: c.level for c in out["checks"]}
    assert levels["mass_points"] == "soft" and levels["support"] == "pass" and levels["compliance"] == "pass" and levels["first_stage"] == "pass"
    assert "covariate_continuity" in levels and "density" in levels
    primary = next(e for e in out["estimates"] if e.method == "local_linear")
    assert abs(primary.value - (-0.025)) < 0.01 and primary.ci_low < 0 < primary.ci_high
    assert primary.n_control == 194 and primary.n_treated == 291
    assert {e.method for e in out["estimates"]} == {"local_linear", "local_quadratic", "local_linear_adjusted"}  # a sharp design has no first-stage fit
    refs = {x.refuter: x for x in out["refutations"]}
    assert set(refs) == {"placebo_cutoffs", "bandwidth_grid", "donut", "polynomial_grid", "kernel_grid"}
    assert all(refs[n].kind == "falsification" and refs[n].passed is not None for n in ("placebo_cutoffs", "bandwidth_grid", "donut"))
    for n in ("polynomial_grid", "kernel_grid"):  # a sensitivity: a range around the primary, no verdict
        assert refs[n].kind == "sensitivity" and refs[n].passed is None and refs[n].range_low <= primary.value <= refs[n].range_high
    assert "p = 1:" in refs["polynomial_grid"].detail and "p = 3:" in refs["polynomial_grid"].detail and "kernel epa:" in refs["kernel_grid"].detail
    human = fake.humans_of("RDInterpretation")[0]
    assert "[placebo:below_cutoff_vs_above_cutoff.kernel_grid.in_words] sensitivity:" in human and "(source: Foundations 4.2" in human
    assert len(out["interpretations"]) == 1 and not out.get("interpret_errors")
    assert out["interpretations"][0].estimand == "effect_at_cutoff"
    assert "DESIGN" in r["report"] and "ANSWER" in r["report"]
    assert fake.calls.count("CovariateRoles") == 1 and sorted(fake.asked("CovariateRoles")) == ["age", "education"]
    assert fake.calls.count("Score") == 1 and fake.calls.count("Line") == 1 and "DesignAssessment" in fake.calls
    for name in ("Line", "CovariateRoles", "DesignAssessment", "EstimatorPick", "RDInterpretation"):  # every judgement sees the case
        assert fake.humans_of(name) and all("THE CASE" in p and "[change:1.note]" in p for p in fake.humans_of(name)), name
    lad = out["ladder"]
    assert lad.score.by == "judgement" and lad.line.clean and lad.heterogeneity.by == "code" and lad.window is not None and lad.threats is not None
    assert lad.window.by == "judgement" and lad.window.selector == "mserd" and not lad.window.two_sided() and "WindowPick" in fake.calls
    assert "[ladder:score.rule] income_centered treated when below 0" in r["report"] and "[ladder:covariates.age] fixed before the line" in r["report"]
    assert "THE LADDER SO FAR" in fake.humans_of("Line")[0] and "[ladder:shape.kind] sharp" in fake.humans_of("Line")[0]
    assert ["ladder:line.clean", "yes"] in r["ladder"] and "ladder:window.h" in fake.humans_of("RDInterpretation")[0].split("ADDRESSES YOU MAY CITE")[1]
    assert next(x for x in out["checks"] if x.name == "effective_rows").detail.startswith("194 control-side and 291 treated-side rows inside the window mserd")
    run = Path(r["run_dir"])
    assert (run / "design.json").exists() and (run / "bins.csv").exists() and (run / "canon.csv").exists()


def test_uruguay_assess_stop_is_honest():
    script = [DesignAssessment(action="stop", reason="both predetermined covariates jump at the cutoff", cites=[])]
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"], assess_script=script)
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "assess"
    assert out.get("design") is None and not out.get("estimates")


def test_proceed_without_citing_flags_is_rejected():
    script = [DesignAssessment(action="proceed", reason="fine", cites=[])] * 3
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"], assess_script=script)
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "assess"
    assert fake.calls.count("DesignAssessment") == 3
    assert any("does not address" in x for x in out["feasibility"].facts)


def test_proceed_on_density_without_a_note_cite_is_rejected(tmp_path, monkeypatch):
    df = manipulated()
    make_pack(
        tmp_path,
        monkeypatch,
        "manip",
        df,
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    sc = Score(column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    # cites the flag addresses but no pack address, three times
    fake = FakeLLM(sc, {}, "col:x.note", assess_script=[DesignAssessment(action="proceed", reason="fine", cites=["check:above_vs_below.density"])] * 3)
    m = memory("manip")
    m.set(
        "claim:assignment.movable", False, status="confirmed", source="user:turn:1"
    )  # the person says the score could not be moved: the jump is the model's to argue
    out = _run(fake, handoff("manip", "y", None, ["x", "y"], "col:x.note", memory_=m))
    levels = {c.name: c.level for c in out["checks"]}
    assert levels["density"] == "soft", [c.detail for c in out["checks"] if c.name == "density"]
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "assess"
    assert any("no note cited" in x for x in out["feasibility"].facts)


# ------------------------------------------------------------------ tests: the score gates


def test_score_gates_reject_then_stop():
    bad = [
        URUGUAY_SCORE.model_copy(update={"cutoff": 5.0}),
        URUGUAY_SCORE.model_copy(update={"takeup_level": "yes"}),
        URUGUAY_SCORE.model_copy(update={"column": "nope"}),
    ]
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"], score_script=bad)
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "score"
    facts = " ".join(out["feasibility"].facts)
    assert fake.calls.count("Score") == 3 and "not a column in the file" in facts


def test_wrong_side_stops_on_the_shares():
    fake = FakeLLM(URUGUAY_SCORE.model_copy(update={"treated_side": "above"}), URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "score"
    assert "disagree" in out["feasibility"].reason and fake.calls.count("Score") == 1


def test_no_cutoff_stated_is_an_honest_stop():
    fake = FakeLLM(Score(column=None, reason="no note states a cutoff", cites=[]), {}, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "score"
    assert "no cutoff" in out["feasibility"].reason


def test_treatment_column_must_be_the_takeup(tmp_path, monkeypatch):
    # the hand-off names `received`, which is not a function of the cutoff; a Score that ignores it is rejected three times
    make_pack(
        tmp_path, monkeypatch, "fuzzy", fuzzy_above(), "Units with a score at or above zero were offered the grant; about seven in ten took it up.", FUZZY_COLS
    )
    sc = Score(column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    fake = FakeLLM(sc, {}, "col:x.note")
    out = _run(fake, handoff("fuzzy", "y", "received", ["x", "y", "eligible", "received"], "col:x.note"))
    assert out["feasibility"].stage == "score" and fake.calls.count("Score") == 3
    assert any("take-up column" in x for x in out["feasibility"].facts)


def test_takeup_null_is_fine_when_the_treatment_is_the_rule_itself():
    # Uruguay's Participation equals the side of the cutoff exactly, so the model may treat the change as the rule itself
    fake = FakeLLM(URUGUAY_SCORE.model_copy(update={"takeup_column": None, "takeup_level": None}), URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "done" and out["shape"].kind == "sharp" and out["shape"].takeup_left is None


def test_takeup_named_as_the_score_itself_is_treated_as_the_rule():
    # a model that names the score column as the take-up column with a level like ">=0" is describing the rule, not a receipt column
    sc = URUGUAY_SCORE.model_copy(update={"takeup_column": "income_centered", "takeup_level": "<0"})
    fake = FakeLLM(sc, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**{**URUGUAY, "treatment": None}))
    assert out["specialist_result"]["status"] == "done" and out["score"].takeup_column is None and out["shape"].kind == "sharp"


# ------------------------------------------------------------------ tests: synthetic geometries


def test_sharp_treated_below_with_ties_at_the_cutoff(tmp_path, monkeypatch):
    df = sharp_below()
    assert (df["score"] == 50).sum() > 0
    make_pack(tmp_path, monkeypatch, "sharp", df, "Units with a score strictly below 50 got the grant; a unit exactly at 50 did not.", SYNTH_COLS)
    sc = Score(
        column="score",
        cutoff=50.0,
        treated_side="below",
        cutoff_value_treated=False,
        takeup_column="got",
        takeup_level="1",
        reason="r",
        cites=["col:score.note"],
    )
    fake = FakeLLM(sc, {"z": dict(predetermined=True), "later": dict(is_outcome_measure=True)}, "col:score.note")
    out = _run(fake, handoff("sharp", "y", "got", ["score", "y", "got", "z", "later"], "col:score.note"))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    s = out["shape"]
    assert s.rows_at_cutoff > 0 and s.cutoff_shift < 0 and abs(s.cutoff_shift) == 0.25 and s.kind == "sharp"
    assert s.takeup_right == 1.0 and s.takeup_left == 0.0
    d = out["design"]
    assert d.covariates.balance_tested == ["z"] and {x.column for x in d.covariates.excluded} == {"later"}
    primary = next(e for e in out["estimates"] if e.method == "local_linear")
    assert primary.ci_low <= 1.0 <= primary.ci_high, (primary.value, primary.ci_low, primary.ci_high)
    grid = next(x for x in out["refutations"] if x.refuter == "bandwidth_grid")
    assert grid.passed is True, grid.detail
    cuts = next(x for x in out["refutations"] if x.refuter == "placebo_cutoffs")
    assert cuts.passed in (True, None), cuts.detail
    levels = {c.name: c.level for c in out["checks"]}
    assert levels["covariate_continuity"] == "pass" and levels["mass_points"] == "soft"


def test_fuzzy_with_a_strong_first_stage(tmp_path, monkeypatch):
    make_pack(
        tmp_path, monkeypatch, "fuzzy", fuzzy_above(), "Units with a score at or above zero were offered the grant; about seven in ten took it up.", FUZZY_COLS
    )
    sc = Score(
        column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column="received", takeup_level="1", reason="r", cites=["col:x.note"]
    )
    fake = FakeLLM(sc, {}, "col:x.note")
    seen: list[dict] = []
    real_fit = N.adapter.fit
    monkeypatch.setattr(N.adapter, "fit", lambda *a, **k: (seen.append(k), real_fit(*a, **k))[1])
    out = _run(fake, handoff("fuzzy", "y", "eligible", ["x", "y", "eligible", "received"], "col:x.note"))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    s = out["shape"]
    assert s.kind == "fuzzy" and s.takeup_left == 0.0 and 0.6 < s.takeup_right < 0.8
    assert any(k.get("fuzzy") and k.get("sharpbw") for k in seen) and not any(
        k.get("sharpbw") and not k.get("fuzzy") for k in seen
    )  # passed to the library, for fuzzy fits only
    assert next(c for c in out["checks"] if c.name == "one_sided_takeup").level == "pass"
    assert out["check_facts"]["first_stage_status"] == "strong"
    d = out["design"]
    assert d.estimator == "local_linear_fuzzy" and d.estimand == "complier_effect_at_cutoff" and d.sharp_bandwidth_used is True
    assert set(d.also_run) == {"local_linear_itt"}
    primary = next(e for e in out["estimates"] if e.method == "local_linear_fuzzy")
    assert primary.ci_low <= 2.0 <= primary.ci_high, (primary.value, primary.ci_low, primary.ci_high)
    itt = next(e for e in out["estimates"] if e.method == "local_linear_itt")
    assert 1.0 < itt.value < 1.8
    fs = next(e for e in out["estimates"] if e.method == "first_stage")
    assert 0.6 < fs.value < 0.8
    cuts = next(x for x in out["refutations"] if x.refuter == "placebo_cutoffs")
    assert cuts.passed is not False, cuts.detail  # the sharp reduced form runs on both sides, even where take-up never varies
    assert out["interpretations"][0].estimand == "complier_effect_at_cutoff" and not out.get("interpret_errors")


def test_weak_first_stage_leaves_only_itt(tmp_path, monkeypatch):
    make_pack(tmp_path, monkeypatch, "weak", weak_first_stage(), "Units with a score at or above zero were offered the grant; few took it up.", FUZZY_COLS)
    sc = Score(
        column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column="received", takeup_level="1", reason="r", cites=["col:x.note"]
    )
    fake = FakeLLM(sc, {}, "col:x.note")
    out = _run(fake, handoff("weak", "y", "eligible", ["x", "y", "eligible", "received"], "col:x.note"))
    status = out["check_facts"]["first_stage_status"]
    assert status in ("weak", "none"), status
    if status == "weak":
        assert out["specialist_result"]["status"] == "done" and out["design"].estimator == "local_linear_itt" and out["design"].estimand == "itt_at_cutoff"
    else:
        assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "assess"


def test_no_first_stage_is_a_hard_stop(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "none",
        no_first_stage(),
        "Units with a score at or above zero were offered the grant.",
        {k: v for k, v in FUZZY_COLS.items() if k != "eligible"},
    )
    sc = Score(
        column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column="received", takeup_level="1", reason="r", cites=["col:x.note"]
    )
    fake = FakeLLM(sc, {}, "col:x.note", assess_script=[DesignAssessment(action="proceed", reason="ignore", cites=["check:above_vs_below.no_first_stage"])] * 3)
    out = _run(fake, handoff("none", "y", "received", ["x", "y", "received"], "col:x.note"))
    levels = {c.name: c.level for c in out["checks"]}
    assert levels["no_first_stage"] == "hard"
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "assess" and fake.calls.count("DesignAssessment") == 3


def test_discrete_score_flags_support_and_density(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "disc",
        discrete(),
        "Units with a score at or above 7 got the grant.",
        {"x": "The score, an integer from 1 to 12, fixed before the grant.", "y": "The outcome, measured after."},
    )
    sc = Score(column="x", cutoff=7.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    fake = FakeLLM(
        sc, {}, "col:x.note", pick_script=["local_linear"]
    )  # the local polynomial under the support-points rule; local randomisation has its own test
    out = _run(fake, handoff("disc", "y", None, ["x", "y"], "col:x.note"))
    assert "NAMES YOU MAY PICK: local_randomisation, local_linear" in fake.humans_of("EstimatorPick")[0]
    levels = {c.name: c.level for c in out["checks"]}
    details = {c.name: c.detail for c in out["checks"]}
    assert levels["support"] == "soft" and levels["mass_points"] == "soft"
    assert levels["density"] == "soft" and "not computable" in details["density"]
    assert out["specialist_result"]["status"] == "done", out.get("feasibility")
    assert out["shape"].distinct_scores == 12
    d = out["design"]
    assert d.bandwidths.rule == "support_points" and d.bandwidths.h_cer_left is None and d.bandwidths.h_left == 3.5 == d.bandwidths.h_right
    assert out["ladder"].window.by == "code" and "WindowPick" not in fake.calls  # few distinct scores: nothing to judge
    primary = next(e for e in out["estimates"] if e.method == "local_linear")
    assert primary.ci_low <= 1.0 <= primary.ci_high, (primary.value, primary.ci_low, primary.ci_high)
    grid = next(x for x in out["refutations"] if x.refuter == "bandwidth_grid")
    assert "no coverage-error bandwidth" in grid.detail


def test_thin_sides_stop_before_the_library(tmp_path, monkeypatch):
    df = sharp_below(n=60, seed=7)
    df = pd.concat([df[df["score"] < 50].head(15), df[df["score"] >= 50].head(15)])
    make_pack(tmp_path, monkeypatch, "thin", df, "Units with a score strictly below 50 got the grant.", SYNTH_COLS)
    sc = Score(
        column="score",
        cutoff=50.0,
        treated_side="below",
        cutoff_value_treated=False,
        takeup_column="got",
        takeup_level="1",
        reason="r",
        cites=["col:score.note"],
    )
    fake = FakeLLM(sc, {}, "col:score.note")
    out = _run(fake, handoff("thin", "y", "got", ["score", "y", "got"], "col:score.note"))
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "shape_table"


def test_small_but_legal_sample_never_raises(tmp_path, monkeypatch):
    df = sharp_below(n=200, seed=8)
    df = pd.concat([df[df["score"] < 50].head(22), df[df["score"] >= 50].head(22)])
    make_pack(tmp_path, monkeypatch, "small", df, "Units with a score strictly below 50 got the grant.", SYNTH_COLS)
    sc = Score(
        column="score",
        cutoff=50.0,
        treated_side="below",
        cutoff_value_treated=False,
        takeup_column="got",
        takeup_level="1",
        reason="r",
        cites=["col:score.note"],
    )
    fake = FakeLLM(sc, {}, "col:score.note")
    out = _run(fake, handoff("small", "y", "got", ["score", "y", "got"], "col:score.note"))
    assert out["specialist_result"]["status"] in ("done", "infeasible")
    if out["specialist_result"]["status"] == "infeasible":
        assert out["feasibility"].stage in ("assess", "estimate", "freeze_design")


# ------------------------------------------------------------------ tests: the other gates


def test_bad_cites_in_the_covariates_rung_stop_it_after_three_tries():
    class Fake(FakeLLM):
        def answer(self, schema, human):
            self.bad_cites = schema is CovariateRoles
            return super().answer(schema, human)

    fake = Fake(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    assert out["feasibility"].stage == "covariates" and fake.calls.count("CovariateRoles") == 3 and out["episodes"]["covariates"].tries == 3
    assert "education: col:nope.note is not an address you may cite" in fake.humans_of("CovariateRoles")[1]


def test_estimator_outside_list_is_rejected_then_accepted():
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"], pick_script=["magic", "local_quadratic"])
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "done" and fake.calls.count("EstimatorPick") == 2 and out["design"].estimator == "local_quadratic"


def test_interpretation_bad_cite_is_retried():
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"], interpret_bad_first=True)
    out = _run(fake, handoff(**URUGUAY))
    assert out["specialist_result"]["status"] == "done" and fake.calls.count("RDInterpretation") == 2 and not out.get("interpret_errors")


def test_catalogues_and_thresholds():
    from causal_agent.families.discontinuity.lane import adapter
    from causal_agent.families.discontinuity.lane.knowledge import load_checks, load_estimators, load_inference, load_placebos

    toy = fuzzy_above(n=1500, seed=9).rename(columns={"received": "t"})
    toy["side"] = (toy["x"] >= 0).astype(int)
    for e in load_estimators():
        if e.engine != "local_polynomial":
            continue
        f = adapter.fit(e.params, toy, fuzzy=bool(e.fuzzy is True), covs=None)
        assert f.error is None, (e.name, f.error)
    assert [i.name for i in load_inference()] == ["cluster_few", "cluster_entity", "robust_bc"]
    assert next(i for i in load_inference() if i.applies(cluster_column=True, clusters=12)).name == "cluster_few"
    assert next(i for i in load_inference() if i.applies(cluster_column=True, clusters=40)).name == "cluster_entity"
    assert next(i for i in load_inference() if i.applies(cluster_column=False, clusters=None)).name == "robust_bc"
    assert {p.name for p in load_placebos()} == {
        "placebo_cutoffs",
        "bandwidth_grid",
        "donut",
        "polynomial_grid",
        "kernel_grid",
        "window_sensitivity",
        "rosenbaum_bounds",
    }
    assert {p.name for p in load_placebos() if p.applies(engine="local_polynomial", kind="sharp", rule="mse", distinct_scores=100)} == {
        "placebo_cutoffs",
        "bandwidth_grid",
        "donut",
        "polynomial_grid",
        "kernel_grid",
    }
    assert {p.name for p in load_placebos() if p.applies(engine="local_randomisation", kind="sharp", rule="local_randomisation", distinct_scores=12)} == {
        "window_sensitivity",
        "rosenbaum_bounds",
    }
    cfg = load_checks()
    assert cfg["sides"]["min_rows"]["hard"] < cfg["sides"]["min_rows"]["soft"]
    text = Path(__file__).resolve().parents[1].joinpath("lane", "knowledge", "checks.yaml").read_text().lower()
    for name in ("uruguay", "senate", "panes", "gov_transfers", "headstart", "spp", "probation", "students", "cigar", "krueger", "marketing"):
        assert name not in text, f"checks.yaml names a dataset: {name}"


def test_adapter_reproduces_the_senate_known_answer():
    from rdrobust import rdrobust_RDsenate

    from causal_agent.families.discontinuity.lane import adapter

    df = rdrobust_RDsenate()
    canon = pd.DataFrame({"y": df["vote"], "x": df["margin"], "cluster": df["state"].astype(str)})
    canon = canon[np.isfinite(canon["y"]) & np.isfinite(canon["x"])]
    canon["side"] = (canon["x"] >= 0).astype(int)
    f = adapter.fit({"p": 1, "kernel": "tri", "bwselect": "mserd", "masspoints": "adjust", "level": 95}, canon)
    assert abs(f.value - 7.414131) < 1e-4 and abs(f.ci_low - 4.093699) < 1e-4 and abs(f.ci_high - 10.919306) < 1e-4
    assert abs(f.h - 17.754397) < 1e-4 and abs(f.b - 28.028087) < 1e-4 and (f.n_left, f.n_right) == (595, 702)
    bw = adapter.bandwidths({"p": 1, "kernel": "tri", "bwselect": "mserd", "masspoints": "adjust", "level": 95}, canon)
    assert abs(bw["h_mse"] - f.h) < 1e-9 and 0 < bw["h_cer"] < bw["h_mse"]
    same = adapter.fit({"p": 1, "kernel": "tri", "bwselect": "mserd", "masspoints": "adjust", "level": 95}, canon, h=bw["h_mse"], b=bw["b_mse"])
    assert abs(same.value - f.value) < 1e-6 and abs(same.ci_low - f.ci_low) < 1e-6
    clustered = adapter.fit({"p": 1, "kernel": "tri", "bwselect": "mserd", "masspoints": "adjust", "level": 95}, canon, cluster=True)
    assert clustered.vce == "CR1" and clustered.error is None
    left = canon[canon["x"] < 0]
    pl = adapter.fit(adapter_sharp(), left, c=float(left["x"].median()))
    assert pl.error is None and pl.n_left > 0 and pl.n_right > 0
    d = adapter.density(canon["x"].to_numpy())
    assert d["computable"] and 0.3 < d["p"] < 0.5


def adapter_sharp() -> dict:
    return {"p": 1, "kernel": "tri", "bwselect": "mserd", "masspoints": "adjust", "level": 95}


def test_the_desk_reaches_three_specialists():
    from causal_agent.families.registry import lanes

    SPECIALISTS = lanes()

    assert "score" in SPECIALISTS["discontinuity"].get_graph().nodes
    assert "shape_table" in SPECIALISTS["diff_in_diff"].get_graph().nodes and "score" not in SPECIALISTS["diff_in_diff"].get_graph().nodes
    assert "relate" not in SPECIALISTS["synthetic_control"].get_graph().nodes  # still a stub


def test_raw_columns_named_like_canonical_ones_do_not_collide(tmp_path, monkeypatch):
    # a file whose eligibility column is called t and outcome y: the take-up column must survive the reshape
    df = fuzzy_above().rename(columns={"eligible": "t"})
    cols = {"x": FUZZY_COLS["x"], "y": FUZZY_COLS["y"], "t": "1 if the score was at or above zero.", "received": FUZZY_COLS["received"]}
    make_pack(tmp_path, monkeypatch, "collide", df, "Units with a score at or above zero were offered the grant; about seven in ten took it up.", cols)
    sc = Score(
        column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column="received", takeup_level="1", reason="r", cites=["col:x.note"]
    )
    fake = FakeLLM(sc, {"t": dict(affected_by_treatment=True)}, "col:x.note")
    out = _run(fake, handoff("collide", "y", "t", ["x", "y", "t", "received"], "col:x.note"))
    assert out["shape"].kind == "fuzzy" and 0.6 < out["shape"].takeup_right < 0.8
    assert out["specialist_result"]["status"] == "done" and out["design"].estimator == "local_linear_fuzzy"


# ------------------------------------------------------------------ the pack's facts end judgements


def test_pack_cutoff_block_settles_the_score_without_a_model_call():
    """senate3 carries claims: margin above zero decides who won, so the lane never asks the model for the score."""
    h = handoff("senate3", "vote", None, ["vote", "margin", "state", "year"], "col:margin.note")
    d = h.design
    assert (
        d.kind == "discontinuity"
        and d.score == "margin"
        and d.cutoff == 0.0
        and d.treated_side == "above"
        and d.cutoff_value_treated is True
        and d.takeup is None
    )
    assert h.treatment is None and h.assignment["kind"] == "cutoff_rule"
    fake = FakeLLM(URUGUAY_SCORE, {}, "col:margin.note")
    out = _run(fake, h)
    assert fake.calls.count("Score") == 0
    assert out["score"].column == "margin" and out["score"].cutoff == 0.0 and out["score"].treated_side == "above"
    assert out["score"].cites[0].startswith("claim:assignment.")


# ------------------------------------------------------------------ the lane on the harness: the person's beliefs meet the checks


def _rule(m, *, score="x", cutoff=0.0, side="above", movable=None):
    src = "user:turn:1"
    m.set("claim:assignment.kind", "cutoff_rule", status="confirmed", source=src)
    m.set("claim:assignment.score_column", score, status="confirmed", source=src)
    m.set("claim:assignment.cutoff", cutoff, status="confirmed", source=src)
    m.set("claim:assignment.treated_side", side, status="confirmed", source=src)
    if movable is not None:
        m.set("claim:assignment.movable", movable, status="confirmed", source=src, said="they knew the line and could argue their score")
    return m


def test_cutoff_only_false_stops_by_code_and_unasked_asks_back():
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY, memory_=memory("gov_transfers", cutoff_only=False, said="the pension also kicks in at that income")))
    r = out["specialist_result"]
    assert r["status"] == "infeasible" and out["feasibility"].stage == "assess" and "something else" in out["feasibility"].reason
    assert "DesignAssessment" not in fake.calls
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY, memory_=store.migrate("gov_transfers", write=False)))
    r = out["specialist_result"]
    assert (
        r["status"] == "ask" and r["ask"]["address"] == "claim:cutoff_only.believed" and r["ask"]["options"] == ["yes", "no"] and r["ask"]["stage"] == "assess"
    )


def test_movable_true_hardens_a_real_density_jump_unless_the_score_has_a_set_by_fact(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "manip",
        manipulated(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    m = _rule(memory("manip"), movable=True)
    fake = FakeLLM(URUGUAY_SCORE, {}, "col:x.note")
    out = _run(fake, handoff("manip", "y", None, ["x", "y"], "col:x.note", memory_=m))
    assert fake.calls.count("Score") == 0  # the pack settles the rule
    density = next(c for c in out["checks"] if c.name == "density")
    assert density.level == "hard" and "crossed it on purpose" in density.detail and density.value is not None and density.value < 0.10
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "assess"
    # the person also said who set the score: the jump is a caveat the assessment must argue, not a hard stop
    m2 = _rule(memory("manip"), movable=True)
    m2.set("col:x.set_by", "the registry, from records the unit never saw", status="confirmed", source="user:turn:2")
    fake = FakeLLM(URUGUAY_SCORE, {}, "col:x.note")
    out = _run(fake, handoff("manip", "y", None, ["x", "y"], "col:x.note", memory_=m2))
    assert next(c for c in out["checks"] if c.name == "density").level == "soft" and out["specialist_result"]["status"] == "done"


def test_movable_unasked_with_a_real_density_jump_asks_back(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "manip",
        manipulated(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    fake = FakeLLM(URUGUAY_SCORE, {}, "col:x.note")
    out = _run(fake, handoff("manip", "y", None, ["x", "y"], "col:x.note", memory_=_rule(memory("manip"))))
    r = out["specialist_result"]
    assert (
        r["status"] == "ask"
        and r["ask"]["address"] == "claim:assignment.movable"
        and "density test p = " in r["ask"]["question"]
        and r["ask"]["evidence"] == ["check:above_cutoff_vs_below_cutoff.density"]
    )


def test_a_missing_cutoff_asks_back_instead_of_stopping():
    m = memory("gov_transfers")
    m.set("claim:assignment.kind", "cutoff_rule", status="confirmed", source="user:turn:1")
    m.set("claim:assignment.score_column", "Income_Centered", status="confirmed", source="user:turn:1")
    fake = FakeLLM(Score(column=None, reason="no note states a cutoff", cites=[]), {}, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY, memory_=m))
    r = out["specialist_result"]
    assert (
        r["status"] == "ask" and r["ask"]["address"] == "claim:assignment.cutoff" and "Income_Centered" in r["ask"]["question"] and r["ask"]["stage"] == "score"
    )


def test_on_treated_is_substituted_with_a_record_the_interpretation_cites():
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY, target="on_treated"))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    d = next(x for x in out["declines"] if x.about == "scope.target")
    assert d.kind == "substituted" and d.pack_value == "on_treated" and d.took == "effect_at_cutoff" and d.address == "decline:load.scope_target"
    assert d.address in out["interpretations"][0].cites and "DISAGREEMENTS WITH THE PACK" in r["report"]


def test_covariates_allowed_is_honoured_and_settled_timing_is_not_asked_again(tmp_path, monkeypatch):
    h = handoff(**URUGUAY)
    h.design.covariates_allowed = ["education"]
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, h)
    assert out["specialist_result"]["status"] == "done"
    c = out["design"].covariates
    assert c.balance_tested == ["education"] and "design.covariates_allowed" in {x.column: x.why for x in c.excluded}["age"]
    # the person's word on a column's timing and on what measures the outcome settles the claims the model would be asked
    make_pack(tmp_path, monkeypatch, "sharp", sharp_below(), "Units with a score strictly below 50 got the grant; a unit exactly at 50 did not.", SYNTH_COLS)
    m = memory("sharp")
    m.set("col:z.when", "before", status="confirmed", source="user:turn:2")
    m.set("col:later.measures_outcome", True, status="confirmed", source="user:turn:2")
    sc = Score(
        column="score",
        cutoff=50.0,
        treated_side="below",
        cutoff_value_treated=False,
        takeup_column="got",
        takeup_level="1",
        reason="r",
        cites=["col:score.note"],
    )

    fake = FakeLLM(sc, {"z": dict(predetermined=True)}, "col:score.note")
    out = _run(fake, handoff("sharp", "y", "got", ["score", "y", "got", "z", "later"], "col:score.note", memory_=m))
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    assert fake.asked("CovariateRoles") == ["z"] and out["design"].covariates.balance_tested == ["z"]
    assert "COLUMN 'z'" in fake.humans_of("CovariateRoles")[0] and "predetermined = true [col:z.when]" in fake.humans_of("CovariateRoles")[0]
    assert "col:later.measures_outcome" in {x.column: x.why for x in out["design"].covariates.excluded}["later"]


def test_a_rejected_score_block_is_recorded_and_the_model_told_why():
    h = handoff("senate3", "vote", None, ["vote", "margin", "state", "year"], "col:margin.note")
    h.design.cutoff = 999.0
    sc = Score(
        column="margin",
        cutoff=0.0,
        treated_side="above",
        cutoff_value_treated=True,
        takeup_column=None,
        takeup_level=None,
        reason="r",
        cites=["col:margin.note"],
    )

    class Fake(FakeLLM):
        def answer(self, schema, human):
            if schema is Score:
                assert "PREVIOUS ANSWER WAS REJECTED" in human and "not strictly inside" in human
            return super().answer(schema, human)

    fake = Fake(sc, {}, "col:margin.note")
    out = _run(fake, h)
    assert fake.calls.count("Score") == 1 and out["score"].cutoff == 0.0
    d = next(x for x in out["declines"] if x.about == "claim:assignment.cutoff")
    assert d.kind == "replaced" and "999" in d.pack_value and d.check == "score.rule_in_file"


def test_placebo_points_are_kept_for_the_figures():
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    pts = out["placebo_points"]
    assert set(pts) == {"placebo_cutoffs", "bandwidth_grid", "donut", "polynomial_grid", "kernel_grid"} and len(pts["bandwidth_grid"]) == 4
    assert [p["label"] for p in pts["polynomial_grid"]] == ["p = 1", "p = 2", "p = 3"] and [p["label"] for p in pts["kernel_grid"]] == [
        "kernel tri",
        "kernel epa",
        "kernel uni",
    ]
    grid = [p for p in pts["bandwidth_grid"] if "value" in p]
    assert grid and all(p["at"] > 0 and p["lo"] <= p["value"] <= p["hi"] for p in grid)
    import json

    arts = json.loads((Path(out["run_dir"]) / "artifacts.json").read_text())
    assert "placebo_points" in arts and arts["case"]["beliefs"]["cutoff_only"] == "confirmed_true"
    assert "effect_below_cutoff_vs_above_cutoff" in [f["id"] for f in json.loads((Path(out["run_dir"]) / "figures.json").read_text())]


def test_the_run_leaves_the_jump_the_density_the_covariates_and_the_bandwidths_as_figures():
    import json

    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    c = out["design"].contrast.key
    figs = {f["id"]: f for f in json.loads((Path(r["run_dir"]) / "figures.json").read_text())}
    assert list(figs) == [f"rd_plot_{c}", f"density_{c}", f"continuity_{c}", f"bandwidths_{c}", f"spec_sensitivity_{c}", f"placebo_cutoffs_{c}", f"effect_{c}"]
    spec = figs[f"spec_sensitivity_{c}"]
    assert (
        spec["series"][0]["x"] == ["p = 1", "p = 2", "p = 3", "kernel tri", "kernel epa", "kernel uni"] and spec["marks"][1]["label"] == "the design's estimate"
    )
    plot = figs[f"rd_plot_{c}"]
    assert [s["name"] for s in plot["series"]] == ["binned means", "fit, control side", "fit, treated side"] and plot["marks"][0]["label"] == "the cutoff"
    jump = plot["series"][2]["y"][0] - plot["series"][1]["y"][-1]
    primary = next(e for e in out["estimates"] if e.method == "local_linear")
    assert primary.ci_low - 0.1 <= jump <= primary.ci_high + 0.1, (jump, primary.value)
    assert "drawn by side" in figs[f"density_{c}"]["note"]  # the rows were sampled by side of the cutoff: the test says nothing here
    assert set(figs[f"continuity_{c}"]["series"][0]["x"]) == {"Education", "Age"}
    assert len(figs[f"bandwidths_{c}"]["series"][0]["x"]) == 4 and figs[f"bandwidths_{c}"]["marks"][0]["label"] == "h used"
    assert "the cutoff" in figs[f"placebo_cutoffs_{c}"]["series"][0]["x"]
    assert not any(x["check"] == "figure.check" for x in r["declines"])


# ------------------------------------------------------------------ the line rung and the effect by a predetermined characteristic


def test_a_risk_the_line_rung_names_is_a_flag_the_assessment_answers_and_the_interpretation_cites():
    risk = Risk(name="cutoff_known_in_advance", reason="households knew the income line before the survey", cites=["change:1.note"])
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"], risks=[risk])
    out = _run(fake, handoff(**URUGUAY))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert not out["ladder"].line.clean and "cutoff_known_in_advance" in [t.name for t in out["ladder"].threats.items]
    flag = next(c for c in out["checks"] if c.name == "threat.cutoff_known_in_advance")
    assert flag.level == "soft" and "knew the income line" in flag.detail and flag.address in out["interpretations"][0].cites
    assert "[ladder:line.risk.cutoff_known_in_advance] households knew" in r["report"] and not out.get("interpret_errors")


def test_the_effect_at_the_cutoff_is_estimated_within_each_level_of_a_predetermined_characteristic(tmp_path, monkeypatch):
    """The covariates rung marks z as one the effect could differ by; the heterogeneity rung picks it; the primary spec is fitted
    again within each quarter of z at the design's bandwidth; the estimates, the figure and the material carry them."""
    df = sharp_below()
    make_pack(tmp_path, monkeypatch, "sharp", df, "Units with a score strictly below 50 got the grant; a unit exactly at 50 did not.", SYNTH_COLS)
    sc = Score(
        column="score",
        cutoff=50.0,
        treated_side="below",
        cutoff_value_treated=False,
        takeup_column="got",
        takeup_level="1",
        reason="r",
        cites=["col:score.note"],
    )
    fake = FakeLLM(sc, {"z": dict(predetermined=True, modifier_candidate=True), "later": dict(is_outcome_measure=True)}, "col:score.note")
    out = _run(fake, handoff("sharp", "y", "got", ["score", "y", "got", "z", "later"], "col:score.note"))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert fake.calls.count("Heterogeneity") == 1 and "[col:z]" in fake.humans_of("Heterogeneity")[0].split("CANDIDATES")[1]
    het = out["ladder"].heterogeneity
    assert het.by == "judgement" and [m.column for m in het.modifiers] == ["z"] and out["design"].modifiers == ["z"]
    within = [e for e in out["estimates"] if e.modifier == "z"]
    assert len(within) == 4 and all(e.level for e in within)
    fitted = [e for e in within if e.error is None]
    assert fitted and all(e.ci_low <= 1.0 <= e.ci_high or abs(e.value - 1.0) < 0.5 for e in fitted)
    c = out["design"].contrast.key
    assert all(e.tag.startswith(f"estimate:{c}.by.z.") for e in within)
    assert {f"{e.tag}.value" for e in fitted} <= set(out["interpretations"][0].cites) and not out.get("interpret_errors")
    figs = {f["id"]: f for f in __import__("json").loads(open(f"{r['run_dir']}/figures.json").read())}
    assert f"effect_by_modifier_{c}" in figs and figs[f"effect_by_modifier_{c}"]["series"][0]["x"][0] == "all rows"
    assert "within z = " in r["report"] and "[ladder:heterogeneity.modifiers] z" in r["report"] and "modifiers    z" in r["report"]


# ------------------------------------------------------------------ the density rung: evidence before the line is judged


def test_the_density_rung_is_computed_once_before_the_line_and_read_by_the_check_and_the_judgement(tmp_path, monkeypatch):
    from causal_agent.families.discontinuity.lane import adapter as A

    make_pack(
        tmp_path,
        monkeypatch,
        "manip",
        manipulated(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    calls: list[int] = []
    real = A.density
    monkeypatch.setattr(N.adapter, "density", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    m = _rule(memory("manip"), movable=False)
    fake = FakeLLM(URUGUAY_SCORE, {}, "col:x.note")
    out = _run(fake, handoff("manip", "y", None, ["x", "y"], "col:x.note", memory_=m))
    assert len(calls) == 1  # the rung ran the library; the check read the rung
    dens = out["ladder"].density
    assert dens.status == "tested" and dens.flagged and dens.p < 0.10 and len(dens.windows) == 5 and dens.windows[0].width < dens.windows[-1].width
    assert dens.windows[0].n_right > dens.windows[0].n_left and dens.histogram and dens.mass_share_left == 0.0
    human = fake.humans_of("Line")[0]
    assert (
        "[ladder:density.test] density" in human
        and "the rows bunch on one side" in human
        and "[ladder:density.windows]" in human
        and "[ladder:density.histogram]" in human
    )
    assert human.index("[ladder:density.test]") < human.index("THE LINE")  # the evidence sits in the ladder the rung reads first
    line = out["ladder"].line
    assert line.clean and "ladder:density.test" in line.cites  # the fake, like a careful model, answered the evidence
    density = next(c for c in out["checks"] if c.name == "density")
    assert density.level == "soft" and "coin-toss p" in density.detail and density.value == pytest.approx(dens.p, abs=1e-4)
    assert "[ladder:density.test]" in out["specialist_result"]["report"]


def test_a_clean_line_over_bunching_is_re_prompted_until_it_answers_the_evidence(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "manip",
        manipulated(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    m = _rule(memory("manip"), movable=True)
    first = Line(clean=True, why="the registry set the score", risks=[], cites=["col:x.note"])
    second = Line(
        clean=False,
        why="the rows bunch just above the line and the person says a unit could move its score",
        risks=[Risk(name="manipulation", reason="units bunch just on the treated side", cites=["ladder:density.test", "col:x.note"])],
        cites=["col:x.note", "ladder:density.test"],
    )
    fake = FakeLLM(URUGUAY_SCORE, {}, "col:x.note", line_script=[first, second])
    out = _run(fake, handoff("manip", "y", None, ["x", "y"], "col:x.note", memory_=m))
    assert fake.calls.count("Line") == 2
    rejected = fake.humans_of("Line")[1]
    assert "PREVIOUS ANSWER WAS REJECTED" in rejected
    assert "a clean verdict must answer it, citing that line" in rejected and "name the manipulation risk" in rejected
    assert out["episodes"]["line"].tries == 2
    assert not out["ladder"].line.clean and "manipulation" in [t.name for t in out["ladder"].threats.items]
    assert next(c for c in out["checks"] if c.name == "threat.manipulation").level == "soft"


def test_a_line_judged_clean_without_bunching_needs_no_density_citation(tmp_path, monkeypatch):
    fake = FakeLLM(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    dens = out["ladder"].density
    assert dens.status == "uninformative" and not dens.flagged  # the rows were drawn by side of the line
    assert "[ladder:density.test] uninformative" in fake.humans_of("Line")[0]
    assert out["ladder"].line.clean and out["specialist_result"]["status"] == "done"


# ------------------------------------------------------------------ the balance rung: every candidate's standing at the line before it is placed


def test_the_balance_rung_lines_are_read_by_the_covariates_rung_and_the_continuity_check(tmp_path, monkeypatch):
    df = sharp_below()
    df["kind"] = np.where(np.arange(len(df)) % 3 == 0, "a", "b")  # a category, balanced across the line
    cols = dict(SYNTH_COLS, kind="A category fixed before the grant.")
    make_pack(tmp_path, monkeypatch, "sharp", df, "Units with a score strictly below 50 got the grant; a unit exactly at 50 did not.", cols)
    sc = Score(
        column="score",
        cutoff=50.0,
        treated_side="below",
        cutoff_value_treated=False,
        takeup_column="got",
        takeup_level="1",
        reason="r",
        cites=["col:score.note"],
    )
    fake = FakeLLM(sc, {"z": dict(predetermined=True), "later": dict(is_outcome_measure=True)}, "col:score.note")
    out = _run(fake, handoff("sharp", "y", "got", ["score", "y", "got", "z", "later", "kind"], "col:score.note"))
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    bal = out["ladder"].balance
    by = {i.column: i for i in bal.items}
    assert (
        by["z"].how == "jump"
        and not by["z"].flagged(bal.threshold)
        and by["later"].how == "jump"
        and by["kind"].how == "share"
        and by["kind"].level in ("a", "b")
    )
    human = fake.humans_of("CovariateRoles")[0]
    assert (
        "[ladder:balance.z] jump" in human
        and "[ladder:balance.kind] share of" in human
        and human.index("[ladder:balance.z]") < human.index("THE COLUMNS TO PLACE")
    )
    cont = next(c for c in out["checks"] if c.name == "covariate_continuity")
    assert cont.level == "pass" and "[ladder:balance.z]" in cont.detail
    assert out["check_facts"]["continuity"]["z"]["p"] == pytest.approx(by["z"].p)  # the check read the rung, the figure reads the check
    assert "[ladder:balance.z]" in out["specialist_result"]["report"]


def test_a_predetermined_verdict_on_a_column_that_jumps_must_cite_the_balance_line(tmp_path, monkeypatch):
    df = sharp_below()
    df["z"] = df["z"] + 1.0 * df["got"]  # the "fixed before" column jumps at the line
    make_pack(tmp_path, monkeypatch, "sharp", df, "Units with a score strictly below 50 got the grant; a unit exactly at 50 did not.", SYNTH_COLS)
    sc = Score(
        column="score",
        cutoff=50.0,
        treated_side="below",
        cutoff_value_treated=False,
        takeup_column="got",
        takeup_level="1",
        reason="r",
        cites=["col:score.note"],
    )
    fake = FakeLLM(sc, {"z": dict(predetermined=True), "later": dict(is_outcome_measure=True)}, "col:score.note", ignore_balance=True)
    out = _run(fake, handoff("sharp", "y", "got", ["score", "y", "got", "z", "later"], "col:score.note"))
    bal = out["ladder"].balance
    assert bal.item("z").flagged(bal.threshold)
    assert fake.calls.count("CovariateRoles") == 2
    rejected = fake.humans_of("CovariateRoles")[1]
    assert "- z jumps at the line [ladder:balance.z]; a column called fixed before the line must say why it still is, citing that line" in rejected
    placed = next(r for r in out["ladder"].covariates.items if r.column == "z")
    assert placed.predetermined and any("ladder:balance.z" in rs.cites for rs in placed.reasons)
    cont = next(c for c in out["checks"] if c.name == "covariate_continuity")
    assert cont.level == "soft" and "z differ at the cutoff" in cont.detail


# ------------------------------------------------------------------ the window: a judgement over a table code builds


def uneven_sides(n_left=3000, n_right=900, seed=8) -> pd.DataFrame:
    """Far more rows just below the line than above it, so a width per side is the reasonable pick."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-1, 0, n_left), rng.uniform(0, 1, n_right)])
    y = 1 + 0.5 * x + 0.8 * (x >= 0) + rng.normal(0, 0.5, len(x))
    return pd.DataFrame({"x": x, "y": y})


def test_the_window_is_a_judgement_that_may_set_a_width_per_side(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "uneven",
        uneven_sides(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    sc = Score(column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    pick = WindowPick(selector="msetwo", why="the sides differ in how many rows sit near the line", cites=["ladder:density.test", "ladder:shape.sides"])
    fake = FakeLLM(sc, {}, "col:x.note", window_script=[pick])
    out = _run(fake, handoff("uneven", "y", None, ["x", "y"], "col:x.note", memory_=_rule(memory("uneven"), movable=False)))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    human = fake.humans_of("WindowPick")[0]
    assert "THE WIDTHS ON OFFER (default: mserd)" in human and "msetwo: an MSE-optimal width for each side" in human and "[ladder:density.test]" in human
    w = out["ladder"].window
    assert w.by == "judgement" and w.selector == "msetwo" and w.rule == "mse" and w.two_sided()
    d = out["design"]
    assert d.bandwidths.selector == "msetwo" and d.bandwidths.h_left != d.bandwidths.h_right and d.bandwidths.h_cer_left is not None
    prim = out["primary"]
    assert prim["h"] == pytest.approx(d.bandwidths.h_left) and prim["h_right"] == pytest.approx(d.bandwidths.h_right)  # the fit ran in the chosen window
    within = [p for p in out["placebo_points"]["bandwidth_grid"] if "h_right" in p]
    assert within and any(abs(p["h"] - p["h_right"]) > 1e-9 for p in within)  # the grid scaled both sides
    itp = out["interpretations"][0]
    assert (
        itp.bandwidth_left_stated == pytest.approx(prim["h"], rel=1e-3)
        and itp.bandwidth_right_stated == pytest.approx(prim["h_right"], rel=1e-3)
        and not out.get("interpret_errors")
    )
    assert "on the control side" in r["report"] and "[ladder:window.selector] msetwo (mse; set by the judgement)" in r["report"]


def test_a_width_other_than_the_default_needs_a_reason_and_an_unknown_selector_is_refused(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "uneven",
        uneven_sides(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    sc = Score(column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    script = [
        WindowPick(selector="narrowest", why="tight", cites=[]),
        WindowPick(selector="cerrd", why="the interval matters more than the point", cites=[]),
        WindowPick(selector="cerrd", why="the interval matters more than the point here", cites=["ladder:shape.sides"]),
    ]
    fake = FakeLLM(sc, {}, "col:x.note", window_script=script)
    out = _run(fake, handoff("uneven", "y", None, ["x", "y"], "col:x.note", memory_=_rule(memory("uneven"), movable=False)))
    assert out["specialist_result"]["status"] == "done"
    humans = fake.humans_of("WindowPick")
    assert len(humans) == 3
    assert "'narrowest' is not one of the selectors on offer" in humans[1]
    assert "a width other than the default needs a reason from the ladder: cite the density, the balance or the sides" in humans[2]
    w = out["ladder"].window
    assert w.selector == "cerrd" and w.rule == "cer" and out["episodes"]["window"].tries == 3 and out["design"].bandwidths.rule == "cer"


def test_when_the_width_judgement_never_settles_the_default_stands_with_a_flag(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "uneven",
        uneven_sides(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    sc = Score(column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    fake = FakeLLM(sc, {}, "col:x.note", window_script=[WindowPick(selector="nope", why="", cites=[])] * 3)
    out = _run(fake, handoff("uneven", "y", None, ["x", "y"], "col:x.note", memory_=_rule(memory("uneven"), movable=False)))
    assert out["specialist_result"]["status"] == "done"
    w = out["ladder"].window
    assert w.by == "code" and w.selector == "mserd" and w.unsure and w.unsure[0].about == "window"
    assert any(c.name == "unsure.window" for c in out["design"].checks.results)


# ------------------------------------------------------------------ local randomisation on a discrete score, end to end


def discrete_with_covariate(n=3000, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 13, n)
    age = 30 + 0.3 * x + rng.normal(0, 4, n)
    y = 0.2 * x + 1.0 * (x >= 7) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y, "age": age})


def test_local_randomisation_runs_end_to_end_on_a_discrete_score(tmp_path, monkeypatch):
    from causal_agent.families.discontinuity.lane.knowledge import estimator as estimator_entry
    from causal_agent.families.discontinuity.lane.knowledge import placebo as placebo_entry

    monkeypatch.setitem(estimator_entry("local_randomisation").params, "reps", 150)  # the budgets, kept small in the test
    monkeypatch.setitem(placebo_entry("rosenbaum_bounds").params, "reps", 40)
    make_pack(
        tmp_path,
        monkeypatch,
        "disc",
        discrete_with_covariate(),
        "Units with a score at or above 7 got the grant.",
        {"x": "The score, an integer from 1 to 12, fixed before the grant.", "y": "The outcome, measured after.", "age": "Age, fixed before the grant."},
    )
    sc = Score(column="x", cutoff=7.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    fake = FakeLLM(sc, {"age": dict(predetermined=True)}, "col:x.note")
    out = _run(fake, handoff("disc", "y", None, ["x", "y", "age"], "col:x.note"))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    d = out["design"]
    assert d.estimator == "local_randomisation" and d.estimand == "effect_in_window" and d.bandwidths.rule == "local_randomisation"
    w = out["ladder"].window
    assert w.by == "code" and w.selector.startswith("window ") and "stay balanced" in w.why and "ladder:balance.age" in w.cites
    assert "WindowPick" not in fake.calls  # the balance window is the library's recommendation, not a judgement
    primary = next(e for e in out["estimates"] if e.method == "local_randomisation")
    assert primary.error is None and primary.p_value_source == "randomisation" and primary.p_value is not None and primary.p_value < 0.05
    assert primary.ci_low <= primary.value <= primary.ci_high and primary.value > 0.5  # the jump plus at most a few steps of slope
    assert out["primary"]["vce"] == "randomisation" and "randomisation p =" in fake.humans_of("RDInterpretation")[0]
    assert {x.refuter for x in out["refutations"]} == {"window_sensitivity", "rosenbaum_bounds"}
    sens = next(x for x in out["refutations"] if x.refuter == "window_sensitivity")
    assert sens.kind == "sensitivity" and sens.passed is None and sens.range_low is not None and sens.range_low <= primary.value <= sens.range_high + 1e-9
    bounds = next(x for x in out["refutations"] if x.refuter == "rosenbaum_bounds")
    assert bounds.kind == "sensitivity" and "gamma 0.1: p between" in bounds.detail
    itp = out["interpretations"][0]
    assert itp.estimand == "effect_in_window" and not out.get("interpret_errors")
    figs = {f["id"]: f for f in __import__("json").loads(open(f"{r['run_dir']}/figures.json").read())}
    c = d.contrast.key
    assert f"windows_{c}" in figs and figs[f"windows_{c}"]["title"] == "The estimate across windows" and f"bandwidths_{c}" not in figs


def test_local_randomisation_without_a_covariate_falls_back_to_the_support_points_window(tmp_path, monkeypatch):
    from causal_agent.families.discontinuity.lane.knowledge import estimator as estimator_entry
    from causal_agent.families.discontinuity.lane.knowledge import placebo as placebo_entry

    monkeypatch.setitem(estimator_entry("local_randomisation").params, "reps", 150)
    monkeypatch.setitem(placebo_entry("rosenbaum_bounds").params, "reps", 40)
    make_pack(
        tmp_path,
        monkeypatch,
        "disc",
        discrete(n=900),
        "Units with a score at or above 7 got the grant.",
        {"x": "The score, an integer from 1 to 12, fixed before the grant.", "y": "The outcome, measured after."},
    )
    sc = Score(column="x", cutoff=7.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    fake = FakeLLM(sc, {}, "col:x.note")
    out = _run(fake, handoff("disc", "y", None, ["x", "y"], "col:x.note"))
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    w = out["ladder"].window
    assert w.selector == "support_points" and w.rule == "local_randomisation" and "no predetermined covariate" in w.why and w.h_left == 3.5 == w.h_right
    assert out["design"].estimator == "local_randomisation" and next(e for e in out["estimates"] if e.method == "local_randomisation").error is None


def test_no_yaml_key_is_dead():
    """Every key of every catalogue entry is a field of its model, and every field the model declares is read somewhere in the
    lane's code: a yaml line nobody reads is a promise the design does not keep."""
    import yaml

    from causal_agent.families.discontinuity.lane.knowledge import EstimatorEntry, InferenceEntry, PlaceboEntry

    here = Path(__file__).resolve().parents[1] / "lane"
    harness = Path(__file__).resolve().parents[3] / "lane" / "knowledge.py"  # the shared loader reads rank and prefer_over
    code = (
        "".join((here / f).read_text() for f in ("nodes.py", "checks.py", "adapter.py"))
        + (here / "knowledge" / "__init__.py").read_text()
        + harness.read_text()
    )
    for file, model in (("estimators.yaml", EstimatorEntry), ("placebos.yaml", PlaceboEntry), ("inference.yaml", InferenceEntry)):
        raw = yaml.safe_load((here / "knowledge" / file).read_text())
        fields = set(model.model_fields) - {"name"}
        for name, entry in raw.items():
            extra = set(entry) - fields
            assert not extra, f"{file}: {name} has keys no field reads: {sorted(extra)}"
        for field in fields:
            assert f".{field}" in code, f"{file}: the field {field!r} is declared on {model.__name__} but nothing in the lane reads it"


def clustered_sharp(n_clusters=12, per=120, seed=13) -> pd.DataFrame:
    """Units nested in a dozen clusters, each with its own level; a sharp line at zero."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_clusters):
        x = rng.uniform(-1, 1, per)
        rows.append(pd.DataFrame({"x": x, "y": 1 + 0.5 * x + 1.0 * (x >= 0) + rng.normal(0, 0.3, per) + 0.4 * g, "unit": f"g{g}"}))
    return pd.concat(rows, ignore_index=True)


def test_few_clusters_pick_the_corrected_variance_and_flag_the_count(tmp_path, monkeypatch):
    make_pack(
        tmp_path,
        monkeypatch,
        "clust",
        clustered_sharp(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after.", "unit": "The group the unit belongs to."},
        entity=["unit"],
    )
    sc = Score(column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    fake = FakeLLM(sc, {}, "col:x.note")
    out = _run(fake, handoff("clust", "y", None, ["x", "y", "unit"], "col:x.note", memory_=_rule(memory("clust"), movable=False)))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert out["shape"].cluster_column == "unit" and out["shape"].clusters == 12
    d = out["design"]
    assert d.inference == "cluster_few" and d.vce == "cr2" and d.cluster == "unit"
    few = next(c for c in out["checks"] if c.name == "few_clusters")
    assert few.level == "soft" and few.value == 12.0 and few.address in out["interpretations"][0].cites
    assert out["primary"]["vce"].upper() == "CR2"


# ------------------------------------------------------------------ the threats this design names by code, from the rungs below


def test_the_threats_rung_names_bunching_the_line_rung_did_not_a_coarse_score_and_one_sided_takeup(tmp_path, monkeypatch):
    from causal_agent.families.discontinuity.lane.knowledge import estimator as estimator_entry
    from causal_agent.families.discontinuity.lane.knowledge import placebo as placebo_entry

    # bunching the line rung called clean (citing the evidence): the threats rung still carries it as a flag
    make_pack(
        tmp_path,
        monkeypatch,
        "manip",
        manipulated(),
        "Units with a score at or above zero got the grant.",
        {"x": "The score, fixed before the grant.", "y": "The outcome, measured after."},
    )
    fake = FakeLLM(URUGUAY_SCORE, {}, "col:x.note")
    out = _run(fake, handoff("manip", "y", None, ["x", "y"], "col:x.note", memory_=_rule(memory("manip"), movable=False)))
    names = {t.name: t for t in out["ladder"].threats.items}
    assert (
        out["ladder"].line.clean
        and "manipulation" in names
        and "did not name it" in names["manipulation"].text
        and "ladder:density.test" in names["manipulation"].cites
    )
    assert next(c for c in out["checks"] if c.name == "threat.manipulation").address in out["interpretations"][0].cites
    # a coarse score
    monkeypatch.setitem(estimator_entry("local_randomisation").params, "reps", 100)
    monkeypatch.setitem(placebo_entry("rosenbaum_bounds").params, "reps", 30)
    make_pack(
        tmp_path,
        monkeypatch,
        "disc",
        discrete(n=900),
        "Units with a score at or above 7 got the grant.",
        {"x": "The score, an integer from 1 to 12, fixed before the grant.", "y": "The outcome, measured after."},
    )
    sc = Score(column="x", cutoff=7.0, treated_side="above", cutoff_value_treated=True, takeup_column=None, takeup_level=None, reason="r", cites=["col:x.note"])
    out = _run(FakeLLM(sc, {}, "col:x.note"), handoff("disc", "y", None, ["x", "y"], "col:x.note"))
    assert (
        "discrete_score" in {t.name for t in out["ladder"].threats.items}
        and "12 distinct values" in next(t for t in out["ladder"].threats.items if t.name == "discrete_score").text
    )
    # take-up on one side only
    make_pack(
        tmp_path, monkeypatch, "fuzzy", fuzzy_above(), "Units with a score at or above zero were offered the grant; about seven in ten took it up.", FUZZY_COLS
    )
    sc = Score(
        column="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, takeup_column="received", takeup_level="1", reason="r", cites=["col:x.note"]
    )
    out = _run(FakeLLM(sc, {}, "col:x.note"), handoff("fuzzy", "y", "eligible", ["x", "y", "eligible", "received"], "col:x.note"))
    t = {t.name: t for t in out["ladder"].threats.items}
    assert "one_sided_takeup" in t and t["one_sided_takeup"].cites == ["ladder:shape.kind"] and "thin_side" not in t
