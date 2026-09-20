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
from causal_agent.families.discontinuity.lane.contracts import CovariateRelation, DesignAssessment, EstimatorPick, RDInterpretation, Score
from causal_agent.families.discontinuity.lane.graph import compile_local
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
    """Scripted answers: a Score, relations per column, and defaults for the rest that read the material like a careful model would."""

    def __init__(
        self, score: Score, relations: dict, cite: str, *, bad_cites=False, assess_script=None, pick_script=None, score_script=None, interpret_bad_first=False
    ):
        self.score, self.relations, self.cite, self.bad_cites = score, relations, cite, bad_cites
        self.assess_script, self.pick_script, self.score_script = list(assess_script or []), list(pick_script or []), list(score_script or [])
        self.interpret_bad_first = interpret_bad_first
        self.calls: list[str] = []

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

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        cite = "col:nope.note" if self.bad_cites else self.cite
        if schema is Score:
            s = (self.score_script.pop(0) if self.score_script else self.score).model_copy()
            s.cites = [cite]
            return s
        if schema is CovariateRelation:
            col = re.search(r"for column '([^']+)'", human).group(1)
            flags = dict(predetermined=False, affected_by_treatment=False, is_outcome_measure=False)
            flags.update(self.relations.get(col, {}))
            return CovariateRelation(column=col, reasons=[Cited(reason=f"{col}: {k}", cites=[cite]) for k, v in flags.items() if v], **flags)
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
            h = float(re.search(r"\[estimate:%s\.bandwidth\] h = ([-\d.eE+]+)" % re.escape(c), human).group(1))
            required = human.split("ADDRESSES YOU MUST CITE")[1].split("\n\n")[0].strip().splitlines()
            bad = self.interpret_bad_first and "PREVIOUS ANSWER WAS REJECTED" not in human
            return RDInterpretation(
                contrast=c,
                answer=f"At the cutoff the effect is {value:.3g}, interval {lo:.3g} to {hi:.3g}; local to units at the cutoff.",
                effect_stated=value,
                caveats=["local to the cutoff"],
                cites=["nope:x"] if bad else required,
                estimand=estimand,
                bandwidth_stated=h,
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
    assert {x.refuter for x in out["refutations"]} == {"placebo_cutoffs", "bandwidth_grid", "donut"}
    assert len(out["interpretations"]) == 1 and not out.get("interpret_errors")
    assert out["interpretations"][0].estimand == "effect_at_cutoff"
    assert "DESIGN" in r["report"] and "ANSWER" in r["report"]
    assert fake.calls.count("CovariateRelation") == 2 and fake.calls.count("Score") == 1 and "DesignAssessment" in fake.calls
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
    out = _run(fake, handoff("fuzzy", "y", "eligible", ["x", "y", "eligible", "received"], "col:x.note"))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    s = out["shape"]
    assert s.kind == "fuzzy" and s.takeup_left == 0.0 and 0.6 < s.takeup_right < 0.8
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
    fake = FakeLLM(sc, {}, "col:x.note")
    out = _run(fake, handoff("disc", "y", None, ["x", "y"], "col:x.note"))
    levels = {c.name: c.level for c in out["checks"]}
    details = {c.name: c.detail for c in out["checks"]}
    assert levels["support"] == "soft" and levels["mass_points"] == "soft"
    assert levels["density"] == "soft" and "not computable" in details["density"]
    assert out["specialist_result"]["status"] == "done", out.get("feasibility")
    assert out["shape"].distinct_scores == 12
    d = out["design"]
    assert d.bandwidths.rule == "support_points" and d.bandwidths.h_cer is None and d.bandwidths.h == 3.5
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


def test_relate_bad_cites_loop_then_stop():
    class Fake(FakeLLM):
        def answer(self, schema, human):
            self.bad_cites = schema is CovariateRelation
            return super().answer(schema, human)

    fake = Fake(URUGUAY_SCORE, URUGUAY_RELATIONS, URUGUAY["cite"])
    out = _run(fake, handoff(**URUGUAY))
    assert out["feasibility"].stage == "verify" and fake.calls.count("CovariateRelation") == 2 * 3


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
        if e.name == "local_randomisation":
            continue
        f = adapter.fit(e.params, toy, fuzzy=bool(e.fuzzy is True), covs=None)
        assert f.error is None, (e.name, f.error)
    assert [i.name for i in load_inference()] == ["cluster_entity", "robust_bc"]
    assert {p.name for p in load_placebos()} == {"placebo_cutoffs", "bandwidth_grid", "donut"}
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
    from causal_agent.desk.route import graph as route_graph
    from causal_agent.families.registry import lanes

    SPECIALISTS = lanes()

    assert "score" in SPECIALISTS["discontinuity"].get_graph().nodes
    assert "shape_table" in SPECIALISTS["diff_in_diff"].get_graph().nodes and "score" not in SPECIALISTS["diff_in_diff"].get_graph().nodes
    assert "relate" not in SPECIALISTS["synthetic_control"].get_graph().nodes  # still a stub
    assert "specialist_discontinuity" in route_graph.get_graph().nodes


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

    class Fake(FakeLLM):
        def answer(self, schema, human):
            if schema is CovariateRelation:
                assert "for column 'z'" in human and "predetermined = true [col:z.when]" in human
            return super().answer(schema, human)

    fake = Fake(sc, {"z": dict(predetermined=True)}, "col:score.note")
    out = _run(fake, handoff("sharp", "y", "got", ["score", "y", "got", "z", "later"], "col:score.note", memory_=m))
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    assert fake.calls.count("CovariateRelation") == 1 and out["design"].covariates.balance_tested == ["z"]
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
    assert set(pts) == {"placebo_cutoffs", "bandwidth_grid", "donut"} and len(pts["bandwidth_grid"]) == 4
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
    assert list(figs) == [f"rd_plot_{c}", f"density_{c}", f"continuity_{c}", f"bandwidths_{c}", f"placebo_cutoffs_{c}", f"effect_{c}"]
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
