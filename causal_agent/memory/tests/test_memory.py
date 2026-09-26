"""The memory: records, the claim-table view both ways, the write gate, the consistency rules, what is open, the store."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from causal_agent.families import registry as R
from causal_agent.memory import ops, store
from causal_agent.memory.catalogue import load_catalogue
from causal_agent.memory.claims import ClaimTable
from causal_agent.memory.records import Memory
from causal_agent.memory.store import load_claims
from causal_agent.profile.datasets import ROOT, dataset_entries
from causal_agent.profile.profiler import Profile, profile

CAT = load_catalogue()
STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


def students3() -> tuple[Memory, ClaimTable]:
    e = dataset_entries()["students3"]
    prof = Profile.model_validate(json.loads((ROOT / e["profile"]).read_text()))
    table, _ = load_claims(ROOT / e["claims"])
    return Memory.from_claims("students3", table, profile=prof, csv=e["csv"]), table


# ------------------------------------------------------------------ records and the two views


def test_claims_round_trip_through_memory():
    m, table = students3()
    back = m.to_claims()
    for key, c in table.claims.items():
        b = back.claims[key]
        assert b.kind == c.kind and b.status == c.status and b.source == c.source, key
        want = {("moved_by_change" if k == "affected_by_treatment" else k): v for k, v in c.fields.items() if v is not None}
        assert want == b.fields, key
    # the renamed field carries over
    assert m.field("col:reading_score.moved_by_change").value is True
    # every profiled column carries the file's facts, even one with no claim
    assert m.column("math score").facts.kind == "numeric" and m.column("math score").facts.distinct > 50
    # the map holds only what was said: no empty field for a column nobody described
    assert not any(a.startswith("col:math_score.") for a in m.fields) or m.field("col:math_score.meaning") is not None


def test_field_addresses_and_raw_set():
    m, _ = students3()
    assert m.field("col:lunch.when").value == "after" and m.field("claim:col:lunch.when") is m.field("col:lunch.when")
    assert m.field("claim:assignment.kind").value == "own_choice" and m.field("assignment.kind").value == "own_choice"
    assert m.field("col:lunch.nope") is None and m.field("claim:nope.kind") is None
    v = m.version
    f = m.set("col:lunch.set_by", "the district", status="confirmed", source="user:turn:9", said="the district sets it")
    assert f.said == "the district sets it" and m.version == v + 1
    assert "col:lunch.set_by" in m.addresses() and "claim:assignment.kind" in m.addresses()
    assert "[col:lunch.when] after" in m.render()


def test_collapse_keeps_required_confirmed_when_only_an_optional_is_drafted():
    m, _ = students3()
    m.set("col:lunch.stands_for", "household income", status="drafted", source="model:infer")
    assert m.to_claims().claims["col:lunch"].status == "confirmed"
    m.set("col:lunch.when", "before", status="drafted", source="model:infer")
    assert m.to_claims().claims["col:lunch"].status == "drafted"
    m.set("col:lunch.when", "before", status="confirmed", source="user:turn:3")
    m.field("col:lunch.when").status = "refuted"
    assert m.to_claims().claims["col:lunch"].status == "refuted"


def test_snapshot_and_fork_do_not_share_state():
    m, _ = students3()
    s, f = m.snapshot(), m.fork()
    assert f.version == m.version + 1 and s.version == m.version
    f.set("claim:assignment.kind", "lottery", status="confirmed", source="user:turn:11")
    assert m.field("claim:assignment.kind").value == "own_choice" and s.field("claim:assignment.kind").value == "own_choice"


# ------------------------------------------------------------------ seed


def test_seed_from_the_profile_alone():
    prof = profile(STUDENTS)
    m = ops.seed("students", prof, csv=str(STUDENTS))
    assert len(m.columns) == 8 and m.column("lunch").facts.levels() == ["standard", "free/reduced"]
    assert m.field("claim:missing.why").value == "none" and m.field("claim:missing.why").source == "data"
    assert m.field("claim:grain.panel") is None  # no entity declared, so the file cannot say
    assert set(m.fields) == {"claim:missing.why"}  # a wide file is a small memory until someone speaks


# ------------------------------------------------------------------ apply: the gate


def test_apply_gates_source_confirmed_beliefs_and_values():
    m, _ = students3()
    U = ops.Update
    rejected = ops.apply(
        m,
        [
            U(address="col:lunch.when", value="before", status="confirmed", source=""),  # no source
            U(address="col:lunch.when", value="before", status="confirmed", source="model:infer"),  # a model may only draft
            U(address="col:lunch.when", value="before", status="drafted", source="model:infer"),  # confirmed field, model cannot touch
            U(address="claim:unobserved.exists", value="false", status="drafted", source="model:infer"),  # a belief needs the person
            U(address="col:lunch.when", value="sometime", status="confirmed", source="user:turn:8"),  # not an option
            U(address="col:nope.when", value="before", status="confirmed", source="user:turn:8"),  # no such column
            U(address="claim:assignment.depends_on", value="lunch, nope", status="confirmed", source="user:turn:8"),  # a column list with a stranger
            U(address="col:lunch.when", value="before", status="confirmed", source="user:turn:8", said="the district set it in September"),  # accepted
            U(address="col:gender.stands_for", value="sex", status="drafted", source="model:infer"),  # a draft on an empty optional field: accepted
            U(address="claim:unobserved.exists", value="true", status="confirmed", source="user:turn:8", said="the counsellor knew the kids"),  # accepted
            U(address="claim:sampling.detail", value=None, status="unknown", source="user:turn:8"),  # unknown: accepted
        ],
    )
    assert len(rejected) == 7, rejected
    assert "no source" in rejected[0] and "may only draft" in rejected[1] and "confirmed by the person" in rejected[2]
    assert "belief" in rejected[3] and "must be one of" in rejected[4] and "not a column" in rejected[5] and "not columns" in rejected[6]
    f = m.field("col:lunch.when")
    assert f.value == "before" and f.status == "confirmed" and f.said == "the district set it in September"
    assert m.field("col:gender.stands_for").status == "drafted"
    assert m.field("claim:unobserved.exists").value is True and m.field("claim:sampling.detail").status == "unknown"


# ------------------------------------------------------------------ roles and consistency


def test_roles_are_a_view_of_the_dataset_fields():
    m, _ = students3()
    v = m.version
    r = ops.roles(m, outcome="math score")
    assert r["math_score"] == "outcome" and r["test_preparation_course"] == "treatment" and r["lunch"] == "depends_on"
    assert "gender" not in r
    assert m.version == v and not any(a.endswith(".role") for a in m.fields)  # nothing written
    # a panel's key columns are units except the period column, which the memory or the caller names
    m.set("claim:grain.panel", True, status="confirmed", source="data")
    m.set("claim:grain.key_columns", ["gender", "lunch"], status="drafted", source="data")
    assert ops.roles(m, time="lunch")["gender"] == "unit" and ops.roles(m, time="lunch")["lunch"] == "time"


def test_consistency_refutes_without_overwriting():
    m, _ = students3()  # lunch is 'after' in this session's claims, yet the offer depended on it
    findings = ops.consistency(m, outcome="math score")
    rules = {(f.address, f.rule) for f in findings}
    assert ("col:lunch.when", "depends_on_before") in rules
    f = m.field("col:lunch.when")
    assert f.value == "after" and f.status == "refuted" and "check:col:lunch.when.depends_on_before" in f.evidence
    # an outcome marked before, a score marked after, a before-column marked moved
    m.set("col:math_score.when", "before", status="confirmed", source="user:turn:2")
    m.set("col:gender.moved_by_change", True, status="drafted", source="model:infer")
    rules = {(f.address, f.rule) for f in ops.consistency(m, outcome="math score")}
    assert ("col:math_score.when", "outcome_after") in rules and ("col:gender.moved_by_change", "moved_not_before") in rules
    assert m.field("col:gender.moved_by_change").value is True  # never overwritten


def test_check_runs_the_data_facts_and_marks_fields():
    m, _ = students3()
    df = pd.read_csv(STUDENTS)
    prof = profile(STUDENTS)
    m.set("claim:assignment.treated_level", "finished", status="confirmed", source="user:turn:3")
    findings = ops.check(m, df, prof)
    bad = [f for f in findings if f.passed is False]
    assert any(f.rule == "treated_level" and f.address == "claim:assignment.treated_level" for f in bad)
    assert m.field("claim:assignment.treated_level").status == "refuted"
    assert m.field("claim:assignment.treated_level").value == "finished"


# ------------------------------------------------------------------ probe, fit, open


def test_fit_and_open_on_students3():
    m, _ = students3()
    df = pd.read_csv(STUDENTS)
    probes = ops.probe(m, df, R.REGISTRY.values())
    assert any(p.family == "adjustment" and p.name == "arms" and p.passed for p in probes)
    st = ops.fit(m, probes, R.needs())
    assert "adjustment" in st.surviving and st.ready
    assert ops.open(m, st, R.needs()) == []
    # make one required field vague and one optional draft: both come back, required first
    m.set("col:lunch.when", None, status="empty", source=None)
    m.set("col:gender.stands_for", "sex", status="drafted", source="model:infer")
    st = ops.fit(m, probes, R.needs())
    opened = ops.open(m, st, R.needs())
    assert {o.address for o in opened} == {"col:gender.stands_for", "col:lunch.when"}
    lunch = next(o for o in opened if o.address == "col:lunch.when")
    assert lunch.options == ["before", "at", "after", "unknown"] and "adjustment" in lunch.because and not lunch.optional
    assert next(o for o in opened if o.address == "col:gender.stands_for").status == "drafted"
    assert not st.ready


# ------------------------------------------------------------------ store


def test_store_round_trip_and_migration(tmp_path):
    root = tmp_path
    (root / "data").mkdir()
    (root / "data" / "datasets.yaml").write_text(Path(ROOT / "data/datasets.yaml").read_text())
    for rel in ("data/profiles/students3.json", "data/claims/students3.yaml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text((ROOT / rel).read_text())
    m = store.migrate("students3", root)
    assert store.exists("students3", root)
    back = store.load("students3", root)
    assert back.model_dump() == m.model_dump()
    assert back.field("claim:assignment.kind").value == "own_choice" and back.column("lunch").facts.distinct == 2
    assert (root / "data/memory/students3/fields.yaml").exists() and (root / "data/memory/students3/said.jsonl").exists()
    d = store.snapshot(back, 1, root)
    assert (d / "memory.json").exists() and Memory.model_validate_json((d / "memory.json").read_text()).name == "students3"
    with pytest.raises(KeyError):
        store.migrate("nope", root)


def test_the_next_design_id_skips_every_design_on_disk(tmp_path):
    assert store.next_design_id("fresh", tmp_path) == 1
    for n in (1, 2, 7):
        (tmp_path / "data/memory/fresh/designs" / str(n)).mkdir(parents=True)
    (tmp_path / "data/memory/fresh/designs" / "notes").mkdir()
    assert store.next_design_id("fresh", tmp_path) == 8


# ------------------------------------------------------------------ what each assignment kind needs


def test_a_cutoff_rule_needs_its_score_and_cutoff_and_a_lottery_does_not():
    kind = CAT.kinds["assignment"]
    assert "score_column" not in kind.required({}) and "treatment_column" not in kind.required({})
    assert {"score_column", "cutoff", "treated_side", "movable"} <= set(kind.required({"kind": "cutoff_rule"}))
    assert "score_column" not in kind.required({"kind": "lottery"}) and {"treatment_column", "treated_level"} <= set(kind.required({"kind": "own_choice"}))
    assert "period_value" in CAT.kinds["change"].required({"date_column": "year"}) and "period_value" not in CAT.kinds["change"].required({})
    # in the memory: a cutoff rule with no cutoff is a drafted claim, not a settled one, and the cutoff is open
    m, _ = students3()
    m.set("claim:assignment.kind", "cutoff_rule", status="confirmed", source="user:turn:2")
    assert m.to_claims().claims["assignment"].status == "drafted"
    df = pd.read_csv(STUDENTS)
    st = ops.fit(m, ops.probe(m, df, R.REGISTRY.values()), R.needs())
    opened = {o.address: o for o in ops.open(m, st, R.needs())}
    assert "claim:assignment.score_column" in opened and not opened["claim:assignment.score_column"].optional
    assert "claim:assignment.depends_on" not in opened
