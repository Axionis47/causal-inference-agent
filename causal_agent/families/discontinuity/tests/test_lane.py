"""RD context boundaries and engine honesty. Scripted judgements are the existing suite's floor."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Candidate, ColumnBrief, ColumnFacts, Fact, Handoff, Scope
from causal_agent.common.llm import set_llm
from causal_agent.explore.tools import RunCodeStep
from causal_agent.families.discontinuity.design import RdDesign
from causal_agent.families.discontinuity.lane.contracts import AssumptionDecision, AssumptionDecisions, Line, RDInterpretation
from causal_agent.families.discontinuity.lane.engine import fit as adapter
from causal_agent.families.discontinuity.lane.graph import compile_local
from causal_agent.families.discontinuity.lane.knowledge import CATALOGUE, SPEC, load_falsifications, load_sensitivities
from causal_agent.lane.endpoints import admit, validate_context, verify_catalogue


def admitted_pack(tmp_path, *, fuzzy=False, local_randomisation=False, first_stage=True):
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, 1000)
    t = ((x >= 0) & (rng.random(1000) < (0.6 if first_stage else 0))) if fuzzy else x >= 0
    df = pd.DataFrame({"x": x, "t": t.astype(int), "y": 2 + 3 * t + rng.normal(size=1000)})
    csv = tmp_path / "input.csv"
    df.to_csv(csv, index=False)
    values = {
        "target.locality": "cutoff",
        "target.comparison": "receipt",
        "assignment.kind": "cutoff_rule",
        "assignment.score_column": "x",
        "assignment.cutoff": 0.0,
        "assignment.treated_side": "above",
        "assignment.cutoff_value_treated": True,
        "assignment.movable": False,
        "assignment.compliance": "fuzzy" if fuzzy else "sharp",
        "grain.row_is": "one independently sampled person",
        "sampling.how": "random sample",
        "cutoff_only.believed": True,
        "assignment.treatment_column": "t",
        "assignment.treated_level": "1",
        "exclusion.believed": True,
        "monotonicity.believed": True,
        "local_randomisation.believed": local_randomisation,
    }
    facts = {f"claim:{a}": Fact(value=v, status="confirmed", source="user:turn:1", said="The declared study account.") for a, v in values.items()}
    facts["col:x.when"] = Fact(value="before", status="confirmed", source="user:turn:1")
    h = Handoff(
        family="discontinuity",
        specialist="rdrobust",
        supported_now=True,
        outcome="y",
        treatment="t",
        scope=Scope(),
        pack_name="rd_eval",
        csv=str(csv),
        question="Did receipt change the outcome?",
        relevant_columns=[Candidate(column=k, reason="the analysis pair and score", cites=[f"col:{k}"]) for k in df],
        chosen_assumption="continuity or admitted local randomisation",
        reasons=[],
        columns=[ColumnBrief(name=k, key=k, when="after" if k == "y" else "before", facts=ColumnFacts(kind="numeric", distinct=df[k].nunique())) for k in df],
        assignment={k.partition(".")[2]: v for k, v in values.items() if k.startswith("assignment.")},
        facts=facts,
        design=RdDesign(
            score="x", cutoff=0.0, treated_side="above", cutoff_value_treated=True, score_fixed_before=True, movable=False, takeup={"column": "t", "level": "1"}
        ),
    )
    return admit(h, CATALOGUE)


class FakeLLM:
    """Rescripted schemas of the old RD workflow, with the real engines underneath."""

    def __init__(self, schema=None):
        self.schema = schema

    def with_structured_output(self, schema, **kwargs):
        return FakeLLM(schema)

    def bind_tools(self, tools):
        return FakeLLM()

    def invoke(self, messages):
        text = str(messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1].content)
        if self.schema is None:
            return AIMessage(content="The provided evidence is sufficient.")
        if self.schema is RunCodeStep:
            return RunCodeStep(done=True)
        if self.schema is Line:
            return Line(clean=True, why="The score was set before assignment, under the declared study account.", cites=["claim:assignment.movable"])
        if self.schema is AssumptionDecisions:
            ids = re.findall(r"\[endpoint:([^\]]+)\]", text)
            return AssumptionDecisions(
                items=[
                    AssumptionDecision(
                        endpoint_id=e,
                        admissible=True,
                        reason="the admitted account supports this conditional endpoint",
                        cites=[f"endpoint:{e}"],
                        assumptions=["the stated identifying assumptions hold"],
                    )
                    for e in ids
                ]
            )
        if self.schema is RDInterpretation:
            did = re.search(r"\[design:([^.]+)\.target\]", text).group(1)
            eid = re.search(r'"estimand": "([^"]+)"', text).group(1)
            mapping = {
                "effect_at_cutoff": "sharp_local_polynomial",
                "complier_effect_at_cutoff": "fuzzy_local_polynomial",
                "itt_at_cutoff": "eligibility_local_polynomial",
                "effect_in_window": "sharp_local_randomisation",
                "complier_effect_in_window": "fuzzy_local_randomisation",
                "itt_in_window": "eligibility_local_randomisation",
            }
            tag = re.search(r"\[(estimate:[^\]]+)\.value\] ([^\n]+)", text)
            value = None if tag.group(2) == "None" else float(tag.group(2))
            cites = re.findall(r"\[([^\]]+)\]", text)
            return RDInterpretation(
                target_contrast="assignment",
                design_id=did,
                endpoint_id=mapping[eid],
                contrast="treated_side_vs_control_side",
                effect_stated=value,
                answer="The effect is local to the recorded population and identifying assumptions.",
                caveats=["conditional on the admitted study account"],
                cites=cites,
            )
        raise AssertionError(self.schema)


def run_pack(h):
    set_llm(FakeLLM())
    try:
        return compile_local().invoke({"handoff": h, "question": h.question}, {"configurable": {"thread_id": "rd-test"}})
    finally:
        set_llm(None)


def test_unadmitted_context_stops_before_reasoning(tmp_path):
    h = admitted_pack(tmp_path).model_copy(update={"admission": None})
    out = run_pack(h)
    assert out["context_deficiency"].addresses == ["context.admission"]
    assert not out.get("estimates")
    assert out["specialist_result"]["status"] == "context_deficient"
    assert out["specialist_result"]["terminal"]["kind"] == "context_deficiency"


def test_context_tampering_is_a_typed_deficiency(tmp_path):
    h = admitted_pack(tmp_path)
    assert validate_context(h, CATALOGUE) is None
    changed = h.model_copy(update={"memory_version": h.memory_version + 1})
    assert validate_context(changed, CATALOGUE).addresses == ["context.identity"]
    fuzzy = admitted_pack(tmp_path, fuzzy=True)
    for required in ("claim:target.locality", "claim:target.comparison"):
        facts = {a: f for a, f in fuzzy.facts.items() if a != required}
        incomplete = admit(fuzzy.model_copy(update={"facts": facts}), CATALOGUE)
        assert not incomplete.admission.ready and required in validate_context(incomplete, CATALOGUE).addresses


def test_sharp_design_is_frozen_and_execution_is_design_qualified(tmp_path):
    out = run_pack(admitted_pack(tmp_path))
    assert out["specialist_result"]["status"] == "done"
    d = out["designs"][out["lead_design_id"]]
    assert d.endpoint_id == "sharp_local_polynomial" and d.target["estimand"] == "effect_at_cutoff"
    e = next(e for e in out["estimates"] if not e.secondary)
    assert e.design_id == d.id and abs(e.value - 3) < 0.6
    assert {f.refuter for f in out["falsifications"]} == {"placebo_cutoffs"}
    assert "bandwidth_grid" in {s.method for s in out["sensitivities"]}
    assert "design" not in out["specialist_result"] and "refutations" not in out["specialist_result"]
    assert out["episode_visits"] and out["ledger"].history
    result = out["specialist_result"]
    assert result["dataset_identity"] == out["handoff"].dataset_identity
    assert result["context_admission"] and result["context_facts"]
    assert result["execution_attempts"] and result["reasoning_history"]
    assert result["terminal"]["kind"] == "valid_design"
    assert (Path(out["run_dir"]) / "result.json").is_file()


def test_no_first_stage_preserves_eligibility_endpoint(tmp_path):
    out = run_pack(admitted_pack(tmp_path, fuzzy=True, first_stage=False))
    assert out["specialist_result"]["status"] == "partial"
    assert out["lead_design_id"] is None and out["specialist_result"]["terminal"]["kind"] == "unidentified_effect"
    assert not out["lead_selection"]["requested_target_identified"]
    assert {d.endpoint_id for d in out["designs"].values()} == {"eligibility_local_polynomial"}
    rejected = next(c for c in out["candidates"] if c.endpoint_id == "fuzzy_local_polynomial")
    assert rejected.status == "blocked" and "first stage" in rejected.reason


def test_semantic_endpoints_reach_the_declared_freeze():
    from causal_agent.families.discontinuity.lane.knowledge import load_estimators

    assert len(CATALOGUE.endpoints) == 6
    assert not verify_catalogue(CATALOGUE, SPEC, {e.name for e in load_estimators()})
    assert all("freeze_designs" in e.trajectory for e in CATALOGUE.endpoints)
    assert {f.name for f in load_falsifications()} == {"placebo_cutoffs"}
    assert {"donut", "bandwidth_grid"} <= {s.name for s in load_sensitivities()}


def test_lead_selection_is_order_independent_and_uses_the_admitted_target_before_catalogue_rank(tmp_path):
    from causal_agent.common.contracts import Checks, Contrast
    from causal_agent.families.discontinuity.lane.contracts import Bandwidths, Covariates, Design, Score, ShapeFacts
    from causal_agent.families.discontinuity.lane.reason.selection import select_lead
    from causal_agent.lane.ledger import Entry, Evidence, Ledger

    h = admitted_pack(tmp_path, fuzzy=True, local_randomisation=True)

    def frozen(did, endpoint_id, context=None):
        context = context or h
        endpoint = CATALOGUE.endpoint(endpoint_id)
        locality = "window" if endpoint.configuration["framework"] == "local_randomisation" else "cutoff"
        return Design(
            id=did,
            endpoint_id=endpoint_id,
            context_id=context.context_id,
            ledger_revision=1,
            target={"contrast": "eligibility" if endpoint_id.startswith("eligibility") else "receipt", "locality": locality},
            contrast=Contrast(treated="treated", control="control", reason="frozen comparison", cites=[]),
            score=Score(column="x", cutoff=0, reason="the admitted rule", cites=["claim:assignment.cutoff"]),
            shape=ShapeFacts(
                rows_file=100,
                rows_score=100,
                rows_primary=100,
                rows_covariates=100,
                n_left=50,
                n_right=50,
                kind="fuzzy",
                distinct_scores=100,
                duplicate_share_left=0,
                duplicate_share_right=0,
                rows_at_cutoff=0,
                cutoff_shift=0,
                score_min=-1,
                score_max=1,
            ),
            covariates=Covariates(),
            checks=Checks(),
            estimator=endpoint.estimators[0],
            estimand=endpoint.targets[0],
            spec={},
            inference="randomisation" if locality == "window" else "robust_bc",
            vce="nn",
            bandwidths=Bandwidths(selector="declared", h_left=0.5, h_right=0.5, b_left=0.5, b_right=0.5, n_h_left=50, n_h_right=50),
            fuzzy=endpoint_id.startswith("fuzzy"),
            assumptions=endpoint.assumptions,
            table_fingerprint="declared-table",
            identification_certificate={"assumption_entry": f"rel:endpoint.{endpoint_id}"},
            support={"effective_rows_control": 50, "effective_rows_treated": 50},
            weights_policy={},
            selector_policy={},
        )

    designs = {
        "cutoff": frozen("cutoff", "fuzzy_local_polynomial"),
        "window": frozen("window", "fuzzy_local_randomisation"),
        "eligibility": frozen("eligibility", "eligibility_local_polynomial"),
    }
    ledger = Ledger(
        entries={
            f"rel:endpoint.{d.endpoint_id}": Entry(
                id=f"rel:endpoint.{d.endpoint_id}",
                kind="assumption",
                value=True,
                status="proposed",
                reason="conditional on the admitted study account",
                evidence=[Evidence(source="pack", address="claim:cutoff_only.believed")],
            )
            for d in designs.values()
        }
    )
    cutoff = select_lead(h, designs, ledger)
    assert cutoff == select_lead(h, dict(reversed(list(designs.items()))), ledger)
    assert cutoff["chosen_id"] == "cutoff" and cutoff["effect_blind"]
    assert next(r for r in cutoff["ranking"] if r["design_id"] == "window")["catalogue_rank"] == 0
    window_context = admit(
        h.model_copy(update={"facts": {**h.facts, "claim:target.locality": Fact(value="window", status="confirmed", source="user:turn:2")}}), CATALOGUE
    )
    assert select_lead(window_context, {did: frozen(did, d.endpoint_id, window_context) for did, d in designs.items()}, ledger)["chosen_id"] == "window"
    eligibility_context = admit(
        h.model_copy(update={"facts": {**h.facts, "claim:target.comparison": Fact(value="eligibility", status="confirmed", source="user:turn:2")}}), CATALOGUE
    )
    assert (
        select_lead(eligibility_context, {did: frozen(did, d.endpoint_id, eligibility_context) for did, d in designs.items()}, ledger)["chosen_id"]
        == "eligibility"
    )
    assert select_lead(h, {"eligibility": designs["eligibility"]}, ledger)["chosen_id"] is None


def test_continuity_support_preserves_the_source_of_metadata_and_diagnostics(tmp_path):
    from causal_agent.families.discontinuity.lane.reason.provenance import continuity_entry
    from causal_agent.lane.ledger import Ledger, apply

    h = admitted_pack(tmp_path)
    h.facts["probe:data.score_association"] = Fact(value=0.5, status="confirmed", source="data")
    for cites in (["col:x.when"], ["probe:data.score_association"], ["col:x.profile.numeric.mean"], ["claim:target.locality"]):
        entry = continuity_entry(h, Line(clean=True, why="conditional continuity remains a premise", cites=cites))
        assert entry.status == "proposed" and not any(e.category == "study_account" for e in entry.evidence)
        assert not apply(Ledger(), [entry], by="line")[1]
    data_entry = continuity_entry(h, Line(clean=True, why="conditional premise", cites=["probe:data.score_association"]))
    assert data_entry.evidence[0].source == "data" and data_entry.evidence[0].category == "diagnostic"
    account = continuity_entry(
        h, Line(clean=True, why="conditional on the declared account", cites=["claim:assignment.movable", "probe:data.score_association"])
    )
    assert account.status == "supported" and account.evidence[0].category == "study_account"
    assert not apply(Ledger(), [account], by="line")[1]


def test_adapter_pairs_the_robust_estimate_interval_and_standard_error():
    from rdrobust import rdrobust_RDsenate

    df = rdrobust_RDsenate()
    canon = pd.DataFrame({"y": df.vote, "x": df.margin, "cluster": df.state.astype(str)}).dropna(subset=["y", "x"])
    spec = {"p": 1, "kernel": "tri", "masspoints": "adjust", "level": 95}
    f = adapter.fit(spec, canon)
    assert f.error is None
    assert abs(f.value - (f.ci_low + f.ci_high) / 2) < 1e-9
    assert abs(f.value - 7.5065025) < 1e-4
    bw = adapter.bandwidths(spec, canon)
    same = adapter.fit(spec, canon, h=bw["h_mse"], b=bw["b_mse"])
    assert abs(same.value - f.value) < 1e-6
    clustered = adapter.fit(spec, canon, cluster=True, vce="cr2")
    assert clustered.error is None and clustered.vce == "CR2"
