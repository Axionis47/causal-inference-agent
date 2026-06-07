"""End-to-end runs of the deterministic spine with stub specialists.

These drive the real execute() with lightweight stubs to pin the order the
orchestrator now follows: required agents run in the fixed sequence, optional
agents are skipped when the data does not call for them, the run completes, and
a fatal brief mid-spine halts before estimation. No LLM, no router.
"""
from __future__ import annotations

import pytest

from src.analysis.agents import CritiqueDecision, CritiqueFeedback
from src.analysis.agents.base import DataProfile
from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst.output import SensitivityResult
from src.analysis.orchestrator.standard.agent import StandardOrchestrator
from src.domain.approval import HumanApproval
from src.domain.briefs import AgentBrief, Flag


class _Stub:
    """Specialist stub: records its dispatch, sets outputs, seals a brief."""

    REQUIRED_STATE_FIELDS: tuple[str, ...] = ()

    def __init__(self, name, order, sets=None, flags=None, on_execute=None):
        self._name = name
        self._order = order
        self._sets = sets or {}
        self._flags = flags or []
        self._on = on_execute

    async def execute_with_tracing(self, state: AnalysisState) -> AnalysisState:
        self._order.append(self._name)
        for attr, value in self._sets.items():
            setattr(state, attr, value)
        if self._on is not None:
            self._on(state)
        # critique is a decision agent, not subject to the readiness gate.
        if self._name != "critique":
            state.agent_briefs[self._name] = AgentBrief(
                agent=self._name, status="done", headline=f"{self._name} ok", flags=self._flags
            )
        return state


def _state() -> AnalysisState:
    # No metadata and no treatment/outcome pair, so domain_knowledge and
    # ps_diagnostics are skipped. Approved up front so the data gate does not park.
    s = AnalysisState(job_id="job-1", dataset_info=DatasetInfo(url="kaggle.com/x"))
    s.human_approval = HumanApproval.approve()
    return s


def _profile() -> DataProfile:
    return DataProfile(
        n_samples=100, n_features=3, feature_names=["t", "y", "x"],
        feature_types={"t": "binary"}, missing_values={"t": 0},
        numeric_stats={}, categorical_stats={},
        treatment_candidates=["t"], outcome_candidates=["y"],
    )


def _effect() -> TreatmentEffectResult:
    return TreatmentEffectResult(method="OLS", estimand="ATE", estimate=1.0,
                                 std_error=0.1, ci_lower=0.8, ci_upper=1.2)


def _sens() -> SensitivityResult:
    return SensitivityResult(method="evalue", robustness_value=2.0, interpretation="robust")


def _approve() -> CritiqueFeedback:
    return CritiqueFeedback(
        decision=CritiqueDecision.APPROVE, iteration=1,
        scores={"statistical_validity": 5, "assumption_checking": 5,
                "method_selection": 5, "completeness": 5,
                "reproducibility": 5, "interpretation": 5},
        issues=[], improvements=[], reasoning="looks good",
    )


def _register_backbone(orch: StandardOrchestrator, order: list, dag_flags=None) -> None:
    orch.register_specialist("data_profiler", _Stub(
        "data_profiler", order, sets={"data_profile": _profile(), "dataframe_path": "/tmp/df.parquet"}))
    orch.register_specialist("eda_agent", _Stub("eda_agent", order, sets={"eda_result": {"ok": True}}))
    orch.register_specialist("causal_discovery", _Stub(
        "causal_discovery", order, sets={"discovered_dag": {"nodes": ["t", "y"]}}))
    orch.register_specialist("dag_expert", _Stub(
        "dag_expert", order, sets={"refined_dag": {"nodes": ["t", "y"]}}, flags=dag_flags))
    orch.register_specialist("effect_estimator", _Stub(
        "effect_estimator", order, sets={"treatment_effects": [_effect()]}))
    orch.register_specialist("sensitivity_analyst", _Stub(
        "sensitivity_analyst", order, sets={"sensitivity_results": [_sens()]}))
    orch.register_specialist("critique", _Stub(
        "critique", order, on_execute=lambda s: s.critique_history.append(_approve())))
    orch.register_specialist("notebook_generator", _Stub(
        "notebook_generator", order, sets={"notebook_path": "/tmp/nb.ipynb"}))


@pytest.mark.asyncio
async def test_full_run_follows_the_deterministic_spine_to_completion():
    order: list[str] = []
    orch = StandardOrchestrator()
    _register_backbone(orch, order)
    # Register the optional agents too (no outputs): if a skip rule wrongly ran
    # one, it would append to `order`, so the "not in order" checks below become
    # precise guards rather than relying on the agent simply not existing.
    for opt in ("domain_knowledge", "data_repair", "confounder_discovery", "ps_diagnostics"):
        orch.register_specialist(opt, _Stub(opt, order))

    result = await orch.execute(_state())

    assert result.status == JobStatus.COMPLETED
    # Optional agents the data did not call for were skipped.
    for skipped in ("domain_knowledge", "data_repair", "confounder_discovery", "ps_diagnostics"):
        assert skipped not in order
    # The backbone ran in the fixed order, with eda/discovery as the parallel pair.
    assert order[0] == "data_profiler"
    assert set(order[1:3]) == {"eda_agent", "causal_discovery"}
    assert order[3:] == [
        "dag_expert", "effect_estimator", "sensitivity_analyst",
        "critique", "notebook_generator",
    ]


@pytest.mark.asyncio
async def test_a_cyclic_dag_halts_the_spine_before_estimation():
    order: list[str] = []
    orch = StandardOrchestrator()
    _register_backbone(orch, order, dag_flags=[Flag.CYCLE_DETECTED])

    result = await orch.execute(_state())

    assert result.status == JobStatus.FAILED
    assert "cycle_detected" in (result.error_message or "")
    # The run stopped at the DAG; estimation and everything after never ran.
    assert "dag_expert" in order
    assert "effect_estimator" not in order
    assert "notebook_generator" not in order
