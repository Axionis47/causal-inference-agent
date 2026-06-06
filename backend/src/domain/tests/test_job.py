"""JobInput / BudgetSpec / UserContext: the analysis entry contract."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.job import (
    BudgetSpec,
    JobInput,
    UserContext,
    budget_for_tier,
)


def test_budget_spec_defaults_are_sane():
    b = BudgetSpec()
    assert b.max_agent_loops == 20
    assert b.max_tool_calls_per_agent == 30
    assert b.max_total_tool_calls == 200
    assert b.max_llm_tokens == 500_000
    assert b.wall_clock_seconds == 3000


def test_budget_spec_is_frozen():
    b = BudgetSpec()
    with pytest.raises(ValidationError):
        b.max_agent_loops = 999


def test_budget_spec_rejects_zero_and_negative():
    with pytest.raises(ValidationError):
        BudgetSpec(max_agent_loops=0)
    with pytest.raises(ValidationError):
        BudgetSpec(wall_clock_seconds=-1)


@pytest.mark.parametrize("tier", ["quick", "standard", "deep"])
def test_budget_tier_returns_valid_spec(tier):
    spec = budget_for_tier(tier)
    assert isinstance(spec, BudgetSpec)


def test_budget_tier_quick_is_tighter_than_deep():
    quick = budget_for_tier("quick")
    deep = budget_for_tier("deep")
    assert quick.max_agent_loops < deep.max_agent_loops
    assert quick.max_total_tool_calls < deep.max_total_tool_calls
    assert quick.wall_clock_seconds < deep.wall_clock_seconds


def test_user_context_defaults_are_empty():
    u = UserContext()
    assert u.notes is None
    assert u.hypothesis is None
    assert u.known_confounders == []
    assert u.immutable_vars == []
    assert u.skip_stages == []


def test_user_context_round_trip():
    u = UserContext(
        notes="binary treatment, observational",
        hypothesis="fertilizer increases yield",
        known_confounders=["soil_ph", "rainfall"],
        immutable_vars=["region"],
        skip_stages=["sensitivity"],
    )
    parsed = UserContext.model_validate(u.model_dump())
    assert parsed == u


def test_job_input_minimal_construction():
    job = JobInput(job_id="abc123", download_id="dl_xyz")
    assert job.orchestrator_mode == "standard"
    assert job.treatment_variable is None
    assert isinstance(job.budget, BudgetSpec)
    assert isinstance(job.user_context, UserContext)


def test_job_input_rejects_empty_ids():
    with pytest.raises(ValidationError):
        JobInput(job_id="", download_id="dl_xyz")
    with pytest.raises(ValidationError):
        JobInput(job_id="abc", download_id="")


def test_job_input_rejects_unknown_orchestrator():
    with pytest.raises(ValidationError):
        JobInput(
            job_id="abc",
            download_id="dl",
            orchestrator_mode="opus_mode",  # type: ignore[arg-type]
        )


def test_job_input_rejects_extra_fields():
    with pytest.raises(ValidationError):
        JobInput.model_validate(
            {
                "job_id": "abc",
                "download_id": "dl",
                "secret_backdoor": True,
            }
        )


def test_job_input_round_trip_with_full_context():
    job = JobInput(
        job_id="abc123",
        download_id="dl_xyz",
        treatment_variable="fertilizer",
        outcome_variable="yield",
        orchestrator_mode="react",
        budget=budget_for_tier("deep"),
        user_context=UserContext(
            notes="binary treatment",
            known_confounders=["soil_ph"],
        ),
    )
    parsed = JobInput.model_validate(job.model_dump())
    assert parsed == job


