"""Source for the notebook's inline-recompute code cells.

These are Python source strings, not run at build time. They reference the
globals the load cell binds (``df``, ``PLAN``, ``SPEC``, ``DAG``, plus ``pd``
and ``np``) and re-run the exact deterministic lane the pipeline used, so the
notebook's numbers match the app's without echoing stored JSON. There is no
self-check assert: the estimate is recomputed, not compared to a frozen copy.
"""
from __future__ import annotations

# Re-run the chosen lane on the loaded frame; bind RESULT and a tidy effects
# frame for the estimate, conclusion, and forest-plot cells.
ESTIMATE = """\
from src.analysis_v2.agents.method_lane.lanes import LANES
from src.analysis_v2.spec import CausalSpec, MethodLane, MethodPlan

plan_model = MethodPlan.model_validate(PLAN)
spec_model = CausalSpec.model_validate(SPEC)
outcome = LANES[MethodLane(PLAN['lane'])](df, plan_model, spec_model)
RESULT = outcome.result
effects = pd.DataFrame([e.model_dump() for e in RESULT.effects])
print(f"{RESULT.estimator} on {RESULT.n_rows_used:,} rows")
effects[['estimand', 'estimate', 'std_error', 'ci_lower', 'ci_upper', 'p_value']]"""
