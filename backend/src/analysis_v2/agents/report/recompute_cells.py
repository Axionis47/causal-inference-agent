"""Source for the notebook's inline-recompute code cells.

These are Python source strings, not run at build time. They reference the
globals the load cell binds (``df``, ``PLAN``, ``SPEC``, ``DAG``, plus ``pd``
and ``np``) and re-run the exact deterministic lane the pipeline used, so the
notebook's numbers match the app's without echoing stored JSON. There is no
self-check assert: the estimate is recomputed, not compared to a frozen copy.
"""
from __future__ import annotations

# A per-column overview of the loaded frame: dtype, missing fraction, and
# cardinality, recomputed from df rather than read from a stored profile.
PROFILE = """\
summary = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'missing': df.isna().mean().round(3),
    'n_unique': df.nunique(),
})
print(f'{len(df):,} rows x {len(df.columns)} columns')
summary"""


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


# Covariate balance: the matching lane reports before/after SMDs directly;
# every other design gets the raw before-treatment SMD computed from the frame.
# Runs after ESTIMATE so the matching balance table is available on `outcome`.
BALANCE = """\
treat_col = PLAN.get('treatment')
covs = [c for c in (PLAN.get('covariates') or []) if c in df.columns]
BALANCE = None
for art in getattr(outcome, 'artifacts', []):
    if art.name == 'balance_after_matching' and art.kind == 'table':
        BALANCE = pd.DataFrame(art.payload)
        break
if BALANCE is None and covs and treat_col in df.columns and df[treat_col].nunique() == 2:
    g = sorted(df[treat_col].dropna().unique())
    def _smd(col):
        a = pd.to_numeric(df[df[treat_col] == g[1]][col], errors='coerce').dropna()
        b = pd.to_numeric(df[df[treat_col] == g[0]][col], errors='coerce').dropna()
        pooled = np.sqrt((a.var() + b.var()) / 2)
        return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
    BALANCE = pd.DataFrame([{'covariate': c, 'smd_before': _smd(c)} for c in covs])
if BALANCE is None:
    BALANCE = pd.DataFrame(columns=['covariate', 'smd_before'])
BALANCE"""


# Re-run the design's diagnostic and sensitivity checks through the same
# dispatch the S8 agent uses (compose.run_lane_checks), so the notebook cannot
# drift from the pipeline. Leakage is run explicitly, as the agent does. Binds
# DIAG and SENS for the sensitivity verdict cell.
DIAGNOSTICS = """\
from src.analysis_v2.agents.diagnostics import checks as C
from src.analysis_v2.agents.diagnostics.compose import run_lane_checks
from src.analysis_v2.agents.method_lane.lanes import LANES
from src.analysis_v2.spec import MethodLane

runner = LANES[MethodLane(PLAN['lane'])]
leakage = C.detect_leakage(df, plan_model)
lane_diag, SENS = run_lane_checks(df, plan_model, spec_model, RESULT, runner)
DIAG = [leakage, *lane_diag]
diag_rows = [
    {'check': c.name, 'status': c.status.value, 'detail': c.detail} for c in DIAG
]
pd.DataFrame(diag_rows)"""


# The robustness verdict mapped from the worst check status by the same rubric
# the agent applies, plus the sensitivity check table.
SENSITIVITY = """\
from src.analysis_v2.agents.diagnostics.compose import rubric

robustness, reason = rubric(DIAG, SENS)
print('robustness verdict:', robustness.value, '-', reason)
sens_rows = [
    {'check': c.name, 'status': c.status.value, 'detail': c.detail} for c in SENS
]
pd.DataFrame(sens_rows) if sens_rows else None"""
