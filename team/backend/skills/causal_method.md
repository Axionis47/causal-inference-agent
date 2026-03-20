# Skill: Write Causal Estimation Method

Load when: adding or modifying causal estimation methods in backend/src/causal/methods/.

## Method contract

Every method extends BaseCausalMethod:

```python
from .base import BaseCausalMethod, MethodResult, register_method


@register_method("my_method")
class MyMethod(BaseCausalMethod):
    METHOD_NAME = "my_method"
    ESTIMAND = "ATE"  # ATE, ATT, CATE, or LATE

    def fit(self, df, treatment_col, outcome_col, covariates=None, **kwargs):
        treatment = self._binarize_treatment(df[treatment_col])
        outcome = df[outcome_col].values.astype(float)
        # ... estimation logic ...
        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=estimate,
            std_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_treated=int(treatment.sum()),
            n_control=int((1 - treatment).sum()),
        )
        self._fitted = True
        return self

    def estimate(self):
        if not self._fitted:
            raise ValueError("Must call fit() before estimate()")
        return self._result
```

## Rules

1. Always use @register_method decorator
2. Implement fit() and estimate() — two-step interface
3. Return MethodResult with all fields populated
4. Use helper methods from BaseCausalMethod: _binarize_treatment, _compute_ci, _compute_p_value
5. Implement validate_assumptions() for method-specific checks
6. Import in methods/__init__.py to trigger registration
7. Handle edge cases: insufficient samples, singular matrices, convergence failures
8. Never raise unhandled exceptions — return MethodResult with NaN estimates on failure

## MethodResult fields

```python
@dataclass
class MethodResult:
    method: str          # method name
    estimand: str        # ATE, ATT, CATE, LATE
    estimate: float      # point estimate
    std_error: float     # standard error
    ci_lower: float      # lower CI bound
    ci_upper: float      # upper CI bound
    p_value: float       # p-value
    n_treated: int       # treatment group size
    n_control: int       # control group size
```

## Integration with pipeline

Methods are called through EffectEstimatorEngine.run_method_safe(), which bridges
MethodResult to the pipeline's TreatmentEffectResult contract. You only need to
implement the BaseCausalMethod interface — the bridge handles the rest.
