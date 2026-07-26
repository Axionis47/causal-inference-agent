"""What every lane returns, and what it raises when it can't run."""
from __future__ import annotations

from dataclasses import dataclass, field


class LaneError(ValueError):
    """The data cannot support this design. The message names the reason."""


@dataclass
class Estimate:
    """One causal effect estimate.

    `value` is on the scale the estimand implies: a difference in the outcome's
    units for att/ate/late, a ratio for hazard_ratio. `ci_low`/`ci_high` are the
    95% interval and are None only when the estimator cannot produce one.
    """

    estimand: str  # att | ate | late | hazard_ratio | indirect | level_shift | ...
    value: float
    n: int
    estimator: str
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        ci = ""
        if self.ci_low is not None and self.ci_high is not None:
            ci = f" [{self.ci_low:.4g}, {self.ci_high:.4g}]"
        return f"{self.estimand}={self.value:.4g}{ci} (n={self.n}, {self.estimator})"
