"""Independent numerical oracles for future generated-program comparisons.

Passing these tests verifies the fixtures and their transformations. It does not
establish model reasoning, application execution, or causal identification.
No production estimator code is imported here.
"""

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

import pytest


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    protocol: str
    expected: Mapping[str, float]

    def csv_bytes(
        self,
        *,
        names: Mapping[str, str] | None = None,
        order: Sequence[str] | None = None,
        padding: int = 0,
    ) -> bytes:
        """Preserve observations while applying an explicit schema mapping."""
        names = names if names is not None else {key: key for key in self.columns}
        order = tuple(order) if order is not None else self.columns
        if (
            set(names) != set(self.columns)
            or set(order) != set(self.columns)
            or len(order) != len(self.columns)
        ):
            raise ValueError("A schema transformation must cover every source column exactly once.")
        if padding < 0:
            raise ValueError("Padding must be nonnegative.")
        header = [names[key] for key in order] + [f"bookkeeping_{i}" for i in range(padding)]
        if len(set(header)) != len(header):
            raise ValueError("A schema transformation must preserve unique column bindings.")
        positions = [self.columns.index(key) for key in order]
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(header)
        for index, row in enumerate(self.rows):
            writer.writerow(
                [row[position] for position in positions]
                + [(index * 17 + i * 13) % 997 for i in range(padding)]
            )
        return output.getvalue().encode()


def reference_cases() -> tuple[ReferenceCase, ...]:
    factorial = tuple(
        (
            float(a * 16 + b * 8 + replicate),
            float(a),
            float(b),
            1.0 + 2 * a + 3 * b + 4 * a * b + (-1 if replicate % 2 else 1),
        )
        for a in (0, 1)
        for b in (0, 1)
        for replicate in range(8)
    )
    adoption = tuple(
        (
            float(cohort * 10 + unit),
            float(cohort),
            float(period),
            float(cohort > 0 and period >= cohort),
            float(
                10
                + unit
                + cohort
                + 3 * period
                + (5 + 2 * (period - cohort) if cohort > 0 and period >= cohort else 0)
            ),
        )
        for cohort in (0, 2, 3)
        for unit in range(4)
        for period in (0, 1, 2, 3)
    )
    sharp = tuple(
        (
            float(index * 8 + unit),
            float(score),
            float(score >= 0),
            2.0 + 3 * score + (4 + 1.5 * score if score >= 0 else 0) + (-0.5 if unit % 2 else 0.5),
        )
        for index, score in enumerate((-3, -2, -1, 1, 2, 3))
        for unit in range(8)
    )
    fuzzy = tuple(
        (
            float(index * 10 + unit),
            float(score),
            float(score >= 0),
            float(unit < takeup),
            2.0 + 0.5 * score + 2 * (unit < takeup) + (-0.25 if unit % 2 else 0.25),
        )
        for index, (score, takeup) in enumerate(((-2, 2), (-1, 3), (1, 8), (2, 9)))
        for unit in range(10)
    )
    return (
        ReferenceCase(
            "factorial_randomized",
            ("unit", "factor_a", "factor_b", "outcome"),
            factorial,
            "Two independently randomized binary factors, each with levels 0 and 1. "
            "Use unadjusted arithmetic means of the four cells. Primary a_marginal is "
            "the effect of A, averaging its two within-B contrasts with equal weights. "
            "Secondary b_marginal analogously averages the within-A contrasts. "
            "interaction is (mu11-mu01)-(mu10-mu00). joint is mu11-mu00. "
            "Include every row. This fixture prespecifies point estimates only.",
            {"a_marginal": 4.0, "b_marginal": 5.0, "interaction": 4.0, "joint": 9.0},
        ),
        ReferenceCase(
            "absorbing_adoption",
            ("unit", "adoption", "period", "treated", "outcome"),
            adoption,
            "Balanced panel. Adoption 0 means never treated; cohorts 2 and 3 adopt "
            "at their named periods and remain treated. For each (g,t) in (2,2), "
            "(2,3), (3,3), estimate the treated cohort's mean change from g-1 to t "
            "minus the never-treated units' mean change over those same periods. "
            "Never use already-treated or not-yet-treated units as controls. "
            "Primary equal_group_time_att averages these three effects equally. "
            "Also return g2_t2, g2_t3 and g3_t3. This fixture prespecifies point "
            "estimates under no anticipation and parallel trends only.",
            {"g2_t2": 5.0, "g2_t3": 7.0, "g3_t3": 5.0, "equal_group_time_att": 17 / 3},
        ),
        ReferenceCase(
            "sharp_discontinuity",
            ("unit", "running", "assigned", "outcome"),
            sharp,
            "Cutoff is zero and assignment is 1[running >= 0]. Fit separate "
            "weighted local linear regressions of outcome on intercept and running "
            "on each side, using triangular weights max(0,1-abs(running)/4). "
            "Primary cutoff_jump is the right intercept minus the left intercept. "
            "Bandwidth 4 is fixed; do not select another bandwidth or polynomial. "
            "This fixture prespecifies a point estimate only.",
            {"cutoff_jump": 4.0},
        ),
        ReferenceCase(
            "fuzzy_discontinuity",
            ("unit", "running", "assigned", "received", "outcome"),
            fuzzy,
            "Cutoff is zero. Assignment is 1[running >= 0], distinct from binary "
            "treatment received. Fit separate weighted local linear regressions "
            "on each side for outcome and received, using triangular weights "
            "max(0,1-abs(running)/3). Freeze bandwidth 3. Let each jump be right "
            "intercept minus left intercept. Primary local_wald is outcome_jump "
            "divided by treatment_jump. Return both component jumps as well. "
            "Fail for a zero first stage. This point-estimate fixture assumes "
            "continuity, exclusion and monotonicity; it does not establish them.",
            {"outcome_jump": 0.6, "treatment_jump": 0.3, "local_wald": 2.0},
        ),
    )


def _intercept(rows: list[dict[str, float]], outcome: str, bandwidth: float) -> float:
    weighted = [(row["running"], row[outcome], max(0.0, 1 - abs(row["running"]) / bandwidth)) for row in rows]
    s0 = sum(weight for _, _, weight in weighted)
    sx = sum(weight * x for x, _, weight in weighted)
    sy = sum(weight * y for _, y, weight in weighted)
    sxx = sum(weight * x * x for x, _, weight in weighted)
    sxy = sum(weight * x * y for x, y, weight in weighted)
    denominator = s0 * sxx - sx * sx
    if denominator <= 0:
        raise ValueError("Each side needs positive weighted support at distinct running-variable values.")
    return (sxx * sy - sx * sxy) / denominator


def reference_estimates(
    case: ReferenceCase,
    source: bytes | None = None,
    *,
    names: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Compute the fixed protocol without importing the application or numerical libraries."""
    names = names if names is not None else {key: key for key in case.columns}
    reader = csv.DictReader(io.StringIO((source if source is not None else case.csv_bytes()).decode()))
    rows = [{key: float(row[names[key]]) for key in case.columns} for row in reader]
    if case.name == "factorial_randomized":
        means = {
            (a, b): fmean(row["outcome"] for row in rows if row["factor_a"] == a and row["factor_b"] == b)
            for a in (0, 1)
            for b in (0, 1)
        }
        return {
            "a_marginal": ((means[1, 0] - means[0, 0]) + (means[1, 1] - means[0, 1])) / 2,
            "b_marginal": ((means[0, 1] - means[0, 0]) + (means[1, 1] - means[1, 0])) / 2,
            "interaction": means[1, 1] - means[0, 1] - means[1, 0] + means[0, 0],
            "joint": means[1, 1] - means[0, 0],
        }
    if case.name == "absorbing_adoption":
        means = {
            (cohort, period): fmean(
                row["outcome"] for row in rows if row["adoption"] == cohort and row["period"] == period
            )
            for cohort in (0, 2, 3)
            for period in (0, 1, 2, 3)
        }
        effects = {
            f"g{g}_t{t}": means[g, t] - means[g, g - 1] - means[0, t] + means[0, g - 1]
            for g, t in ((2, 2), (2, 3), (3, 3))
        }
        return {**effects, "equal_group_time_att": fmean(effects.values())}
    bandwidth = 4.0 if case.name == "sharp_discontinuity" else 3.0

    def jump(outcome: str) -> float:
        return _intercept([row for row in rows if row["running"] >= 0], outcome, bandwidth) - _intercept(
            [row for row in rows if row["running"] < 0], outcome, bandwidth
        )

    outcome_jump = jump("outcome")
    if case.name == "sharp_discontinuity":
        return {"cutoff_jump": outcome_jump}
    if case.name != "fuzzy_discontinuity":
        raise ValueError("Unknown fixed reference protocol.")
    treatment_jump = jump("received")
    if abs(treatment_jump) < 1e-12:
        raise ValueError("The fixed fuzzy discontinuity protocol requires a nonzero first stage.")
    return {
        "outcome_jump": outcome_jump,
        "treatment_jump": treatment_jump,
        "local_wald": outcome_jump / treatment_jump,
    }


@pytest.mark.parametrize("case", reference_cases(), ids=lambda case: case.name)
def test_independent_reference_values(case: ReferenceCase) -> None:
    assert reference_estimates(case) == pytest.approx(case.expected, abs=1e-10, rel=0)


@pytest.mark.parametrize("case", reference_cases(), ids=lambda case: case.name)
@pytest.mark.parametrize("padding", [0, 48, 188])
def test_fixed_reference_is_invariant_to_explicit_schema_transforms(
    case: ReferenceCase, padding: int
) -> None:
    names = {key: f"v_{(index * 37 + 19) % 101:03}" for index, key in enumerate(case.columns)}
    transformed = case.csv_bytes(names=names, order=tuple(reversed(case.columns)), padding=padding)
    assert reference_estimates(case, transformed, names=names) == pytest.approx(
        case.expected, abs=1e-10, rel=0
    )


def test_fuzzy_reference_rejects_zero_first_stage() -> None:
    case = reference_cases()[-1]
    rows = tuple((*row[:3], float(index % 2), row[4]) for index, row in enumerate(case.rows))
    zero_stage = ReferenceCase(case.name, case.columns, rows, case.protocol, {})
    with pytest.raises(ValueError, match="nonzero first stage"):
        reference_estimates(zero_stage)


def test_reference_mapping_rejects_column_collapse() -> None:
    case = reference_cases()[0]
    with pytest.raises(ValueError, match="unique column bindings"):
        case.csv_bytes(names={column: "same" for column in case.columns})
