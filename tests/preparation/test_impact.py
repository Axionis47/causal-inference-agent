"""Dimension impact and per-method structure validation (T-016 §1.2; PRD-003 §9.5, §12)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from causal.preparation import impact as im
from causal.preparation.packs import PreparationPackV1, load_preparation_packs
from causal.preparation.stabilize import StabilizationError

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS = {
    row.method_id: row for row in load_preparation_packs(
        REGISTRIES / "method-pack-preparation.v1.json", REGISTRIES / "method-packs.v1.json").all()
}
UNITS = [f"u{index}" for index in range(20)]

RCT = pl.DataFrame({"unit": UNITS, "arm": [index % 2 for index in range(20)], "y": [1.0] * 20})
AIPW = pl.DataFrame({"unit": UNITS, "treat": [index % 2 for index in range(20)],
                     "y": [float(index) for index in range(20)]})
DID = pl.DataFrame({"unit": [f"u{index // 4}" for index in range(20)],
                    "grp": [(index // 4) % 2 for index in range(20)],
                    "t": [index % 4 for index in range(20)], "y": [1.0] * 20})
RDD = pl.DataFrame({"unit": UNITS, "run": [float(index) for index in range(20)], "y": [1.0] * 20})

RCT_SPEC = im.MethodStructureSpecV1(unit_columns=("unit",), treatment_column="arm",
                                    minimum_cell_rows=2)
AIPW_SPEC = im.MethodStructureSpecV1(unit_columns=("unit",), treatment_column="treat",
                                     outcome_column="y")
DID_SPEC = im.MethodStructureSpecV1(unit_columns=("unit",), group_column="grp", time_column="t",
                                    threshold=2, minimum_cell_rows=2)
RDD_SPEC = im.MethodStructureSpecV1(unit_columns=("unit",), running_variable_column="run",
                                    threshold=10.0, minimum_cell_rows=2)

STRUCTURE = (
    ("randomized_experiment", RCT, RCT_SPEC, im.StructureVerdict.RUNNABLE, None),
    ("randomized_experiment", RCT.head(3), RCT_SPEC, im.StructureVerdict.NOT_RUNNABLE,
     im.BELOW_MINIMUM_ROWS),
    ("randomized_experiment", RCT.filter(pl.col("arm") == 1), RCT_SPEC,
     im.StructureVerdict.CONFLICT, im.ARM_DESTROYED),
    ("aipw", AIPW, AIPW_SPEC, im.StructureVerdict.RUNNABLE, None),
    ("aipw", AIPW.head(10), AIPW_SPEC, im.StructureVerdict.NOT_RUNNABLE, im.BELOW_MINIMUM_UNITS),
    ("aipw", pl.concat([AIPW, AIPW.head(1)]), AIPW_SPEC, im.StructureVerdict.CONFLICT,
     im.UNIT_GRAIN_VIOLATED),
    ("did", DID, DID_SPEC, im.StructureVerdict.RUNNABLE, None),
    ("did", DID, DID_SPEC.model_copy(update={"minimum_cell_rows": 3}),
     im.StructureVerdict.NOT_RUNNABLE, "group_time_cell_min_rows"),
    ("did", DID.filter(~((pl.col("grp") == 1) & (pl.col("t") == 3))), DID_SPEC,
     im.StructureVerdict.CONFLICT, im.GROUP_TIME_CELL_EMPTIED),
    ("sharp_rdd", RDD, RDD_SPEC, im.StructureVerdict.RUNNABLE, None),
    ("sharp_rdd", RDD, RDD_SPEC.model_copy(update={"minimum_cell_rows": 11}),
     im.StructureVerdict.NOT_RUNNABLE, "cutoff_side_min_rows"),
    ("sharp_rdd", RDD.filter(pl.col("run") < 10), RDD_SPEC, im.StructureVerdict.CONFLICT,
     im.CUTOFF_SIDE_EMPTIED),
)


@pytest.mark.parametrize(("method", "frame", "spec", "verdict", "code"), STRUCTURE)
def test_each_pack_validates_its_required_structure(
    method: str, frame: pl.DataFrame, spec: im.MethodStructureSpecV1,
    verdict: im.StructureVerdict, code: str | None
) -> None:
    result = im.validate_structure(frame, spec, PACKS[method])
    assert result.verdict is verdict
    assert (code in result.codes) if code else result.codes == ()
    assert result.status == ("pass" if verdict is im.StructureVerdict.RUNNABLE else "fail")


def test_a_method_without_registered_structure_gates_fails_closed() -> None:
    stray: PreparationPackV1 = PACKS["aipw"].model_copy(update={"method_id": "hand_rolled"})
    with pytest.raises(StabilizationError):
        im.validate_structure(AIPW, AIPW_SPEC, stray)


RETAINED = [index >= 4 for index in range(20)]
DIMENSIONS = (
    im.DimensionSpecV1(dimension_id="overall", kind=im.DimensionKind.OVERALL),
    im.DimensionSpecV1(dimension_id="treatment_group", kind=im.DimensionKind.COLUMN_LEVELS,
                       column="treat"),
    im.DimensionSpecV1(dimension_id="period", kind=im.DimensionKind.PRE_POST, column="y",
                       threshold=10.0),
    im.DimensionSpecV1(dimension_id="cutoff_side", kind=im.DimensionKind.CUTOFF_SIDE, column="y",
                       threshold=4.0),
)


def test_impact_reports_retained_and_excluded_counts_for_every_dimension() -> None:
    overall, arm, period, side = im.dimension_impact(AIPW, RETAINED, DIMENSIONS)
    assert overall.retained_by_level == {"all": 16}
    assert overall.excluded_by_level == {"all": 4}
    assert arm.retained_by_level == {"0": 8, "1": 8}
    assert arm.excluded_by_level == {"0": 2, "1": 2}
    assert period.retained_by_level == {"pre": 6, "post": 10}
    assert side.excluded_by_level == {"below_cutoff": 4, "at_or_above_cutoff": 0}


def test_a_dimension_whose_level_is_wholly_excluded_warns() -> None:
    kept = [value >= 10.0 for value in AIPW.get_column("y").to_list()]
    (report,) = im.dimension_impact(AIPW, kept, (DIMENSIONS[3],))
    assert report.retained_by_level == {"below_cutoff": 0, "at_or_above_cutoff": 10}
    assert report.warning_codes == (f"{im.LEVEL_EMPTIED}:below_cutoff",)


def test_a_null_level_is_reported_rather_than_dropped() -> None:
    frame = AIPW.with_columns(pl.when(pl.col("treat") == 1).then(None).otherwise(pl.col("treat"))
                              .alias("treat"))
    (report,) = im.dimension_impact(frame, [True] * 20, (DIMENSIONS[1],))
    assert report.retained_by_level == {"0": 10, im.NULL_LEVEL: 10}


@pytest.mark.parametrize("spec", [
    im.DimensionSpecV1(dimension_id="ghost", kind=im.DimensionKind.COLUMN_LEVELS, column="absent"),
    im.DimensionSpecV1(dimension_id="period", kind=im.DimensionKind.PRE_POST, column="y"),
])
def test_a_dimension_the_frame_cannot_answer_fails_closed(spec: im.DimensionSpecV1) -> None:
    with pytest.raises(StabilizationError):
        im.dimension_impact(AIPW, [True] * 20, (spec,))
