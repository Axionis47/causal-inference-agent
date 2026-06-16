"""DAG-consistency check: test what the causal graph implies against the data.

The CausalDAG built at S3 entails conditional independencies (its local Markov
set): each node is independent of its non-descendants given its parents. Each
is a hypothesis the data can refute. This check measures the (partial)
correlation behind every implication it can evaluate on numeric, complete data
and flags the ones in tension with the assumed structure. It is descriptive: a
large partial correlation means the graph and the data disagree, never that one
variable causes another.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.analysis_v2.spec import EDACheck, EDACheckStatus

from .common import ArtifactSink, EDAInputs, failed, skipped

logger = structlog.get_logger(__name__)

MAX_IMPLICATIONS = 30  # dense graphs imply many; cap to keep the check bounded
MIN_ROWS = 30  # below this a partial correlation is too noisy to read
TENSION = 0.1  # |partial r| above this reads as tension with the graph


def _numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame | None:
    """Coerce the involved columns to numeric and drop incomplete rows; None
    when nothing survives (e.g. an unencoded categorical conditioning set)."""
    block = frame[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    return block if not block.empty else None


def _residual(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual: np.ndarray = target - design @ coef
    return residual


def _partial_corr(block: pd.DataFrame, x: str, y: str, given: list[str]) -> float:
    """Correlation of x and y after linearly removing the conditioning set;
    with no conditioning set this is the plain Pearson correlation."""
    xv = block[x].to_numpy(float)
    yv = block[y].to_numpy(float)
    if given:
        design = np.column_stack([np.ones(len(block)), block[given].to_numpy(float)])
        xv, yv = _residual(design, xv), _residual(design, yv)
    if xv.std() == 0 or yv.std() == 0:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def dag_consistency(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    dag = inputs.dag
    if dag is None:
        return skipped("dag_consistency", "no causal graph to test")
    implications = sorted(
        dag.testable_implications(),
        key=lambda t: (t[0], t[1], tuple(sorted(t[2]))),
    )
    if not implications:
        return skipped("dag_consistency", "the graph implies no testable independencies")
    total = len(implications)
    truncated = total > MAX_IMPLICATIONS
    if truncated:
        logger.warning("dag_consistency_truncated", total=total, tested=MAX_IMPLICATIONS)
        implications = implications[:MAX_IMPLICATIONS]
    try:
        rows: list[dict] = []
        not_testable = 0
        for x, y, given in implications:
            given_list = sorted(given)
            cols = [x, y, *given_list]
            block = _numeric(inputs.frame, cols) if all(
                c in inputs.frame.columns for c in cols
            ) else None
            if block is None or len(block) < MIN_ROWS:
                not_testable += 1
                continue
            if block[x].nunique() < 2 or block[y].nunique() < 2:
                not_testable += 1
                continue
            r = _partial_corr(block, x, y, given_list)
            rows.append({
                "x": x, "y": y, "given": ", ".join(given_list) or "—",
                "partial_corr": round(r, 4), "n": int(len(block)),
                "verdict": "tension" if abs(r) > TENSION else "consistent",
            })
        if not rows:
            return skipped(
                "dag_consistency",
                f"none of {total} implied independencies were testable on "
                "numeric, complete data",
            )
        table = pd.DataFrame(rows).sort_values(
            "partial_corr", key=lambda s: s.abs(), ascending=False
        )
        table_id = sink.add_table(
            "dag_implications", table, "DAG-implied independence tests"
        )
        n_tension = int((table["verdict"] == "tension").sum())
        max_r = float(table["partial_corr"].abs().max())
        detail = (
            f"{len(rows)} of {total} graph-implied independencies tested; "
            f"{n_tension} in tension (|partial r| > {TENSION}), largest {max_r:.3f}. "
            "Tension means the data and the assumed graph disagree; descriptive only."
        )
        if not_testable:
            detail += f" {not_testable} not testable on numeric data."
        if truncated:
            detail += f" Capped at {MAX_IMPLICATIONS} of the implied set."
        return EDACheck(
            name="dag_consistency",
            status=EDACheckStatus.WARNING if n_tension else EDACheckStatus.OK,
            detail=detail,
            metrics={
                "n_implications": float(total),
                "n_tested": float(len(rows)),
                "n_tension": float(n_tension),
                "max_partial_corr": round(max_r, 4),
            },
            artifact_ids=[table_id],
        )
    except Exception as exc:
        return failed("dag_consistency", exc)
