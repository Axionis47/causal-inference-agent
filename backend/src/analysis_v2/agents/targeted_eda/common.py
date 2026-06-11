"""Shared shapes for EDA checks: inputs bundle and the artifact sink.

A check is a pure-ish function (EDAInputs, ArtifactSink) -> EDACheck. It
computes descriptive facts, may emit plot/table artifacts through the
sink, and never claims a causal effect. Checks that cannot run return
SKIPPED or FAILED with the reason; they do not raise.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis_v2.agents.base import AgentCtx  # noqa: E402
from src.analysis_v2.core import AnalysisStage, ArtifactKind  # noqa: E402
from src.analysis_v2.spec import CausalSpec, EDACheck, ProfileSummary  # noqa: E402


@dataclass
class EDAInputs:
    frame: pd.DataFrame
    spec: CausalSpec
    profile: ProfileSummary


@dataclass
class ArtifactSink:
    """Routes check artifacts through the agent context with eda/ paths."""

    ctx: AgentCtx
    agent: str
    stage: AnalysisStage
    emitted: list[str] = field(default_factory=list)

    def add_plot(self, name: str, fig, title: str) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        artifact_id = f"eda/{name}"
        self.ctx.add_artifact(
            agent=self.agent, stage=self.stage, artifact_id=artifact_id,
            kind=ArtifactKind.PLOT, title=title,
            relative_path=f"eda/plots/{name}.png", payload=buf.getvalue(),
        )
        self.emitted.append(artifact_id)
        return artifact_id

    def add_table(self, name: str, frame: pd.DataFrame, title: str) -> str:
        artifact_id = f"eda/{name}"
        self.ctx.add_table_artifact(
            agent=self.agent, stage=self.stage, artifact_id=artifact_id,
            title=title, relative_path=f"eda/tables/{name}.csv", frame=frame,
        )
        self.emitted.append(artifact_id)
        return artifact_id


CheckFn = Callable[[EDAInputs, ArtifactSink], EDACheck]


def skipped(name: str, reason: str) -> EDACheck:
    from src.analysis_v2.spec import EDACheckStatus

    return EDACheck(name=name, status=EDACheckStatus.SKIPPED, detail=reason)


def failed(name: str, error: Exception) -> EDACheck:
    from src.analysis_v2.spec import EDACheckStatus

    return EDACheck(name=name, status=EDACheckStatus.FAILED, detail=str(error)[:300])


def resolved_col(spec_ref, frame: pd.DataFrame) -> str | None:
    if spec_ref is None or spec_ref.column is None or spec_ref.column not in frame.columns:
        return None
    return spec_ref.column
