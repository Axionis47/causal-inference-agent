"""Method lane runners: one deterministic estimator family per lane."""
from src.analysis_v2.spec import MethodLane

from . import did, iv, matching, mediation, observational, rdd, survival, time_series
from .common import LaneArtifact, LaneInputError, LaneOutcome

LANES = {
    MethodLane.OBSERVATIONAL: observational.run,
    MethodLane.MATCHING: matching.run,
    MethodLane.DID: did.run,
    MethodLane.RDD: rdd.run,
    MethodLane.IV: iv.run,
    MethodLane.TIME_SERIES: time_series.run,
    MethodLane.MEDIATION: mediation.run,
    MethodLane.SURVIVAL: survival.run,
}

__all__ = ["LANES", "LaneArtifact", "LaneInputError", "LaneOutcome"]
