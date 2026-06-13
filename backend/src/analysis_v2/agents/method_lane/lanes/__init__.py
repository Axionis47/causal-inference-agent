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

# Parallel map of the non-raising precondition check for each lane. The
# readiness checker (S6b) calls these; each lane's run() calls its own first.
LANE_CHECKS = {
    MethodLane.OBSERVATIONAL: observational.check_ready,
    MethodLane.MATCHING: matching.check_ready,
    MethodLane.DID: did.check_ready,
    MethodLane.RDD: rdd.check_ready,
    MethodLane.IV: iv.check_ready,
    MethodLane.TIME_SERIES: time_series.check_ready,
    MethodLane.MEDIATION: mediation.check_ready,
    MethodLane.SURVIVAL: survival.check_ready,
}

__all__ = ["LANES", "LANE_CHECKS", "LaneArtifact", "LaneInputError", "LaneOutcome"]
