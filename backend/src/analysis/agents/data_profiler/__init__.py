"""Data profiler agent: dataset statistics + treatment encoding decisions.

DataProfilerAgent lives at .agent and is intentionally not re-exported here;
importing it would close a cycle with `analysis.agents.base` which itself
re-exports the typed output from this package.
"""

from .output import DataProfile, TreatmentEncoding

__all__ = ["DataProfile", "TreatmentEncoding"]
