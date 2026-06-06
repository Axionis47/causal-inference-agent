"""Analysis package: orchestrator, substrate, and specialist agents.

Public surface:
- run(job_input): the seam from JobInput to a finalised AnalysisState.

This package is being grown incrementally. Today substrate is in place,
agents have just been relocated from ``src/agents/``, and ``run()`` is
a stub. The orchestrator moves in during S6 and the API rewires through
``run()`` in S21.
"""
from src.analysis.run import run

__all__ = ["run"]
