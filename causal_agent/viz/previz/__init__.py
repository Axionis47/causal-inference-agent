"""The figures shown before a run, one module per family. Each function is pure over a table and named columns, and
returns the figure and the probe number from the same computation. No memory, no model."""

from causal_agent.viz.previz import adjustment, diff_in_diff, discontinuity

__all__ = ["adjustment", "diff_in_diff", "discontinuity"]
