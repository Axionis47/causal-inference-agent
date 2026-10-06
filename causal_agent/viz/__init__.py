"""Pictures. A FigureSpec is data with addresses that a lane draws from its own artifacts; an Artifact (`memory/artifacts.py`) is what the
exploration agent (`explore/`) made on request, with its code and its numbers, in its sandbox (`explore/sandbox.py`)."""

from causal_agent.viz.spec import FigureSpec, Mark, Series

__all__ = ["FigureSpec", "Mark", "Series"]
