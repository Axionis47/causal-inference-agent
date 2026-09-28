"""Pictures. A FigureSpec is data with addresses that a lane draws from its own artifacts; an Artifact (`store.py`) is a picture the
Drawer (`draw.py`) made on request, with its code and its numbers, in a sandbox (`sandbox.py`)."""

from causal_agent.viz.spec import FigureSpec, Mark, Series

__all__ = ["FigureSpec", "Mark", "Series"]
