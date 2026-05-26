"""Notebook Generator package — decomposed from monolithic notebook_generator.py.

Backward-compatible: `from src.analysis.agents.specialists.notebook import NotebookGeneratorAgent`
"""

from .agent import NotebookGeneratorAgent
from .helpers import deduplicate_effects, deduplicate_sensitivity, save_notebook

__all__ = [
    "NotebookGeneratorAgent",
    "deduplicate_effects",
    "deduplicate_sensitivity",
    "save_notebook",
]
