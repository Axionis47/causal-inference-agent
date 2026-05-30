"""Treatment effects section: results table, forest plot, per-method verification cells.

`render_treatment_effects` is the public entry point. The per-method
cell builders (OLS, IPW, AIPW, DML) live in sibling modules so each
is independently readable; `binarization.py` is the shared code
fragment they all emit at the top of their cells.
"""

from .render import render_treatment_effects

__all__ = ["render_treatment_effects"]
