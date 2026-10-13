"""The engine: the only package that imports DoWhy. `model` builds the causal model and identifies the effect on a graph, which
the reasoning may ask for; `estimators`, `falsify` and `sensitivity` run a frozen design, which only the execution calls.

No branching on method: every estimator is a name plus a dict, every falsification and every sensitivity analysis the same.
Failures come back as facts on the record, never as exceptions.
"""

from __future__ import annotations

import logging
import warnings

import matplotlib

matplotlib.use("Agg")  # the sensitivity analysis draws; nothing here has a screen

logging.getLogger("dowhy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="dowhy")
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="statsmodels")
