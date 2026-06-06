"""Numpy-safe JSON serialization for persisting AnalysisState.

State carries `dict[str, Any]` slots that agents fill with pandas/numpy
analysis results. A `numpy.bool_` (or `int64`/`float64`/`ndarray`) landing in
one of those slots makes `state.model_dump(mode="json")` raise
`PydanticSerializationError: Unable to serialize unknown type`, which kills
the job at the persistence boundary (most visibly when parking the state at
the human-approval gate).

`pydantic_core.to_jsonable_python` produces the same output as
`model_dump(mode="json")` but accepts a `fallback` for types its serializer
does not recognise. We use that to coerce numpy scalars/arrays to native
Python, so the persisted JSON round-trips back through `model_validate`
cleanly with native types.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel
from pydantic_core import to_jsonable_python


def _numpy_fallback(obj: Any) -> Any:
    """Convert numpy scalars/arrays to native Python; str() as last resort."""
    if isinstance(obj, np.generic):  # np.bool_, np.int64, np.float64, ...
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def dump_state_jsonable(state: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model to a JSON-native dict, tolerating numpy.

    Drop-in replacement for `state.model_dump(mode="json")` on models whose
    `Any`-typed fields may contain numpy values.
    """
    return to_jsonable_python(state, fallback=_numpy_fallback)
