"""Tests for numpy-safe state serialization."""

import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.storage.serialize import dump_state_jsonable


class _State(BaseModel):
    name: str
    when: datetime.datetime
    blob: dict[str, Any]


def _state_with(blob: dict[str, Any]) -> _State:
    return _State(name="job", when=datetime.datetime(2026, 6, 5, 12, 0, 0), blob=blob)


def test_numpy_bool_is_coerced_to_native_bool():
    # This is the exact shape that crashed the live run: a numpy.bool_ in an
    # Any-typed slot makes model_dump(mode="json") raise.
    payload = dump_state_jsonable(_state_with({"is_numeric": np.bool_(True)}))

    assert payload["blob"]["is_numeric"] is True
    assert isinstance(payload["blob"]["is_numeric"], bool)


def test_numpy_scalars_and_arrays_become_native():
    payload = dump_state_jsonable(
        _state_with(
            {"n": np.int64(5), "f": np.float64(1.5), "arr": np.array([1, 2, 3])}
        )
    )

    assert payload["blob"] == {"n": 5, "f": 1.5, "arr": [1, 2, 3]}
    assert isinstance(payload["blob"]["n"], int)
    assert isinstance(payload["blob"]["f"], float)


def test_normal_fields_match_model_dump_json():
    # The numpy fallback must not change serialization of ordinary fields.
    state = _state_with({})

    assert dump_state_jsonable(state) == state.model_dump(mode="json")


def test_output_round_trips_through_model_validate():
    payload = dump_state_jsonable(_state_with({"flag": np.bool_(False)}))

    restored = _State.model_validate(payload)

    assert restored.name == "job"
    assert restored.blob["flag"] is False
