"""Integrity of a serialized frozen specification, including its nested fields."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, model_validator


class FrozenDesign(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    integrity: str = ""

    def fingerprint(self) -> str:
        raw = json.dumps(self.model_dump(mode="json", exclude={"integrity"}), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    @model_validator(mode="after")
    def seal(self):
        if self.integrity and self.integrity != self.fingerprint():
            raise ValueError("frozen design does not match its integrity seal")
        if not self.integrity:
            object.__setattr__(self, "integrity", self.fingerprint())
        return self

    def assert_integrity(self) -> None:
        if self.integrity != self.fingerprint():
            raise ValueError("frozen specification changed after design admission")
