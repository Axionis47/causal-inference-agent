"""Phase 3 shares the real storage/app harness with ingestion and library."""

import io
from uuid import uuid4

import pytest


@pytest.fixture
def saved(storage):
    def create(owner, data=b"code,amount\n001,1\n002,2\n", context=" original\ncontext "):
        return storage.ingest(owner, str(uuid4()), "Profile study", context, "study.csv", io.BytesIO(data))

    return create
