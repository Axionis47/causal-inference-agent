import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

# Only the explicitly named test database is reset. The running preview is never touched.
config = dotenv_values(".env")
test_url = (
    os.environ.get("TEST_DATABASE_URL")
    or str(config.get("DATABASE_URL", "")).rsplit("/", 1)[0] + "/causal_test"
)
if make_url(test_url).database != "causal_test":
    raise RuntimeError("Tests require the separate causal_test database.")
os.environ["DATABASE_URL"] = test_url

from backend.auth import passwords  # noqa: E402
from backend.config import settings  # noqa: E402
from backend.db import engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import User  # noqa: E402
from backend.storage import LocalFiles, files  # noqa: E402

PASSWORD = "a-password-for-tests-only"


@pytest.fixture(autouse=True)
def prevent_unmocked_agent_calls(monkeypatch):
    def unexpected(request):
        raise AssertionError("Unit/integration tests must explicitly stub native provider calls; real-model acceptance is opt-in.")
    monkeypatch.setattr("backend.model_access.invoke_tools", unexpected)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path) -> Iterator[None]:
    settings.cache_clear()
    with engine().begin() as db:
        db.execute(
            text("TRUNCATE auth_sessions, dataset_versions, datasets, users, procrastinate_jobs CASCADE")
        )
    with Session(engine()) as db:
        db.add_all(
            [
                User(email=email, password_hash=passwords.hash(PASSWORD))
                for email in ["owner@example.com", "other@example.com"]
            ]
        )
        db.commit()
    app.dependency_overrides[files] = lambda: LocalFiles(tmp_path)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app, headers={"X-Causal-Client": "web"}) as client:
        yield client


@pytest.fixture
def owner(client: TestClient) -> TestClient:
    assert (
        client.post("/api/auth/login", json={"email": "owner@example.com", "password": PASSWORD}).status_code
        == 200
    )
    return client
