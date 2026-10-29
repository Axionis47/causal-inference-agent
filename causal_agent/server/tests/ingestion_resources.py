"""Only harness-created Docker services may host ingestion integration tests.

No deployment database URL, endpoint or bucket is accepted. Every test gets a
fresh database and bucket inside this run's disposable Compose project.
"""

import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from causal_agent.common.config import ingestion
from causal_agent.server.ingestion.service import Store

ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def cluster():
    project = "ingestion-test-" + uuid4().hex
    with tempfile.TemporaryDirectory(prefix=project) as scratch:
        override = Path(scratch) / "ports.yaml"
        override.write_text('services:\n  postgres:\n    ports: ["127.0.0.1::5432"]\n  objects:\n    ports: ["127.0.0.1::8333"]\n')
        argv = ["docker", "compose", "--env-file", os.devnull, "-p", project, "-f", str(ROOT / "compose.yaml"), "-f", str(override)]
        # Avoid accidental deployment interpolation through the parent environment.
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "TMPDIR", "SYSTEMROOT"}}

        def run(*args):
            return subprocess.check_output([*argv, *args], env=env, text=True, stderr=subprocess.STDOUT, timeout=180).strip()

        try:
            run("up", "-d", "--wait", "postgres", "objects")
            pg = run("port", "postgres", "5432")
            s3 = run("port", "objects", "8333")
            yield replace(
                ingestion({}),
                database_url=f"postgresql+psycopg://causal:local-development-only@{pg}/causal",
                endpoint=f"http://{s3}",
                access_key="local-ingestion",
                secret_key="local-development-only",
            )
        finally:
            run("down", "--volumes", "--remove-orphans")


@contextmanager
def database_and_bucket(settings, monkeypatch):
    # Identifiers originate here, never from environment or a user-supplied URL.
    name = "test_" + uuid4().hex
    bucket = name.replace("_", "-")
    admin = create_engine(settings.database_url, isolation_level="AUTOCOMMIT")
    storage = None
    with admin.connect() as db:
        db.execute(text(f'CREATE DATABASE "{name}"'))
    configured = replace(settings, database_url=settings.database_url.rsplit("/", 1)[0] + "/" + name, bucket=bucket)
    try:
        monkeypatch.setenv("DATABASE_URL", configured.database_url)
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        storage = Store(configured)
        storage.objects.client.create_bucket(Bucket=bucket)
        yield storage
    finally:
        if storage is not None:
            client = storage.objects.client
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                for obj in page.get("Contents", []):
                    client.delete_object(Bucket=bucket, Key=obj["Key"])
            client.delete_bucket(Bucket=bucket)
            client.close()
            storage.engine.dispose()
        try:
            with admin.connect() as db:
                db.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        finally:
            admin.dispose()
