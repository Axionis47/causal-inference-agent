"""Dockerized Postgres 18.6 and MinIO fixtures for persistence tests (T-005, D-017)."""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True, check=False)
    return probe.returncode == 0

DOCKER_AVAILABLE = _docker_available()

requires_docker = pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")


def _run_container(image: str, *docker_args: str, command: tuple[str, ...] = ()) -> str:
    argv = ["docker", "run", "-d", "--rm", *docker_args, image, *command]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def _host_port(container_id: str, container_port: int) -> int:
    output = subprocess.run(
        ["docker", "port", container_id, str(container_port)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(output.splitlines()[0].rsplit(":", 1)[1])


def _stop(container_id: str) -> None:
    subprocess.run(["docker", "stop", container_id], capture_output=True, check=False)


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    if not DOCKER_AVAILABLE:
        pytest.skip("docker daemon not available")
    container = _run_container(
        "postgres:18.6", "-e", "POSTGRES_PASSWORD=causal-test", "-p", "127.0.0.1:0:5432",
        "--name", f"causal-test-pg-{uuid.uuid4().hex[:8]}",
    )
    try:
        import psycopg

        port = _host_port(container, 5432)
        dsn = f"postgresql://postgres:causal-test@127.0.0.1:{port}/postgres"
        deadline = time.monotonic() + 60
        while True:
            try:
                psycopg.connect(dsn, connect_timeout=2).close()
                break
            except psycopg.OperationalError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        yield dsn
    finally:
        _stop(container)


@pytest.fixture(scope="session")
def minio_s3() -> Iterator[dict[str, Any]]:
    if not DOCKER_AVAILABLE:
        pytest.skip("docker daemon not available")
    container = _run_container(
        "minio/minio:latest", "-e", "MINIO_ROOT_USER=causal", "-e",
        "MINIO_ROOT_PASSWORD=causal-test", "-p", "127.0.0.1:0:9000",
        "--name", f"causal-test-minio-{uuid.uuid4().hex[:8]}",
        command=("server", "/data"),
    )
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]
        from botocore.exceptions import (  # type: ignore[import-untyped]
            ClientError,
            EndpointConnectionError,
        )

        port = _host_port(container, 9000)
        client = boto3.client(
            "s3",
            endpoint_url=f"http://127.0.0.1:{port}",
            aws_access_key_id="causal",
            aws_secret_access_key="causal-test",
            region_name="us-east-1",
            config=Config(signature_version="s3v4", retries={"max_attempts": 1}),
        )
        bucket = "causal-test"
        deadline = time.monotonic() + 60
        while True:
            try:
                client.create_bucket(Bucket=bucket)
                break
            except (EndpointConnectionError, ClientError) as error:
                # MinIO answers HTTP before it is ready (XMinioServerNotInitialized).
                if time.monotonic() > deadline:
                    raise
                if isinstance(error, ClientError) and error.response.get("Error", {}).get(
                    "Code"
                ) not in ("XMinioServerNotInitialized", "ServiceUnavailable", "SlowDown"):
                    raise
                time.sleep(0.5)
        yield {"client": client, "bucket": bucket}
    finally:
        _stop(container)


@pytest.fixture()
def conn(postgres_dsn: str) -> Iterator[Any]:
    import psycopg

    from causal.shared.persistence import apply_migrations

    admin = psycopg.connect(postgres_dsn, autocommit=True)
    database = f"test_{uuid.uuid4().hex[:10]}"
    admin.execute(f'CREATE DATABASE "{database}"')
    admin.close()
    dsn = postgres_dsn.rsplit("/", 1)[0] + f"/{database}"
    test_conn = psycopg.connect(dsn, autocommit=True)
    apply_migrations(test_conn, MIGRATIONS)
    yield test_conn
    test_conn.close()


@pytest.fixture()
def object_store(minio_s3: dict[str, Any]) -> Any:
    from causal.shared.persistence import ObjectStore

    return ObjectStore(minio_s3["client"], minio_s3["bucket"])
