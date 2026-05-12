"""Shared durable delivery application. Integration registers Chat and Analysis deliveries; Context owns its preparation task."""

import procrastinate
from sqlalchemy.engine import make_url

from backend.config import settings


def dsn() -> str:
    return (
        make_url(settings().database_url).set(drivername="postgresql").render_as_string(hide_password=False)
    )


app = procrastinate.App(connector=procrastinate.PsycopgConnector(conninfo=dsn()))
