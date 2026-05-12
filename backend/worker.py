"""Explicit queue installation and the separate context worker entry point."""

import argparse
import logging

from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import text

from backend.context import jobs as context_jobs  # noqa: F401
from backend.db import engine
from backend.integration import worker as feature_jobs  # noqa: F401
from backend.jobs import app, dsn


def install_schema() -> None:
    # Procrastinate owns this schema. Install once during a release, never on API startup.
    # Future library upgrades use its supplied migrations, not reapplication of schema.sql.
    with engine().begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(714628951)"))
        if connection.scalar(text("SELECT to_regclass('procrastinate_jobs')")) is None:
            connection.exec_driver_sql(app.schema_manager.get_schema().replace("%", "%%"))
    with PostgresSaver.from_conn_string(dsn()) as saver:
        saver.setup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "schema"])
    command = parser.parse_args().command
    if command == "schema":
        install_schema()
        print("Procrastinate schema is installed.")
    else:
        logging.basicConfig(level=logging.INFO)
        with app.open():
            app.run_worker(queues=["context", "chat", "analysis"], concurrency=2)


if __name__ == "__main__":
    main()
