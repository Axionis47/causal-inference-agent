"""Run migrations and tests against the explicitly named isolated database."""

import os
import subprocess
import sys

from dotenv import dotenv_values
from sqlalchemy.engine import make_url

root_url = os.environ.get("DATABASE_URL") or dotenv_values(".env").get("DATABASE_URL")
if not root_url:
    raise SystemExit("Set DATABASE_URL in .env first.")
target = os.environ.get("TEST_DATABASE_URL") or make_url(root_url).set(
    database="causal_test"
).render_as_string(hide_password=False)
if make_url(target).database != "causal_test":
    raise SystemExit("Refusing to run destructive fixtures outside causal_test.")
env = {**os.environ, "DATABASE_URL": target, "TEST_DATABASE_URL": target}
subprocess.run([sys.executable, "scripts/check_dependencies.py"], check=True)
subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], env=env, check=True)
subprocess.run([sys.executable, "-m", "backend.worker", "schema"], env=env, check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], env=env, check=True)
