import argparse
import getpass
import json
import os
from pathlib import Path

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.auth import passwords
from backend.db import engine
from backend.models import AuthSession, User


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["user", "openapi"])
    parser.add_argument("--email")
    args = parser.parse_args()
    if args.command == "openapi":
        from backend.main import app

        Path("web/openapi.json").write_text(json.dumps(app.openapi(), indent=2) + "\n")
        return
    email = (args.email or os.environ.get("BOOTSTRAP_EMAIL", "")).strip().lower()
    try:
        TypeAdapter(EmailStr).validate_python(email)
    except ValidationError:
        parser.error("Provide a valid email in --email or BOOTSTRAP_EMAIL.")
    password = os.environ.get("BOOTSTRAP_PASSWORD") or getpass.getpass("Password: ")
    if len(password) < 12:
        parser.error("Use at least 12 characters.")
    with Session(engine()) as db:
        user = db.scalar(select(User).where(User.email == email))
        if user:
            user.password_hash = passwords.hash(password)
            db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
        else:
            user = User(email=email, password_hash=passwords.hash(password))
            db.add(user)
        db.commit()
    print("Account saved. Existing datasets retain their owner identity.")


if __name__ == "__main__":
    main()
