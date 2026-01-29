import hashlib
import secrets
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pwdlib import PasswordHash
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.contracts import LoginInput, UserView
from backend.db import session
from backend.models import AuthSession, User, now

router = APIRouter(prefix="/api/auth", tags=["auth"])
passwords = PasswordHash.recommended()
dummy_hash = passwords.hash(secrets.token_urlsafe(32))
COOKIE = "causal_session"
Db = Annotated[Session, Depends(session)]


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def current_user(request: Request, db: Db) -> User:
    token = request.cookies.get(COOKIE, "")
    record = db.get(AuthSession, digest(token)) if token else None
    if record is None or record.expires_at <= now():
        raise HTTPException(401, "Please log in to continue.")
    user = db.get(User, record.user_id)
    if user is None:
        raise HTTPException(401, "Please log in to continue.")
    return user


Principal = Annotated[User, Depends(current_user)]


@router.post("/login", response_model=UserView)
def login(body: LoginInput, request: Request, response: Response, db: Db) -> User:
    user = db.scalar(select(User).where(User.email == str(body.email).strip().lower()))
    valid = passwords.verify(body.password, user.password_hash if user else dummy_hash)
    if not valid or user is None:
        raise HTTPException(401, "Email or password is incorrect.")
    previous = request.cookies.get(COOKIE)
    if previous:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == digest(previous)))
    db.execute(delete(AuthSession).where(AuthSession.expires_at <= now()))
    token = secrets.token_urlsafe(32)
    db.add(
        AuthSession(
            token_hash=digest(token),
            user_id=user.id,
            expires_at=now() + timedelta(hours=settings().session_hours),
        )
    )
    db.commit()
    response.set_cookie(
        COOKIE,
        token,
        httponly=True,
        secure=settings().cookie_secure,
        samesite="strict",
        max_age=settings().session_hours * 3600,
        path="/api",
    )
    return user


@router.get("/me", response_model=UserView)
def me(user: Principal) -> User:
    return user


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: Db) -> None:
    token = request.cookies.get(COOKIE)
    if token:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == digest(token)))
        db.commit()
    response.delete_cookie(
        COOKIE, path="/api", httponly=True, secure=settings().cookie_secure, samesite="strict"
    )
