"""Password hashing and signed-cookie sessions.

bcrypt is used directly rather than through passlib: passlib's bcrypt
backend has had long-standing version-detection breakage against
bcrypt>=4.1, and calling bcrypt.hashpw/checkpw directly is one fewer
moving part for something this small.

A session is a signed, timestamped cookie holding nothing but the user's
id. itsdangerous verifies the signature and expiry on every read, so the
cookie can't be forged or replayed past SESSION_MAX_AGE_SECONDS, and there
is no server-side session table to keep in sync with it — logging out just
means the browser stops sending the cookie.
"""

import os

import bcrypt
from fastapi import Cookie, HTTPException
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

import db

SESSION_COOKIE_NAME = "choreo_session"
SESSION_MAX_AGE_SECONDS = 30 * 24 * 60 * 60  # 30 days

_SECRET = os.environ.get("SESSION_SECRET")
if not _SECRET:
    raise RuntimeError(
        "SESSION_SECRET is not set. Sessions are cryptographically signed "
        "with it, so a missing (or default, or shared-across-deploys) "
        "secret must never silently ship. Set it in your .env (see "
        ".env.example) or the process environment before starting the "
        "server."
    )

_serializer = URLSafeTimedSerializer(_SECRET, salt="choreo-session")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def make_session_cookie(user_id: int) -> str:
    return _serializer.dumps({"user_id": user_id})


def get_current_user(
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> db.User:
    """The FastAPI dependency every route in server.py takes.

    Resolves straight from the signed cookie to a live row in `users` on
    every call — no server-side session cache to invalidate, so a deleted
    account (not implemented yet, but this is what makes it safe to add
    later) stops authenticating on its very next request.
    """
    if session is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    try:
        data = _serializer.loads(session, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired) as exc:
        raise HTTPException(status_code=401, detail="Session invalid or expired") from exc

    user = db.get_user_by_id(data["user_id"])
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user
