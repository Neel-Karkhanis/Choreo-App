"""SQLite-backed user accounts.

The only thing in the app that lives in a real database rather than flat
JSON files: track/grid/library data stays content-addressed JSON (see
server.py's module docstring for why), but an account needs an atomic
"does this email already exist" check and a fast look-up by id on every
request — a natural fit for one tiny table rather than a hand-rolled JSON
store keyed by email.

Only the API process ever opens this for writing (the worker process never
touches it), so SQLite's single-writer model is a non-issue here.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class User:
    id: int
    email: str
    password_hash: str
    tos_accepted_at: str
    created_at: str


_DB_PATH: Optional[Path] = None


def init_db(db_path: Path) -> None:
    """Point the module at its database file and ensure the schema exists.

    Called once at process startup (server.py's lifespan). WAL mode is set
    so a slow reader never blocks a writer (or vice versa) — cheap
    insurance even though this process is the only writer today.
    """
    global _DB_PATH
    _DB_PATH = db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                tos_accepted_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


@contextmanager
def _connect():
    if _DB_PATH is None:
        raise RuntimeError("db.init_db() has not been called yet")
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row_to_user(row) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        password_hash=row["password_hash"],
        tos_accepted_at=row["tos_accepted_at"],
        created_at=row["created_at"],
    )


def get_user_by_email(email: str) -> Optional[User]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return _row_to_user(row) if row else None


def get_user_by_id(user_id: int) -> Optional[User]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row) if row else None


def create_user(email: str, password_hash: str) -> User:
    """Insert a new account. Raises sqlite3.IntegrityError on a duplicate email.

    Builds the returned User from the values just written rather than
    re-querying, so the insert's transaction can commit (on the `with`
    block's exit) before anything tries to read the row back.
    """
    email = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, tos_accepted_at, created_at) "
            "VALUES (?, ?, ?, ?)",
            (email, password_hash, now, now),
        )
        user_id = cur.lastrowid
    return User(id=user_id, email=email, password_hash=password_hash, tos_accepted_at=now, created_at=now)
