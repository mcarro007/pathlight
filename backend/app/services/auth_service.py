from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

from fastapi import HTTPException, status
from passlib.context import CryptContext

from backend.app.services.sqlite_db import get_conn

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _expires_iso(hours: int = 24) -> str:
    return (datetime.utcnow() + timedelta(hours=hours)).isoformat()


def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def register_user(email: str, password: str) -> int:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists",
        )

    hashed = _hash_password(password)
    cur.execute(
        "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
        (email, hashed, _now_iso()),
    )
    conn.commit()
    return cur.lastrowid


def login_user(email: str, password: str) -> Tuple[str, int]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()

    if (not row) or (not _verify_password(password, row["password_hash"])):
        # ✅ Correct: 401, not RuntimeError -> no more 500s
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = secrets.token_hex(32)
    user_id = int(row["id"])

    cur.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, _now_iso(), _expires_iso(24)),
    )
    conn.commit()

    return token, user_id


def resolve_token(token: str) -> Optional[int]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,))
    row = cur.fetchone()
    if not row:
        return None

    # Enforce expiration
    try:
        exp = datetime.fromisoformat(row["expires_at"])
        if exp < datetime.utcnow():
            cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return None
    except Exception:
        return None

    return int(row["user_id"])


def logout_token(token: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
