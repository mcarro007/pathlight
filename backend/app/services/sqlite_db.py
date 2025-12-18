from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional, Tuple


# In Docker/App Runner, this resolves to /app/data/job_analyzer.db
# Locally, it resolves to <repo>/data/job_analyzer.db
DB_PATH = Path(
    os.getenv(
        "JOB_ANALYZER_DB_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "job_analyzer.db"),
    )
)


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_conn() -> sqlite3.Connection:
    """
    Small compatibility wrapper used by auth_service and other services.
    """
    return connect()


def init_db() -> None:
    conn = connect()
    cur = conn.cursor()

    # Users for auth
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          email TEXT NOT NULL UNIQUE,
          password_hash TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )

    # Sessions (tokens) for auth
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
          token TEXT PRIMARY KEY,
          user_id INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    # Profiles
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
          user_id INTEGER PRIMARY KEY,
          name TEXT,
          location TEXT,
          salary_min REAL,
          skills TEXT,
          interests TEXT,
          experience TEXT,
          education TEXT,
          resume_text TEXT,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    # Orgs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS orgs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )

    # Org memberships
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS org_memberships (
          user_id INTEGER NOT NULL,
          org_id INTEGER NOT NULL,
          role TEXT NOT NULL,
          created_at TEXT NOT NULL,
          PRIMARY KEY (user_id, org_id),
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY(org_id) REFERENCES orgs(id) ON DELETE CASCADE
        );
        """
    )

    # Saved matches
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_matches (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    # Enterprise artifacts
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS enterprise_artifacts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          org_id INTEGER NOT NULL,
          user_id INTEGER NOT NULL,
          kind TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(org_id) REFERENCES orgs(id) ON DELETE CASCADE,
          FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


def fetchone(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchone()


def fetchall(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchall()


def execute(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> int:
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.lastrowid
