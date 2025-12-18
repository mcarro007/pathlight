@'
import os
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

DB_PATH = os.getenv("JOB_ANALYZER_DB", os.path.join(os.path.dirname(__file__), "job_analyzer.sqlite3"))

def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _conn()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS consumer_profile (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      data_json TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS consumer_shortlist (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      job_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS consumer_screens (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      jd_text TEXT NOT NULL,
      result_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS corporate_roles (
      id TEXT PRIMARY KEY,
      intake_json TEXT NOT NULL,
      current_json TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS corporate_versions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      role_id TEXT NOT NULL,
      version_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    """)

    con.commit()
    con.close()

def upsert_profile(data: Dict[str, Any]) -> None:
    con = _conn()
    now = datetime.utcnow().isoformat()
    con.execute(
        "INSERT INTO consumer_profile (id, data_json, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at",
        (json.dumps(data), now),
    )
    con.commit()
    con.close()

def get_profile() -> Optional[Dict[str, Any]]:
    con = _conn()
    row = con.execute("SELECT data_json FROM consumer_profile WHERE id=1").fetchone()
    con.close()
    if not row:
        return None
    return json.loads(row["data_json"])

def add_shortlist(job: Dict[str, Any]) -> int:
    con = _conn()
    now = datetime.utcnow().isoformat()
    cur = con.execute(
        "INSERT INTO consumer_shortlist (job_json, created_at) VALUES (?, ?)",
        (json.dumps(job), now),
    )
    con.commit()
    rid = int(cur.lastrowid)
    con.close()
    return rid

def list_shortlist(limit: int = 100) -> list:
    con = _conn()
    rows = con.execute(
        "SELECT id, job_json, created_at FROM consumer_shortlist ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({"id": r["id"], "created_at": r["created_at"], "job": json.loads(r["job_json"])})
    return out

def add_screen(jd_text: str, result: Dict[str, Any]) -> int:
    con = _conn()
    now = datetime.utcnow().isoformat()
    cur = con.execute(
        "INSERT INTO consumer_screens (jd_text, result_json, created_at) VALUES (?, ?, ?)",
        (jd_text, json.dumps(result), now),
    )
    con.commit()
    rid = int(cur.lastrowid)
    con.close()
    return rid

def save_role(role_id: str, intake: Dict[str, Any], current: Optional[Dict[str, Any]]) -> None:
    con = _conn()
    now = datetime.utcnow().isoformat()
    con.execute(
        "INSERT INTO corporate_roles (id, intake_json, current_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET intake_json=excluded.intake_json, current_json=excluded.current_json, updated_at=excluded.updated_at",
        (role_id, json.dumps(intake), json.dumps(current) if current is not None else None, now, now),
    )
    con.commit()
    con.close()

def add_version(role_id: str, version: Dict[str, Any]) -> int:
    con = _conn()
    now = datetime.utcnow().isoformat()
    cur = con.execute(
        "INSERT INTO corporate_versions (role_id, version_json, created_at) VALUES (?, ?, ?)",
        (role_id, json.dumps(version), now),
    )
    con.commit()
    rid = int(cur.lastrowid)
    con.close()
    return rid
'@ | Set-Content -Encoding utf8 .\backend\app\db.py
