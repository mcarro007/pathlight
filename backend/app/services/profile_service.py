from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.app.services.sqlite_db import connect, execute, fetchone

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_profile(user_id: int) -> Dict[str, Any]:
    conn = connect()
    try:
        row = fetchone(conn, "SELECT * FROM profiles WHERE user_id = ?", (user_id,))
        if not row:
            execute(conn, "INSERT OR REPLACE INTO profiles(user_id, updated_at) VALUES(?,?)", (user_id, _now_iso()))
            row = fetchone(conn, "SELECT * FROM profiles WHERE user_id = ?", (user_id,))
        return dict(row) if row else {"user_id": user_id}
    finally:
        conn.close()

def upsert_profile(user_id: int, patch: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"name","location","salary_min","skills","interests","experience","education","resume_text"}
    cleaned = {k: patch.get(k) for k in allowed if k in patch}

    conn = connect()
    try:
        existing = fetchone(conn, "SELECT user_id FROM profiles WHERE user_id = ?", (user_id,))
        if not existing:
            execute(conn, "INSERT INTO profiles(user_id, updated_at) VALUES(?,?)", (user_id, _now_iso()))

        # Build dynamic update
        sets = []
        params = []
        for k, v in cleaned.items():
            sets.append(f"{k} = ?")
            params.append(v)
        sets.append("updated_at = ?")
        params.append(_now_iso())
        params.append(user_id)

        execute(conn, f"UPDATE profiles SET {', '.join(sets)} WHERE user_id = ?", tuple(params))
        row = fetchone(conn, "SELECT * FROM profiles WHERE user_id = ?", (user_id,))
        return dict(row) if row else {"user_id": user_id}
    finally:
        conn.close()

def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())

def parse_resume(resume_text: str, mode: str = "auto") -> Dict[str, Any]:
    """
    mode: "auto" (OpenAI if available), "openai", "heuristic"
    """
    text = (resume_text or "").strip()
    if not text:
        return {"ok": False, "message": "No resume text provided."}

    if mode in ("openai","auto") and _has_openai_key():
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            prompt = f"""
Extract a structured candidate profile from this resume text.

Return STRICT JSON ONLY with keys:
- name (string or null)
- location (string or null)
- skills (string)  (comma-separated is fine)
- interests (string)
- experience (string) (summary)
- education (string) (summary)
- suggested_roles (array of strings)
- suggested_industries (array of strings)
- notes (string) (brief)

Resume:
{text}
"""
            resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
            raw = (resp.output_text or "").strip()
            # best-effort JSON extraction
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            obj = json.loads(m.group(0) if m else raw)
            if not isinstance(obj, dict):
                raise ValueError("Non-dict JSON")
            obj["ok"] = True
            obj["parser"] = "openai"
            return obj
        except Exception as e:
            if mode == "openai":
                return {"ok": False, "message": f"OpenAI parse failed: {repr(e)}"}
            # fall through to heuristic

    # Heuristic fallback
    skills = []
    skill_keywords = [
        "python","sql","excel","power bi","tableau","aws","azure","gcp","ml","machine learning","nlp","data analysis",
        "project management","jira","confluence","react","next.js","fastapi","streamlit","docker","kubernetes",
        "statistics","experimentation","a/b","communication","leadership","training","instructional design"
    ]
    low = text.lower()
    for kw in skill_keywords:
        if kw in low:
            skills.append(kw)

    # crude education extraction
    edu_lines = []
    for line in text.splitlines():
        l = line.strip()
        if any(x in l.lower() for x in ["b.s", "b.a", "m.s", "m.a", "ph.d", "degree", "university", "college", "cert"]):
            edu_lines.append(l)
    education = "\n".join(edu_lines[:8]).strip()

    # crude experience summary
    exp_lines = []
    for line in text.splitlines():
        l = line.strip()
        if re.search(r"\b(20\d{2}|19\d{2})\b", l) or any(x in l.lower() for x in ["experience","work history","employment","responsibilities"]):
            exp_lines.append(l)
    experience = "\n".join(exp_lines[:12]).strip()

    # suggestions (generic, but useful)
    suggested_roles = []
    if "python" in skills and ("data analysis" in skills or "sql" in skills):
        suggested_roles += ["Data Analyst", "Analytics Engineer", "Business Intelligence Analyst"]
    if "machine learning" in skills or "ml" in skills:
        suggested_roles += ["Machine Learning Engineer", "Applied Scientist", "Data Scientist"]
    if "training" in skills or "instructional design" in skills:
        suggested_roles += ["Learning Experience Designer", "Technical Trainer", "Enablement Manager"]

    suggested_roles = list(dict.fromkeys(suggested_roles))[:10]

    return {
        "ok": True,
        "parser": "heuristic",
        "name": None,
        "location": None,
        "skills": ", ".join(skills),
        "interests": "",
        "experience": experience,
        "education": education,
        "suggested_roles": suggested_roles,
        "suggested_industries": ["Healthcare", "Public Sector", "Technology", "Education"],
        "notes": "Heuristic extraction. Enable OpenAI for richer parsing."
    }
