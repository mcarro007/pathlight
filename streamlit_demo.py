from __future__ import annotations

import base64
import json
import os
import re
import html
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st



# DEBUG DISABLED: st.write("DEBUG: raw query_params =", dict(st.query_params))
try:
    pass
except Exception:
    pass

# DEBUG DISABLED: st.write("DEBUG: mode =", st.query_params.get("mode"))
# DEBUG DISABLED: st.write("DEBUG: query param read error:", e)

import pandas as pd
import numpy as np

# Optional FAISS (best-effort)
try:
    import faiss  # type: ignore
except Exception:
    faiss = None


# Optional local extractors (best-effort)
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

# =============================================================================
# Pathlight Ã¢â‚¬â€œ Streamlit Demo (Premium, Indeed-like, Hard-Separated) [patch11]
# =============================================================================
API_BASE = os.getenv(
    "JOB_ANALYZER_API_BASE_URL",
    "https://qjpajy5qrz.us-east-1.awsapprunner.com",
).rstrip("/")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

APP_BRAND = "Pathlight"
MODE_EXPLORER = "Explorer"
MODE_TALENT = "Talent Studio"


# ==============================
# API helpers (MUST BE TOP-LEVEL)
# ==============================

def api_headers() -> dict:
    token = st.session_state.get("auth_token")
    if not token:
        return {"Content-Type": "application/json"}
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def api_url(path: str) -> str:
    base = (API_BASE or "").rstrip("/")
    p = (path or "").strip()
    if not p.startswith("/"):
        p = "/" + p
    return base + p


class ApiResponse:
    def __init__(self, ok: bool, data=None, status: int | None = None, text: str | None = None):
        self.ok = ok
        self.data = data
        self.status = status
        self.text = text


def api_get(path: str, timeout: int = 30) -> ApiResponse:
    r = requests.get(api_url(path), headers=api_headers(), timeout=timeout)
    try:
        r.raise_for_status()
        return ApiResponse(True, r.json() if r.content else None, r.status_code)
    except Exception:
        return ApiResponse(False, None, r.status_code, r.text)


def api_post(path: str, payload: dict | None = None, timeout: int = 60) -> ApiResponse:
    r = requests.post(api_url(path), headers=api_headers(), json=(payload or {}), timeout=timeout)
    try:
        r.raise_for_status()
        return ApiResponse(True, r.json() if r.content else None, r.status_code)
    except Exception:
        return ApiResponse(False, None, r.status_code, r.text)


# =============================================================================
DEMO_DB_PATH = Path(
    os.getenv(
        "PATHLIGHT_DEMO_DB_PATH",
        str(Path(__file__).resolve().parent / "data" / "pathlight_demo.db"),
    )
)

ROOT_DIR = Path(__file__).resolve().parent


# ---------------- DEBUG /system/build call ----------------

try:
    r = requests.get(f"{API_BASE}/system/build", timeout=10)
    # DEBUG DISABLED: st.write("DEBUG /system/build status =", r.status_code)
    # DEBUG DISABLED: st.write("DEBUG /system/build json =", r.json())
except Exception as e:
    pass
    # DEBUG DISABLED: st.write("DEBUG /system/build ERROR =", repr(e))
# --------------------------------------------------

PARQUETS: Dict[str, str] = {
    "merged_model_ready_sample_100k": str(ROOT_DIR / "merged_model_ready_sample_100k.parquet"),
    "gold_salary_100k": str(ROOT_DIR / "gold_salary_100k.parquet"),
    "salary_train": str(ROOT_DIR / "salary_train.parquet"),
    "salary_val": str(ROOT_DIR / "salary_val.parquet"),
    "semantic_train": str(ROOT_DIR / "semantic_train.parquet"),
    "semantic_val": str(ROOT_DIR / "semantic_val.parquet"),
    "semantic_test": str(ROOT_DIR / "semantic_test.parquet"),
    "hr_test": str(ROOT_DIR / "hr_test.parquet"),
}

@st.cache_resource(show_spinner=False)
def load_parquets() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for k, p in PARQUETS.items():
        try:
            if Path(p).exists():
                dfs[k] = pd.read_parquet(p)
        except Exception:
            continue
    return dfs

def parquet_status() -> Dict[str, Any]:
    dfs = load_parquets()
    return {
        "loaded": sorted(list(dfs.keys())),
        "missing": sorted([k for k, p in PARQUETS.items() if not Path(p).exists()]),
        "shapes": {k: list(v.shape) for k, v in dfs.items()},
    }


MAGENTA = "#D61A73"
MAGENTA_DARK = "#B51260"
INK = "#14161A"
INK_2 = "#2A2E35"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F6F7FA"
BORDER = "#E6E8EE"
MUTED = "#6B7280"


# =============================================================================
# CSS
# =============================================================================

def inject_css() -> None:
    st.markdown(
        f"""
<style>
html, body, [class*="css"] {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}}
.stApp {{ background: {SURFACE_2}; }}

div[data-testid="stToolbar"] {{ visibility: hidden; height: 0px; }}
header[data-testid="stHeader"] {{
  background: rgba(255,255,255,0.65);
  backdrop-filter: blur(10px);
}}

.pl-card {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 1px 0 rgba(10,10,10,0.02);
}}
.pl-card + .pl-card {{ margin-top: 10px; }}

.pl-title {{ font-weight: 900; color: {INK}; font-size: 24px; margin: 0 0 4px 0; }}
.pl-subtitle {{ color: {MUTED}; font-size: 13px; margin: 0; }}
.pl-h2 {{ font-weight: 900; color: {INK}; font-size: 16px; margin: 0 0 8px 0; }}
.pl-muted {{ color: {MUTED}; font-size: 12px; }}

div.stButton > button {{
  border-radius: 14px !important;
  border: 1px solid {BORDER} !important;
  padding: 0.75rem 1.0rem !important;
  font-weight: 900 !important;
}}

.pl-primary div.stButton > button {{
  background: {MAGENTA} !important;
  border-color: {MAGENTA} !important;
  color: white !important;
}}
.pl-primary div.stButton > button:hover {{
  background: {MAGENTA_DARK} !important;
  border-color: {MAGENTA_DARK} !important;
}}

.pl-topbar {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 10px 12px;
}}
.pl-brand {{
  font-weight: 950;
  font-size: 16px;
  color: {INK};
}}
.pl-mode {{
  font-size: 12px;
  color: {MUTED};
  margin-top: 1px;
}}
.pl-linklike {{
  color: {INK};
  font-weight: 900;
  padding: 8px 10px;
  border-radius: 12px;
  border: 1px solid transparent;
  background: transparent;
}}
.pl-linklike:hover {{
  border-color: {BORDER};
  background: {SURFACE_2};
}}

.pl-section {{
  border: 1px solid {BORDER};
  background: {SURFACE};
  border-radius: 16px;
  padding: 14px;
}}

.pl-divider {{
  height: 1px;
  background: {BORDER};
  margin: 12px 0;
}}

/* =======================================================================
   Response callouts (subtle contrast panels)
   ======================================================================= */

.pl-callout {{
  border: 1px solid #E6E8EE;
  border-radius: 16px;
  padding: 12px 14px;
  background: #FFFFFF;
  position: relative;
}}

.pl-callout + .pl-callout {{ margin-top: 10px; }}

.pl-callout::before {{
  content: "";
  position: absolute;
  left: 0;
  top: 10px;
  bottom: 10px;
  width: 6px;
  border-radius: 12px;
  background: #D61A73; /* default = magenta */
  opacity: 0.95;
}}

.pl-callout .pl-callout-title {{
  font-weight: 950;
  font-size: 13px;
  color: #14161A;
  margin: 0 0 6px 0;
}}

.pl-callout .pl-callout-body {{
  color: #2A2E35;
  font-size: 13px;
  line-height: 1.45;
}}

/* Complementary Ã¢â‚¬Å“AWS console-ishÃ¢â‚¬Â hues */
.pl-callout.teal {{
  background: rgba(14, 165, 233, 0.06);
}}
.pl-callout.teal::before {{
  background: #0EA5E9;
}}

.pl-callout.violet {{
  background: rgba(139, 92, 246, 0.06);
}}
.pl-callout.violet::before {{
  background: #8B5CF6;
}}

.pl-callout.amber {{
  background: rgba(245, 158, 11, 0.07);
}}
.pl-callout.amber::before {{
  background: #F59E0B;
}}


</style>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# Demo DB (Files + Settings + Uploads)
# =============================================================================

def demo_db() -> sqlite3.Connection:
    DEMO_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DEMO_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          mode TEXT NOT NULL,
          user_id TEXT NOT NULL,
          kind TEXT NOT NULL,
          title TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
          mode TEXT NOT NULL,
          user_id TEXT NOT NULL,
          key TEXT NOT NULL,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (mode, user_id, key)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          mode TEXT NOT NULL,
          user_id TEXT NOT NULL,
          filename TEXT NOT NULL,
          mimetype TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          content_b64 TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    return conn


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def demo_save_artifact(mode: str, user_id: str, kind: str, title: str, payload: Dict[str, Any]) -> None:
    conn = demo_db()
    conn.execute(
        "INSERT INTO artifacts (mode, user_id, kind, title, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (mode, user_id, kind, title[:200], json.dumps(payload, ensure_ascii=False), _utc_now_iso()),
    )
    conn.commit()


def demo_list_artifacts(mode: str, user_id: str, kind: Optional[str] = None) -> List[sqlite3.Row]:
    conn = demo_db()
    if kind:
        cur = conn.execute(
            "SELECT * FROM artifacts WHERE mode=? AND user_id=? AND kind=? ORDER BY id DESC",
            (mode, user_id, kind),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM artifacts WHERE mode=? AND user_id=? ORDER BY id DESC",
            (mode, user_id),
        )
    return list(cur.fetchall())


def demo_delete_artifact(artifact_id: int) -> None:
    conn = demo_db()
    conn.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))
    conn.commit()


def demo_get_setting(mode: str, user_id: str, key: str, default: str = "") -> str:
    conn = demo_db()
    cur = conn.execute("SELECT value FROM settings WHERE mode=? AND user_id=? AND key=?", (mode, user_id, key))
    row = cur.fetchone()
    return row["value"] if row else default


def demo_set_setting(mode: str, user_id: str, key: str, value: str) -> None:
    conn = demo_db()
    conn.execute(
        "INSERT INTO settings (mode, user_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(mode,user_id,key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (mode, user_id, key, value, _utc_now_iso()),
    )
    conn.commit()


def demo_save_upload(mode: str, user_id: str, filename: str, mimetype: str, raw_bytes: bytes) -> None:
    conn = demo_db()
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    conn.execute(
        "INSERT INTO uploads (mode,user_id,filename,mimetype,size_bytes,content_b64,created_at) VALUES (?,?,?,?,?,?,?)",
        (mode, user_id, filename[:240], mimetype[:120], len(raw_bytes), b64, _utc_now_iso()),
    )
    conn.commit()


def demo_list_uploads(mode: str, user_id: str) -> List[sqlite3.Row]:
    conn = demo_db()
    cur = conn.execute(
        "SELECT id,filename,mimetype,size_bytes,created_at FROM uploads WHERE mode=? AND user_id=? ORDER BY id DESC",
        (mode, user_id),
    )
    return list(cur.fetchall())


def demo_get_upload_blob(upload_id: int) -> Tuple[str, str, bytes]:
    conn = demo_db()
    cur = conn.execute("SELECT filename,mimetype,content_b64 FROM uploads WHERE id=?", (upload_id,))
    row = cur.fetchone()
    if not row:
        return "", "", b""
    return row["filename"], row["mimetype"], base64.b64decode(row["content_b64"])


def demo_delete_upload(upload_id: int) -> None:
    conn = demo_db()
    conn.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
    conn.commit()


# =============================================================================
# Backend route discovery
# =============================================================================

@st.cache_data(show_spinner=False, ttl=10)
def fetch_openapi() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}/openapi.json", timeout=10)
        if 200 <= r.status_code < 300:
            return r.json()
    except Exception:
        pass
    return {}


def list_paths() -> List[str]:
    o = fetch_openapi()
    paths = o.get("paths", {}) or {}
    return sorted(list(paths.keys()))


def find_first_path(candidates: List[str]) -> str:
    paths = set(list_paths())
    for c in candidates:
        if c in paths:
            return c
    return ""


def endpoint_map() -> Dict[str, str]:
    return {
        "auth_login": find_first_path(["/auth/login", "/api/auth/login", "/login"]),
        # Explorer (consumer)
        "consumer_search": find_first_path(["/consumer/search", "/api/consumer/search", "/consumer/match"]),
        "consumer_profile_get": find_first_path(["/profile", "/consumer/profile", "/api/consumer/profile"]),
        "consumer_profile_save": find_first_path(["/profile", "/consumer/profile", "/api/consumer/profile", "/consumer/profile/update"]),
        "consumer_industry_suggest": find_first_path([
            "/consumer/industry_suggestions",
            "/consumer/industry-suggestions",
            "/consumer/suggest_industries",
            "/consumer/suggestions/industries",
        ]),
        "consumer_jd_pulse": find_first_path([
            "/consumer/jd_pulse",
            "/consumer/jd-pulse",
            "/consumer/jd/pulse",
            "/consumer/jd_analyze",
        ]),
        "consumer_live_hunt": find_first_path([
                        "/consumer/live-jobs",
"/consumer/live_hunt",
            "/consumer/live-hunt",
            "/consumer/live/search",
        ]),
        # Employer (enterprise/corporate)
        "enterprise_profile_get": find_first_path([
            "/profile",
            "/enterprise/profile",
            "/enterprise/company_profile",
            "/corporate/profile",
            "/corporate/company_profile",
        ]),
        "enterprise_profile_save": find_first_path([
            "/profile",
            "/enterprise/profile",
            "/enterprise/profile/update",
            "/enterprise/company_profile",
            "/corporate/profile",
            "/corporate/profile/update",
        ]),
        "enterprise_jd_studio_analyze": find_first_path([
                        "/corporate/analyze-jd",
"/enterprise/jd_studio/analyze",
            "/enterprise/jd_studio",
            "/enterprise/jd/analyze",
            "/corporate/jd/analyze",
        ]),
        "enterprise_jd_studio_rewrite": find_first_path([
                        "/enterprise/rewrite-jd",
"/enterprise/jd_studio/rewrite",
            "/enterprise/jd/rewrite",
            "/corporate/jd/rewrite",
        ]),
        "enterprise_comp_builder": find_first_path([
            "/enterprise/comp_builder",
            "/enterprise/comp_builder/generate",
            "/enterprise/comp",
            "/corporate/comp_builder",
        ]),
    }


def render_backend_debug_panel() -> None:
    with st.expander("Backend Debug (routes)", expanded=False):
        st.write(f"API_BASE: `{API_BASE}`")
        o = fetch_openapi()
        if not o:
            st.error("Could not fetch /openapi.json. Confirm backend is running on 8011.")
            return
        m = endpoint_map()
        st.write("Auto-selected endpoints (for wiring):")
        for k in sorted(m.keys()):
            v = m[k] or "(not found)"
            st.write(f"- {k}: `{v}`")


# =============================================================================
# OpenAI helpers
# =============================================================================

def openai_available() -> bool:
    return bool(OPENAI_API_KEY.strip())


def openai_generate(system: str, user: str, max_tokens: int = 1100) -> Tuple[bool, str]:
    if not openai_available():
        return False, "OPENAI_API_KEY not set."
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=70,
        )
        if 200 <= r.status_code < 300:
            return True, (r.json()["choices"][0]["message"]["content"] or "").strip()
        return False, f"OpenAI error {r.status_code}: {_safe_err(r)}"
    except Exception as e:
        return False, f"OpenAI error: {e}"


# =============================================================================
# Text extraction (resume + uploads)
# =============================================================================

def read_upload_bytes(upload) -> bytes:
    if upload is None:
        return b""
    return upload.read() or b""


def try_extract_text_from_bytes(filename: str, mimetype: str, raw: bytes) -> str:
    name = (filename or "").lower()

    # TXT
    if name.endswith(".txt") or mimetype.startswith("text/"):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(enc, errors="ignore")
            except Exception:
                continue
        return raw.decode("utf-8", errors="ignore")

    # DOCX
    if name.endswith(".docx") and docx is not None:
        try:
            from io import BytesIO
            d = docx.Document(BytesIO(raw))
            parts = [p.text for p in d.paragraphs if (p.text or "").strip()]
            return "\n".join(parts).strip()
        except Exception:
            return ""

    # PDF
    if name.endswith(".pdf") and PdfReader is not None:
        try:
            from io import BytesIO
            reader = PdfReader(BytesIO(raw))
            text_parts = []
            for page in reader.pages[:25]:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
            return "\n".join(text_parts).strip()
        except Exception:
            return ""

    # DOC (legacy) and anything else: not reliably extractable locally without extra tooling
    return ""


def extract_skills_from_text(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,25}", text)
    stop = {
        "the","and","with","for","from","that","this","have","has","will","your","you","are","was","were","not","but",
        "into","over","under","their","they","them","our"
    }
    out, seen = [], set()
    for t in tokens:
        lt = t.lower()
        if lt in stop or len(t) < 2:
            continue
        if lt in seen:
            continue
        seen.add(lt)
        out.append(t)

        if len(out) >= 50:
            break
    return out


# =============================================================================
# Auth + routing
# =============================================================================

def reset_auth() -> None:
    for k in ["auth_token", "user", "user_id", "authed_mode"]:
        if k in st.session_state:
            del st.session_state[k]


def ensure_authed_for_mode(mode: str) -> bool:
    return bool(st.session_state.get("auth_token")) and st.session_state.get("authed_mode") == mode


def nav_to(page: str) -> None:
    st.session_state["page"] = page


def current_user_email() -> str:
    u = st.session_state.get("user") or {}
    return str(u.get("email") or u.get("username") or u.get("user_email") or "")


# =============================================================================
# Premium UI helpers (no JSON)
# =============================================================================


def _html_escape(val: Any) -> str:
    """HTML-escape for unsafe_allow_html=True blocks."""
    try:
        return html.escape(str(val if val is not None else ""), quote=True)
    except Exception:
        return ""



def callout(title: str, body: str, tone: str = "teal") -> None:
    """Render a subtle, high-contrast response panel without changing app logic."""
    tone = (tone or "teal").strip().lower()
    if tone not in ("teal", "violet", "amber"):
        tone = "teal"
    st.markdown(
        f"""
        <div class="pl-callout {tone}">
          <div class="pl-callout-title">{_html_escape(title)}</div>
          <div class="pl-callout-body">{_html_escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, subtitle: str = "") -> None:
    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='pl-h2'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='pl-muted'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def bullets(items: List[str]) -> None:
    for s in [x for x in items if (x or "").strip()]:
        st.write(f"Ã¢â‚¬Â¢ {s}")


def render_job_detail(sel: Dict[str, Any]) -> None:
    title = sel.get("title") or sel.get("job_title") or "Untitled role"
    company = sel.get("company") or sel.get("company_name") or ""
    location = sel.get("location") or ""
    industry = sel.get("industry") or sel.get("industry_name") or ""
    salary = sel.get("salary") or sel.get("salary_range") or sel.get("salary_text") or ""
    desc = sel.get("description") or sel.get("jd_text") or sel.get("summary") or ""

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='pl-title'>{title}</div>", unsafe_allow_html=True)
    meta = " Ã¢â‚¬Â¢ ".join([x for x in [company, location, industry] if x])
    if meta:
        st.markdown(f"<div class='pl-subtitle'>{meta}</div>", unsafe_allow_html=True)
    if salary:
        st.markdown(f"<div class='pl-muted' style='margin-top:6px;'>Salary: {salary}</div>", unsafe_allow_html=True)

    st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
    if desc:
        st.markdown("<div class='pl-h2'>Job details</div>", unsafe_allow_html=True)
        st.write(desc)
    else:
        st.info("No job description text attached to this result.")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Profile (Explorer)
# =============================================================================

def _default_profile() -> Dict[str, Any]:
    return {
        "name": "",
        "location": "",
        "preferred_title": "",
        "salary_min": "",
        "skills": "",
        "interests": "",
        "experience": "",
        "education": "",
        "resume_text": "",
        "desired_industries": [],
    }


def get_profile() -> Dict[str, Any]:
    m = endpoint_map()
    gp = m.get("consumer_profile_get", "")
    if gp:
        res = api_get(gp)
        if res.ok and res.data:
            p = res.data.get("profile", res.data)
            out = _default_profile()
            out.update({k: p.get(k, out.get(k)) for k in out.keys()})
            if isinstance(p.get("desired_industries"), list):
                out["desired_industries"] = p.get("desired_industries")
            return out

    # Local fallback: load latest saved local profile if present
    out = _default_profile()
    try:
        items = demo_list_artifacts(MODE_EXPLORER, st.session_state.get("user_id", "demo"), kind="profile_local")
        if items:
            payload = json.loads(items[0]["payload_json"])
            if isinstance(payload, dict):
                out.update({k: payload.get(k, out.get(k)) for k in out.keys()})
    except Exception:
        pass
    return out


def save_profile(profile: Dict[str, Any]) -> Tuple[bool, str]:
    m = endpoint_map()
    sp = m.get("consumer_profile_save", "")
    if sp:
        res = api_post(sp, {"profile": profile})
        if res.ok:
            return True, ""
        res2 = api_post(sp, profile)
        if res2.ok:
            return True, ""
        return False, res.error or res2.error or "Unknown backend error"

    demo_save_artifact(MODE_EXPLORER, st.session_state["user_id"], "profile_local", "Profile (local)", profile)
    return True, ""


def render_explorer_profile_page() -> None:
    profile = get_profile()
    card("Profile", "This is your baseline. Match & Explore can add temporary context before searching.")
    updated = dict(profile)

    tabs = st.tabs(["Resume upload", "Manual"])

    with tabs[0]:
        st.markdown('<div class="pl-section">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Upload resume</div>", unsafe_allow_html=True)

        up = st.file_uploader(
            "Upload resume (PDF, DOCX, DOC, TXT)",
            type=["pdf", "docx", "doc", "txt"],
            key="ex_resume_upload",
        )

        extracted = ""
        if up is not None:
            raw = read_upload_bytes(up)
            demo_save_upload(MODE_EXPLORER, st.session_state["user_id"], up.name, up.type or "application/octet-stream", raw)

            extracted = try_extract_text_from_bytes(up.name, up.type or "", raw)
            if extracted.strip():
                st.success("Resume text extracted. You can draft skills automatically.")
                st.text_area("Extracted text (editable)", value=extracted, height=220, key="ex_resume_extracted")
                updated["resume_text"] = st.session_state.get("ex_resume_extracted", extracted)
            else:
                st.warning("Saved to Files, but could not extract text locally. Paste resume text below.")
                updated["resume_text"] = st.text_area("Paste resume text", value=updated.get("resume_text") or "", height=220, key="ex_resume_paste")

        else:
            updated["resume_text"] = st.text_area("Paste resume text (optional)", value=updated.get("resume_text") or "", height=220, key="ex_resume_paste")

        st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)

        st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
        if st.button("Draft skills from resume text", use_container_width=True):
            text = (updated.get("resume_text") or "").strip()
            if not text:
                st.warning("Add resume text first.")
            else:
                skills = extract_skills_from_text(text)
                updated["skills"] = ", ".join(skills[:35])
                if not updated.get("experience"):
                    updated["experience"] = "Drafted from resume. Refine as needed."
                st.success("Drafted skills. Review Manual tab and Save.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="pl-section">', unsafe_allow_html=True)
        updated["name"] = st.text_input("Name", value=updated.get("name") or "")
        updated["location"] = st.text_input("Location", value=updated.get("location") or "", placeholder="Remote or City, State")
        updated["salary_min"] = st.text_input("Target minimum salary (optional)", value=str(updated.get("salary_min") or ""), placeholder="e.g., 120000")
        updated["skills"] = st.text_area("Skills (comma-separated)", value=updated.get("skills") or "", height=90)
        updated["interests"] = st.text_area("Interests (comma-separated)", value=updated.get("interests") or "", height=70)
        updated["experience"] = st.text_area("Experience summary", value=updated.get("experience") or "", height=110)
        updated["education"] = st.text_area("Education", value=updated.get("education") or "", height=80)

        # Desired industries up to 3 (freeform via select + text)
        preset = ["Technology", "Healthcare", "Education", "Government", "Finance", "Manufacturing", "Retail", "Energy", "Transportation", "Media"]
        desired = updated.get("desired_industries") or []
        if not isinstance(desired, list):
            desired = []
        options = sorted(list(set(preset + desired)))
        picks = st.multiselect("Desired industries (choose up to 3)", options=options, default=desired[:3], key="ex_ind_pick")[:3]
        updated["desired_industries"] = picks
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    if st.button("Save profile", use_container_width=True, key="ex_save_profile"):
        ok, err = save_profile(updated)
        if ok:
            st.success("Profile saved.")
        else:
            st.error(f"Save failed: {err}")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Match & Explore (inline context + Search by profile + 2 lanes)
# =============================================================================

def _normalize_results(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[Dict[str, Any]] = []
        for x in raw:
            if isinstance(x, dict):
                out.append(x)
        return out
    return []


def _bucket_by_industry(results: List[Dict[str, Any]], desired_industries: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    desired = [d.strip().lower() for d in (desired_industries or []) if d.strip()]
    if not desired:
        # If user didn't choose industries, treat all as primary
        return results, []

    primary: List[Dict[str, Any]] = []
    adjacent: List[Dict[str, Any]] = []

    for r in results:
        ind = str(r.get("industry") or r.get("industry_name") or "").lower()
        hay = " ".join([
            str(r.get("title") or r.get("job_title") or ""),
            str(r.get("company") or r.get("company_name") or ""),
            str(r.get("location") or ""),
            ind,
            str(r.get("description") or r.get("summary") or ""),
        ]).lower()

        if ind and any(d in ind for d in desired):
            primary.append(r)
        elif any(d in hay for d in desired):
            primary.append(r)
        else:
            adjacent.append(r)

    return primary, adjacent


def build_search_context_inline(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Inline context (always visible). Returns (ctx, search_by_profile_clicked).
    """
    ctx = dict(profile)

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Search by profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='pl-muted'>Use your profile, then optionally add extra context to refine results.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    search_by_profile = st.button("Search by profile", use_container_width=True, key="mx_search_by_profile")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Add context (optional)</div>", unsafe_allow_html=True)
    st.caption("This affects search results. Save if you want it to become your Profile defaults.")

    ctx["skills"] = st.text_area("Skills (comma-separated)", value=ctx.get("skills") or "", height=80, key="mx_ctx_skills")
    ctx["education"] = st.text_area("Education / certs", value=ctx.get("education") or "", height=60, key="mx_ctx_edu")
    ctx["interests"] = st.text_area("Interests", value=ctx.get("interests") or "", height=50, key="mx_ctx_interests")
    ctx["experience"] = st.text_area("Experience summary", value=ctx.get("experience") or "", height=80, key="mx_ctx_exp")

    preset = ["Technology", "Healthcare", "Education", "Government", "Finance", "Manufacturing", "Retail", "Energy", "Transportation", "Media"]
    desired = ctx.get("desired_industries") or []
    if not isinstance(desired, list):
        desired = []
    options = sorted(list(set(preset + desired)))
    ctx["desired_industries"] = st.multiselect(
        "Desired industries (choose up to 3)",
        options=options,
        default=desired[:3],
        key="mx_ctx_ind",
    )[:3]

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
        if st.button("Save to Profile", use_container_width=True, key="mx_save_ctx_profile"):
            ok, err = save_profile(ctx)
            if ok:
                st.success("Saved to Profile.")
            else:
                st.error(err)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.caption("Or keep it temporary and just search.")

    st.markdown("</div>", unsafe_allow_html=True)

    return ctx, search_by_profile


def _render_result_list(results: List[Dict[str, Any]], lane_key: str, selected_key: str) -> None:
    for i, item in enumerate(results):
        title = item.get("title") or item.get("job_title") or "Untitled role"
        company = item.get("company") or item.get("company_name") or ""
        location = item.get("location") or ""
        industry = item.get("industry") or ""

        st.markdown("<div class='pl-card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight:900; color:{INK}; font-size:14px;'>{title}</div>", unsafe_allow_html=True)
        meta = " Ã¢â‚¬Â¢ ".join([x for x in [company, location, industry] if x])
        if meta:
            st.markdown(f"<div class='pl-muted'>{meta}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([0.6, 0.4], gap="small")
        with c1:
            if st.button("View", key=f"{lane_key}_view_{i}", use_container_width=True):
                st.session_state[selected_key] = i
                st.rerun()
        with c2:
            if st.button("Save", key=f"{lane_key}_save_{i}", use_container_width=True):
                demo_save_artifact(MODE_EXPLORER, st.session_state["user_id"], "saved_job", title, item)
                st.success("Saved to Files.")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_industry_suggestions(ctx: Dict[str, Any], query: str, location: str) -> None:
    """
    Backend if present, else OpenAI fallback (optional).
    """
    m = endpoint_map()
    sug_path = m.get("consumer_industry_suggest", "")

    suggestions: List[Dict[str, Any]] = []

    if sug_path:
        sres = api_post(sug_path, {"profile": ctx, "query": query, "location": location}, timeout=60)
        if sres.ok:
            raw = sres.data.get("suggestions") or sres.data.get("results") or []
            if isinstance(raw, list):
                for x in raw[:10]:
                    if isinstance(x, dict):
                        suggestions.append(x)
        return _render_industry_suggestions_ui(suggestions)

    # Optional AI fallback if endpoint missing
    if not openai_available():
        return

    ok, text = openai_generate(
        system="You are a career strategist. Return a short list with headings and bullets. No JSON.",
        user=(
            "Based on this profile, suggest 6 industries where this person could excel that they might not have considered. "
            "For each: give 1 sentence why it fits.\n\n"
            f"Profile skills: {ctx.get('skills','')}\n"
            f"Education: {ctx.get('education','')}\n"
            f"Experience: {ctx.get('experience','')}\n"
            f"Interests: {ctx.get('interests','')}\n"
            f"Target industries: {', '.join(ctx.get('desired_industries') or [])}\n"
            f"Target query: {query}\n"
        ),
        max_tokens=450,
    )
    if ok and text.strip():
        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Industries where your skills can shine (you may not have considered)</div>", unsafe_allow_html=True)
        callout("Recommendations", text, tone="violet")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_industry_suggestions_ui(suggestions: List[Dict[str, Any]]) -> None:
    if not suggestions:
        return
    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Industries where your skills can shine (you may not have considered)</div>", unsafe_allow_html=True)
    for s in suggestions[:8]:
        name = s.get("industry") or s.get("name") or ""
        why = s.get("why") or s.get("rationale") or ""
        if name:
            st.markdown(f"<div style='font-weight:900; color:{INK};'>{name}</div>", unsafe_allow_html=True)
        if why:
            st.markdown(f"<div class='pl-muted'>{why}</div>", unsafe_allow_html=True)
        st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)




def _render_career_paths_suggestions(ctx: Dict[str, Any], query: str) -> None:
    """
    Rich, persistent career-path + roadmap helper for Match & Explore.
    - Always safe to call (renders nothing if OpenAI missing).
    - Stores outputs in session_state so the page does NOT look like it reset after a click.
    """
    if not openai_available():
        return

    name = (ctx.get("name") or "").strip()
    who = name if name else "you"

    # Persist last known context + query so button clicks don't "lose" the panel on rerun.
    st.session_state["mx_last_ctx"] = ctx
    st.session_state["mx_last_q"] = query

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Career paths you may not have considered</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='pl-muted'>High-detail pathways based on your profile and search intent. "
        "Includes realistic steps (education, experience, portfolio, networking) and role examples.</div>",
        unsafe_allow_html=True,
    )

    # Render any previously generated content first (so it persists across reruns)
    prior = (st.session_state.get("mx_career_paths_text") or "").strip()
    if prior:
        callout("Career paths", prior, tone="teal")

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    gen_paths = st.button("Generate detailed career paths", use_container_width=True, key="mx_gen_paths")
    st.markdown("</div>", unsafe_allow_html=True)

    if gen_paths:
        with st.spinner("Generating detailed career paths..."):
            ok, text = openai_generate(
                system="You are a meticulous career strategist. Write in clear sections with headings. No JSON.",
                user=(
                    f"Write HIGHLY DETAILED career-path guidance for {who}.\n\n"
                    "Requirements:\n"
                    "Ã¢â‚¬Â¢ Suggest 6Ã¢â‚¬â€œ8 career paths the person may not have considered, including adjacent roles and specialized niches.\n"
                    "Ã¢â‚¬Â¢ For EACH path: write 3Ã¢â‚¬â€œ4 paragraphs (not bullets-only).\n"
                    "Ã¢â‚¬Â¢ Each path must include:\n"
                    "  - Why it's a fit (based on their skills/experience)\n"
                    "  - Typical titles and where these roles live (teams/industries)\n"
                    "  - What to learn next (specific skill categories, cert types, portfolio artifacts)\n"
                    "  - A step-by-step plan for the next 30/60/90 days\n"
                    "  - A \"bridge\" option: a role they can realistically get sooner that leads into the target path\n\n"
                    "Also: include a short section up front called 'Industry expansions' that lists 4 industries "
                    "they could pivot into and WHY (2Ã¢â‚¬â€œ3 paragraphs per industry).\n\n"
                    f"Search intent / role keywords: {query}\n\n"
                    f"Profile name: {ctx.get('name','')}\n"
                    f"Location: {ctx.get('location','')}\n"
                    f"Target salary minimum: {ctx.get('salary_min','')}\n"
                    f"Skills: {ctx.get('skills','')}\n"
                    f"Interests: {ctx.get('interests','')}\n"
                    f"Education: {ctx.get('education','')}\n"
                    f"Experience: {ctx.get('experience','')}\n"
                    f"Desired industries: {', '.join(ctx.get('desired_industries') or [])}\n"
                ),
                max_tokens=1500,
            )
        if ok and text.strip():
            st.session_state["mx_career_paths_text"] = text.strip()
            callout("Career paths", st.session_state["mx_career_paths_text"], tone="teal")
            demo_save_artifact(
                MODE_EXPLORER,
                st.session_state.get("user_id", "demo"),
                "career_paths",
                f"Career paths: {(query or 'profile')}",
                {"query": query, "text": text[:6000]},
            )
        else:
            st.error(text or "Could not generate career paths.")

    # Roadmap generation (separate, based on the generated paths)
    st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Career roadmaps</div>", unsafe_allow_html=True)
    st.markdown("<div class='pl-muted'>Creates a structured roadmap for each suggested path without resetting results.</div>", unsafe_allow_html=True)

    prior_rm = (st.session_state.get("mx_roadmap_text") or "").strip()
    if prior_rm:
        callout("Career roadmaps", prior_rm, tone="teal")

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    gen_rm = st.button("Generate career roadmaps", use_container_width=True, key="mx_gen_roadmaps")
    st.markdown("</div>", unsafe_allow_html=True)

    if gen_rm:
        base = (st.session_state.get("mx_career_paths_text") or "").strip()
        if not base:
            st.warning("Generate detailed career paths first.")
        else:
            with st.spinner("Building roadmaps..."):
                ok, text = openai_generate(
    system="You are a career coach. Return readable sections with headings. No JSON.",
    user=(
        "You already suggested career paths. Now build a detailed career roadmap for EACH suggested path. "
        "For each path, write 3Ã¢â‚¬â€œ4 paragraphs and then a step-by-step plan with milestones (30 days, 90 days, 6 months, 12 months). "
        "Include: skills to build, certifications/education options, portfolio projects, networking moves, and how to translate current experience. "
        "Also include 3 example job titles to target next and 3 keywords to search on job boards. "
        "Be specific and practical. Do NOT return JSON.\n\n"
        "Use these as anchors if relevant: the industries present in the current search results and the user's desired industries.\n\n"
        f"Industries seen in results: {', '.join(_parse_industry_list(st.session_state.get('mx_industries_seen', [])))}\n"
        f"Desired industries: {', '.join(ctx.get('desired_industries') or [])}\n"
        f"Target query: {query}\n\n"
        f"Profile summary:\n"
        f"- Skills: {ctx.get('skills','')}\n"
        f"- Education: {ctx.get('education','')}\n"
        f"- Experience: {ctx.get('experience','')}\n"
        f"- Interests: {ctx.get('interests','')}\n"
    ),
    max_tokens=1600,
)
            if ok and text.strip():
                st.session_state["mx_roadmap_text"] = text.strip()
                callout("Career roadmaps", st.session_state["mx_roadmap_text"], tone="teal")
                demo_save_artifact(
                    MODE_EXPLORER,
                    st.session_state.get("user_id", "demo"),
                    "career_roadmaps",
                    f"Career roadmaps: {(query or 'profile')}",
                    {"query": query, "text": text[:6000]},
                )
            else:
                st.error(text or "Could not generate roadmaps.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_match_explore_page() -> None:
    profile = get_profile()
    m = endpoint_map()

    card("Match & Explore", "Parquet-backed matching + industry intelligence. No JSON.")
    render_backend_debug_panel()

    # Inline context
    ctx, search_by_profile = build_search_context_inline(profile)

    # Search bar
    a, b, c = st.columns([0.52, 0.32, 0.16], gap="small")
    with a:
        q = st.text_input("Role / keywords", key="mx_q", placeholder="e.g., Instructional Designer, Training Manager")
    with b:
        loc = st.text_input("Location", key="mx_loc", placeholder=profile.get("location") or "Remote / City, State")
    with c:
        st.markdown('<div class="pl-primary" style="margin-top: 26px;">', unsafe_allow_html=True)
        go = st.button("Search", use_container_width=True, key="mx_go")
        st.markdown("</div>", unsafe_allow_html=True)

    if "mx_results_adjacent" not in st.session_state:
        st.session_state["mx_results_adjacent"] = []
    if "mx_selected_primary" not in st.session_state:
        st.session_state["mx_selected_primary"] = 0
    if "mx_selected_adjacent" not in st.session_state:
        st.session_state["mx_selected_adjacent"] = 0

    # Resolve once for both Match & Explore and Live Hunt
    search_path = m.get("consumer_search", "")
    if not search_path:
        st.error("Consumer search endpoint not found in backend OpenAPI.")
        st.stop()

    run_search = go or search_by_profile
    if run_search:
        # Use query if provided; otherwise, build a reasonable query from profile/context.
        q_effective = (q or "").strip()
        if not q_effective and search_by_profile:
            try:
                q_effective = build_live_query_from_profile(ctx)
            except Exception:
                q_effective = ""

        if not q_effective:
            st.warning("Enter keywords (or use Search by profile) to run a match search.")
            st.stop()

        loc_effective = (loc or "").strip() or (profile.get("location") or "").strip()

        payload = {
            "query": q_effective,
            "location": loc_effective,
            "profile": ctx,
            "limit": 40,
        }

        with st.spinner("Searching matches..."):
            res = api_post(search_path, payload, timeout=90)

        if not res.ok:
            st.error(f"Search failed: {res.error}")
            st.stop()

        # Backend may return results/matches OR primary/adjacent, handle both
        raw_primary = (
            res.data.get("primary_results")
            or res.data.get("primary")
            or res.data.get("results")
            or res.data.get("matches")
        )
        raw_adjacent = res.data.get("adjacent_results") or res.data.get("adjacent") or []

        primary = _normalize_results(raw_primary)
        adjacent = _normalize_results(raw_adjacent)

        # If backend only returns one list, bucket it ourselves
        if primary and not adjacent and not (
            res.data.get("primary_results") or res.data.get("adjacent_results")
        ):
            primary, adjacent = _bucket_by_industry(primary, ctx.get("desired_industries") or [])

        st.session_state["mx_results_primary"] = primary
        st.session_state["mx_results_adjacent"] = adjacent

        # store industries seen for roadmap generation
        inds_seen: List[str] = []
        for _r in (primary + adjacent):
            _ind = str(_r.get("industry") or _r.get("industry_name") or "").strip()
            if _ind:
                inds_seen.append(_ind)
        st.session_state["mx_industries_seen"] = inds_seen
        st.session_state["mx_selected_primary"] = 0
        st.session_state["mx_selected_adjacent"] = 0

        demo_save_artifact(
            MODE_EXPLORER,
            st.session_state["user_id"],
            "search",
            f"Search: {(q_effective or 'profile')}",
            {
                "query": q_effective,
                "location": loc_effective,
                "desired_industries": ctx.get("desired_industries") or [],
                "counts": {"primary": len(primary), "adjacent": len(adjacent)},
            },
        )

        # Render industry suggestions (backend if exists, AI fallback if configured)
        _render_industry_suggestions(ctx, q_effective, loc_effective)

    # --- Live Hunt (quick search) ---
    lh_q = st.text_input("Search roles (e.g. Instructional Designer)", key="live_hunt_query")
    lh_loc = st.text_input("Location (optional)", key="live_hunt_location")

    if st.button("Run Live Hunt"):
        live_path = m.get("consumer_live_hunt", "")
        if not live_path:
            st.error("Live Hunt endpoint not found in backend OpenAPI.")
            st.stop()

        if not (lh_q or "").strip():
            st.warning("Enter a search query to run Live Hunt")
            st.stop()

        live_payload: Dict[str, Any] = {
            "query": lh_q.strip(),
            "location": (lh_loc or "").strip(),
            "limit": 24,
        }

        # Optional salary min (from profile) if present and parseable
        try:
            sm = str(profile.get("salary_min") or "").strip()
            if sm:
                live_payload["salary_min"] = int(re.sub(r"[^0-9]", "", sm) or "0") or None
        except Exception:
            pass

        with st.spinner("Running Live Hunt..."):
            res = api_post(live_path, live_payload, timeout=90)

        if not res.ok:
            st.error(f"Live Hunt failed: {res.error}")
            st.stop()

        # Persist for the dedicated Live Hunt page, too
        st.session_state["lh_results"] = (
            res.data.get("results")
            or res.data.get("jobs")
            or res.data.get("matches")
            or []
        )
        st.session_state["lh_last_query"] = lh_q.strip()
        st.session_state["lh_last_location"] = (lh_loc or "").strip()

        # If your renderer expects the raw backend shape, fall back
        try:
            if "render_live_hunt_results" in globals():
                render_live_hunt_results(res.data)
            else:
                st.success("Live Hunt ran successfully. Open the Live Hunt page to review results.")
        except Exception as e:
            st.error(f"Live Hunt ran, but the renderer errored: {e}")
    primary = st.session_state.get("mx_results_primary") or []
    adjacent = st.session_state.get("mx_results_adjacent") or []

    # Keep the high-detail career panel visible across reruns (e.g., after clicking a button).
    # If the user clicked a button inside the panel, run_search will be False on that rerun.
    if (primary or adjacent):
        ctx_last = st.session_state.get("mx_last_ctx") or ctx
        q_last = st.session_state.get("mx_last_q") or q
        _render_career_paths_suggestions(ctx_last, q_last)


    if not primary and not adjacent:
        st.info("Run a search to see results.")
        return

    st.markdown("<div class='pl-card'><div class='pl-h2'>Results</div><div class='pl-muted'>Left = your target industries. Right = adjacent industries where your skills still fit strongly.</div></div>", unsafe_allow_html=True)

    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown("<div class='pl-card'><div class='pl-h2'>Jobs in my industry</div><div class='pl-muted'>Aligned to your chosen industries / desired field.</div></div>", unsafe_allow_html=True)
        if not primary:
            st.info("No results in your selected industries. Try broader keywords, or adjust desired industries.")
        else:
            _render_result_list(primary, "mxp", "mx_selected_primary")
            st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
            sel = primary[int(st.session_state.get("mx_selected_primary", 0))]
            render_job_detail(sel)

            st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
            if st.button("Send to JD Pulse", use_container_width=True, key="mx_primary_to_pulse"):
                st.session_state["jd_seed"] = sel.get("description") or sel.get("jd_text") or sel.get("summary") or ""
                nav_to("explorer.pulse")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='pl-card'><div class='pl-h2'>Jobs outside my industry</div><div class='pl-muted'>Matches your skills and interests strongly.</div></div>", unsafe_allow_html=True)
        if not adjacent:
            st.info("No adjacent matches returned yet.")
        else:
            _render_result_list(adjacent, "mxa", "mx_selected_adjacent")
            st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
            sel = adjacent[int(st.session_state.get("mx_selected_adjacent", 0))]
            render_job_detail(sel)

            st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
            if st.button("Send to JD Pulse", use_container_width=True, key="mx_adj_to_pulse"):
                st.session_state["jd_seed"] = sel.get("description") or sel.get("jd_text") or sel.get("summary") or ""
                nav_to("explorer.pulse")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# JD Pulse (backend if present, otherwise OpenAI fallback; no JSON)
# =============================================================================

def render_jd_pulse_page() -> None:
    profile = get_profile()
    m = endpoint_map()

    card("JD Pulse", "Should you apply? What to watch for. Fit, risk, and negotiation prep. No rewrites. No JSON.")

    # Optional seed from Match/Live Hunt
    seed = st.session_state.get("jd_seed", "") or ""
    source_url = st.session_state.get("jd_source_url", "") or ""

    jd = st.text_area("Paste job description", value=seed, height=280, key="pulse_jd")

    # Best-effort: try to fetch more text from a listing URL (many sites block scraping; this is optional)
    if source_url:
        st.markdown("<div class='pl-card'>", unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Source</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pl-muted'>Listing link saved from Live Hunt.</div>", unsafe_allow_html=True)
        st.link_button("Open listing", source_url, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Show last output (so it persists even if the page reruns)
    last = (st.session_state.get("pulse_last_text") or "").strip()
    if last:
        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Most recent JD Pulse</div>", unsafe_allow_html=True)
        callout("Most recent JD Pulse", last, tone="violet")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    run = st.button("Run JD Pulse", use_container_width=True, key="pulse_run")
    st.markdown("</div>", unsafe_allow_html=True)
    if not run:
        return

    if not (jd or "").strip():
        st.warning("Paste a job description first.")
        return

    path = m.get("consumer_jd_pulse", "")

    # Prefer backend (if route exists)
    if path:
        with st.spinner("Running JD Pulse..."):
            res = api_post(path, {"jd_text": jd, "profile": profile}, timeout=90)
        if not res.ok:
            st.error(f"JD Pulse failed: {res.text or res.status}")
            return

        report = res.data.get("report") or res.data.get("pulse") or res.data

        # Try to support multiple backend shapes, but ALWAYS render a text-heavy output
        parts: List[str] = []

        summary = report.get("summary") or report.get("overview") or ""
        score = report.get("score") or report.get("quality_score")
        risks = report.get("risks") or report.get("red_flags") or []
        fit = report.get("fit") or report.get("fit_notes") or ""
        comp = report.get("compensation") or report.get("salary_notes") or ""
        tactics = report.get("negotiation") or report.get("negotiation_tactics") or ""
        questions = report.get("questions") or report.get("questions_to_ask") or []

        if summary:
            parts.append("## Executive summary\n" + str(summary).strip())

        if score is not None:
            parts.append(f"## Opportunity score\n{score}")

        if fit:
            parts.append("## Fit vs your profile\n" + str(fit).strip())

        # If backend gives "issues", reinterpret them as risks/clarity problems, not rewrites
        issues = report.get("issues") or []
        if issues:
            lines: List[str] = ["## Red flags, ambiguity, and what it signals"]
            for it in issues[:14]:
                if not isinstance(it, dict):
                    continue
                title = it.get("title") or "Issue"
                why = it.get("why") or it.get("detail") or ""
                impact = it.get("impact") or ""
                lines.append(f"**{title}**")
                if why:
                    lines.append(str(why).strip())
                if impact:
                    lines.append(f"Impact: {impact}")
                lines.append("")
            parts.append("\n".join(lines).strip())

        if risks and isinstance(risks, list):
            lines = ["## Red flags and risk signals"]
            for r in risks[:14]:
                if isinstance(r, str) and r.strip():
                    lines.append(f"Ã¢â‚¬Â¢ {r.strip()}")
            parts.append("\n".join(lines).strip())

        # Negotiation + offer prep
        parts.append(
            "## Negotiation prep (if you get an offer)\n"
            "Ã¢â‚¬Â¢ What to negotiate: base pay, signing, equity/bonus, leveling, remote flexibility, start date, title, learning budget, PTO\n"
            "Ã¢â‚¬Â¢ How to anchor: tie your anchor to outcomes you can deliver in 90 days\n"
            "Ã¢â‚¬Â¢ What to ask for in writing: job level, reporting line, bonus plan details, benefits, remote policy"
        )

        if tactics:
            parts.append("## Negotiation tactics tailored to this posting\n" + str(tactics).strip())
        if comp:
            parts.append("## Compensation clues in the posting\n" + str(comp).strip())
        if questions and isinstance(questions, list):
            lines = ["## Questions to ask in the first interview"]
            for q in questions[:12]:
                if isinstance(q, str) and q.strip():
                    lines.append(f"Ã¢â‚¬Â¢ {q.strip()}")
            parts.append("\n".join(lines).strip())

        # If backend returned a single text field, prefer it (but still keep our structure)
        text_blob = report.get("analysis_text") or report.get("text") or ""
        if text_blob and isinstance(text_blob, str) and len(text_blob.strip()) > 80:
            parts.append("## Additional analysis\n" + text_blob.strip())

        out_text = "\n\n".join([p for p in parts if p.strip()]).strip()
        if not out_text:
            out_text = "JD Pulse ran, but the backend did not return a readable report format."

        st.session_state["pulse_last_text"] = out_text
        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>JD Pulse output</div>", unsafe_allow_html=True)
        callout("Output", out_text, tone="violet")
        st.markdown("</div>", unsafe_allow_html=True)

        demo_save_artifact(MODE_EXPLORER, st.session_state.get("user_id", "demo"), "jd_pulse", f"JD Pulse {_utc_now_iso()}", {"mode": "backend"})
        return

    # Fallback: OpenAI (so JD Pulse is never dead)
    if not openai_available():
        st.error("JD Pulse backend endpoint is missing and OPENAI_API_KEY is not set.")
        return

    name = (profile.get("name") or "").strip()
    who = name if name else "you"

    with st.spinner("Running JD Pulse (AI)..."):
        ok, text = openai_generate(
            system="You are a pragmatic career coach and recruiter. Output must be deep, structured, and not JSON.",
            user=(
                f"Analyze this job description for {who}. DO NOT rewrite the JD.\n\n"
                "Output requirements (be very detailed):\n"
                "1) Should I apply? Provide a yes/no/lean-yes/lean-no verdict and explain.\n"
                "2) Fit analysis vs my profile (strengths, gaps, and how risky those gaps are).\n"
                "3) Red flags and hidden expectations (what the wording suggests).\n"
                "4) What I should clarify in interview (12 specific questions).\n"
                "5) Offer strategy: negotiation levers + how to phrase asks (give example scripts).\n"
                "6) Compensation signals: what to infer, what not to infer, and what to request.\n"
                "7) 90-day success plan: what I would deliver if hired.\n\n"
                f"My profile context:\n"
                f"Name: {profile.get('name','')}\n"
                f"Location: {profile.get('location','')}\n"
                f"Target minimum salary: {profile.get('salary_min','')}\n"
                f"Skills: {profile.get('skills','')}\n"
                f"Education: {profile.get('education','')}\n"
                f"Experience: {profile.get('experience','')}\n"
                f"Interests: {profile.get('interests','')}\n\n"
                f"JD:\n{jd}"
            ),
            max_tokens=1600,
        )

    if not ok:
        st.error(text)
        return

    st.session_state["pulse_last_text"] = text.strip()
    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>JD Pulse output</div>", unsafe_allow_html=True)
    callout("JD Pulse output", st.session_state["pulse_last_text"], tone="violet")
    st.markdown("</div>", unsafe_allow_html=True)

    demo_save_artifact(MODE_EXPLORER, st.session_state.get("user_id", "demo"), "jd_pulse", f"JD Pulse {_utc_now_iso()}", {"mode": "ai_fallback"})



def _parse_industry_list(value: Any) -> List[str]:
    """Accepts list or comma/newline-separated string and returns a clean list of industries."""
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
    else:
        s = str(value)
        parts = re.split(r"[\n,;]+", s)
        items = [p.strip() for p in parts if p.strip()]
    out: List[str] = []
    seen = set()
    for it in items:
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= 10:
            break
    return out

# =============================================================================
# Live Hunt (Gate screen + split into 2 sections)
# =============================================================================

def serpapi_available() -> bool:
    return bool(SERPAPI_API_KEY.strip())


def _bucket_live_jobs(results: List[Dict[str, Any]], desired_industries: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    desired = [d.strip().lower() for d in (desired_industries or []) if d.strip()]
    if not desired:
        return results, []
    in_industry, out_industry = [], []
    for r in results:
        hay = " ".join([
            str(r.get("title","")),
            str(r.get("snippet","")),
            str(r.get("company","")),
        ]).lower()
        if any(d in hay for d in desired):
            in_industry.append(r)
        else:
            out_industry.append(r)
    return in_industry, out_industry



# Live Hunt helpers
def _parse_industry_list(raw: Any) -> List[str]:
    """Accepts list or comma-separated string; returns cleaned list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[;,\n\|]+", s) if p.strip()]
    out, seen = [], set()
    for p in parts:
        lp = p.lower()
        if lp in seen:
            continue
        seen.add(lp)
        out.append(p)
    return out[:6]


def build_live_query_from_profile(profile: Dict[str, Any]) -> str:
    """Build a compact keyword query for SerpAPI using profile signals."""
    skills = str(profile.get("skills") or "").strip()
    interests = str(profile.get("interests") or "").strip()
    experience = str(profile.get("experience") or "").strip()
    desired = _parse_industry_list(profile.get("desired_industries") or [])

    preferred_title = str(profile.get("preferred_title") or profile.get("desired_title") or profile.get("job_title") or "").strip()


    role_hint = preferred_title
    if (not role_hint) and experience:
        role_hint = re.split(r"[\n\.;]+", experience)[0].strip()

    def _tokenize_terms(s: str, limit: int) -> str:
        parts = [p.strip() for p in re.split(r"[\n,;/|]+", s or "") if p.strip()]
        return " ".join(parts[:limit])

    skill_terms = _tokenize_terms(skills, 8)
    interest_terms = _tokenize_terms(interests, 6)

    chunks = []
    if role_hint and len(role_hint.split()) <= 10:
        chunks.append(role_hint)
    if skill_terms:
        chunks.append(skill_terms)
    if interest_terms:
        chunks.append(interest_terms)
    if desired:
        chunks.append(" ".join(desired[:3]))

    q = " ".join([c for c in chunks if c]).strip()
    return q or "jobs"




# -----------------------------
# Live Hunt debug helpers
# -----------------------------
def _lh_set_debug(provider: str, q: str, location: str, error: str = "", details: Optional[Dict[str, Any]] = None) -> None:
    st.session_state["lh_last_provider"] = provider
    st.session_state["lh_last_q"] = q
    st.session_state["lh_last_loc"] = location
    st.session_state["lh_last_error"] = error or ""
    st.session_state["lh_last_details"] = details or {}

def serp_ai_search(q: str, location: str = "") -> List[Dict[str, Any]]:
    """Fallback live job search using SerpAPI (google_jobs engine).

    Returns a list of dicts with: title, company, location, url, snippet.
    """
    if not (q or "").strip():
        return []
    if not SERPAPI_API_KEY:
        _lh_set_debug("serpapi", q, location, error="SERPAPI_API_KEY is not set. Configure env var SERPAPI_API_KEY.")
        return []

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_jobs",
        "q": q,
        "location": location or "",
        "hl": "en",
        "api_key": SERPAPI_API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=45)
        if r.status_code < 200 or r.status_code >= 300:
            return []
        data = r.json() if r.content else {}
        _lh_set_debug("serpapi", q, location)
    except Exception as e:
        _lh_set_debug("serpapi", q, location, error=f"SerpAPI request failed: {e}")
        return []

    out: List[Dict[str, Any]] = []

    jobs = data.get("jobs_results") or data.get("jobs") or []
    if isinstance(jobs, list):
        for j in jobs[:20]:
            if not isinstance(j, dict):
                continue
            title = (j.get("title") or "").strip()
            company = (j.get("company_name") or j.get("company") or "").strip()
            loc = (j.get("location") or "").strip()
            desc = (j.get("description") or j.get("snippet") or "").strip()
            apply_options = j.get("apply_options") or []
            link = (j.get("related_links") or [])
            url_val = (j.get("job_id") or "").strip()

            # Prefer explicit apply link when present
            apply_link = ""
            if isinstance(apply_options, list) and apply_options:
                for opt in apply_options:
                    if isinstance(opt, dict) and opt.get("link"):
                        apply_link = str(opt.get("link")).strip()
                        break

            # SerpAPI google_jobs often provides a tracking link via apply_options.
            final_url = apply_link
            if not final_url:
                # Try generic 'link' fields
                for k in ("share_link", "job_link", "link", "url"):
                    if j.get(k):
                        final_url = str(j.get(k)).strip()
                        break

            out.append({
                "title": title,
                "company": company,
                "location": loc,
                "url": final_url,
                "snippet": desc,
            })

    # Fallback: organic results if jobs_results missing
    if not out:
        organic = data.get("organic_results") or []
        if isinstance(organic, list):
            for j in organic[:20]:
                if not isinstance(j, dict):
                    continue
                out.append({
                    "title": (j.get("title") or "").strip(),
                    "company": "",
                    "location": "",
                    "url": (j.get("link") or "").strip(),
                    "snippet": (j.get("snippet") or "").strip(),
                })

    # Filter out empty rows
    out = [x for x in out if (x.get("title") or x.get("url"))]

    return out


def _parse_company_profile_autofill(text: str) -> Dict[str, str]:
    """Best-effort parse of AI 'autofill' text into company profile fields."""
    if not (text or "").strip():
        return {}
    t = text.strip()
    sections: Dict[str, str] = {}
    current = None
    buf = []

    heading_map = {
        "about": "about",
        "about the company": "about",
        "values": "values",
        "values / culture": "values",
        "culture": "values",
        "successes": "successes",
        "successes / proof points": "successes",
        "proof points": "successes",
        "benefits": "benefits",
        "benefits / perks": "benefits",
        "perks": "benefits",
        "brand voice": "brand_voice",
        "brand voice for jds": "brand_voice",
        "eeo": "eeo_statement",
        "eeo statement": "eeo_statement",
        "compliance": "compliance_notes",
        "compliance notes": "compliance_notes",
    }

    def flush():
        nonlocal current, buf
        if current and buf:
            sections[current] = "\n".join(buf).strip()
        buf = []

    for line in t.splitlines():
        s = line.strip()
        low = s.lower().strip(" :#*-")
        key = None
        for h, k in heading_map.items():
            if low == h:
                key = k
                break
        if key:
            flush()
            current = key
            continue
        if current:
            buf.append(line)

    flush()

    if not sections:
        return {"about": t}
    return sections


# Live Hunt defaults
loc_default = ""  # empty = no forced location

def render_live_hunt_page() -> None:
    profile = get_profile()
    m = endpoint_map()

    card("Live Hunt", "Find live jobs with working links. Import your profile and/or Match results, or search manually.")

    # --- Session defaults
    st.session_state.setdefault("lh_gate_done", False)
    st.session_state.setdefault("lh_import_profile", True)
    st.session_state.setdefault("lh_import_match", True)
    st.session_state.setdefault("lh_manual_mode", False)
    st.session_state.setdefault("lh_results", [])
    st.session_state.setdefault("lh_last_query", "")
    st.session_state.setdefault("lh_last_location", "")

    desired_inds = _parse_industry_list(st.session_state.get("consumer_desired_industries", ""))

    def _render_results(results: List[Dict[str, Any]]) -> None:
        if not results:
            st.info("No results yet. Run a search to populate Live Hunt.")
            return

        in_industry, out_industry = _bucket_live_jobs(results, desired_inds)

        st.markdown(
            "<div class='pl-card'><div class='pl-h2'>Live results</div>"
            "<div class='pl-muted'>Open a job link, then optionally send a listing into JD Pulse for deeper analysis.</div></div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="large")

        def _render_live_card(j: Dict[str, Any], key_prefix: str, idx: int) -> None:
            title = (j.get("title") or "").strip() or "Job"
            company = (j.get("company") or "").strip()
            location = (j.get("location") or "").strip()
            url = (j.get("url") or "").strip()
            snippet = (j.get("snippet") or "").strip()

            st.markdown("<div class='pl-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='pl-h2'>{_html_escape(title)}</div>", unsafe_allow_html=True)
            meta = " Ã¢â‚¬Â¢ ".join([x for x in [company, location] if x])
            if meta:
                st.markdown(f"<div class='pl-muted'>{_html_escape(meta)}</div>", unsafe_allow_html=True)

            if snippet:
                st.write(snippet)

            if url:
                st.link_button("Open job listing", url, use_container_width=True)

            c1, c2 = st.columns(2, gap="small")
            with c1:
                if st.button("Save job", use_container_width=True, key=f"{key_prefix}_save_{idx}"):
                    demo_save_artifact(
                        MODE_EXPLORER,
                        st.session_state.get("user_id", "demo"),
                        "live_job",
                        f"{title} Ã¢â‚¬â€ {company}".strip(" Ã¢â‚¬â€"),
                        {"title": title, "company": company, "location": location, "url": url, "snippet": snippet},
                    )
                    st.success("Saved. Your results will stay on screen.")

            with c2:
                if st.button("Send to JD Pulse", use_container_width=True, key=f"{key_prefix}_pulse_{idx}"):
                    st.session_state["jd_seed"] = snippet or title
                    st.session_state["jd_source_url"] = url
                    st.session_state["explorer_page"] = "jd_pulse"
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        with left:
            st.markdown("<div class='pl-card'><div class='pl-h2'>In-industry</div></div>", unsafe_allow_html=True)
            if in_industry:
                for i, j in enumerate(in_industry[:12]):
                    _render_live_card(j, "lh_in", i)
            else:
                st.info("No jobs clearly matched your industry list. Try widening keywords or removing location.")

        with right:
            st.markdown("<div class='pl-card'><div class='pl-h2'>Adjacent / out-of-industry</div>"
                        "<div class='pl-muted'>These can be useful pivots or stepping-stone roles.</div></div>",
                        unsafe_allow_html=True)
            if out_industry:
                for i, j in enumerate(out_industry[:12]):
                    _render_live_card(j, "lh_out", i)
            else:
                st.info("No adjacent roles were bucketed this time. This just means the results looked consistently in one lane.")

    # --- Gate: what to import
    if not st.session_state["lh_gate_done"]:
        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Before you start</div>", unsafe_allow_html=True)
        st.markdown("<div class='pl-muted'>Choose what to import, then continue.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.checkbox("Import profile", value=bool(st.session_state.get("lh_import_profile", True)), key="lh_import_profile")
        st.checkbox("Import match results", value=bool(st.session_state.get("lh_import_match", True)), key="lh_import_match")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
        if st.button("Continue", use_container_width=True, key="lh_continue"):
            st.session_state["lh_gate_done"] = True
            # One-shot autorun if importing anything, so the user sees results immediately.
            if bool(st.session_state.get("lh_import_profile", True)) or bool(st.session_state.get("lh_import_match", True)):
                st.session_state["lh_autorun"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return


    # --- Mode selector
    st.toggle(
        "Manual mode (enter everything yourself)",
        value=bool(st.session_state.get("lh_manual_mode", False)),
        key="lh_manual_mode_toggle",
    )

    # Lock manual mode in a stable session key (avoids unexpected resets).
    st.session_state["lh_manual_mode"] = bool(st.session_state.get("lh_manual_mode_toggle", False))

    # --- Build query defaults
    profile = get_profile()
    me_results = st.session_state.get("me_results") or []
    manual = bool(st.session_state.get("lh_manual_mode", False))

    if manual:
        st.markdown("<div class='pl-card'><div class='pl-h2'>Manual search parameters</div><div class='pl-muted'>More fields = better matches.</div></div>", unsafe_allow_html=True)

        ms1, ms2 = st.columns(2, gap="large")
        with ms1:
            manual_role = st.text_input("Target role / keywords", key="lh_m_role", placeholder="e.g., Instructional Designer, Nurse Practitioner")
            manual_skills = st.text_area("Skills (comma-separated)", key="lh_m_skills", height=80, placeholder="e.g., Epic, patient care, curriculum design, stakeholder management")
            manual_interests = st.text_area("Interests", key="lh_m_interests", height=60, placeholder="e.g., leadership, education, healthcare AI")
        with ms2:
            manual_experience = st.text_area("Experience summary", key="lh_m_exp", height=90, placeholder="1Ã¢â‚¬â€œ3 sentences helps a lot.")
            manual_education = st.text_area("Education / certs", key="lh_m_edu", height=60, placeholder="e.g., DNP, RN, PMP")
            manual_industries = st.text_input("Industries (comma-separated, optional)", key="lh_m_inds", placeholder="e.g., Healthcare, Education, Technology")

        manual_profile = {
            "skills": manual_skills,
            "interests": manual_interests,
            "experience": manual_experience,
            "education": manual_education,
            "desired_industries": _parse_industry_list(manual_industries),
        }
        profile_for_search = manual_profile
        q_default = (manual_role or "").strip() or build_live_query_from_profile(manual_profile)
    else:
        # Build an editable profile signal set for Live Hunt so users can tweak it
        # (Import Profile populates these fields once, then preserves user edits).
        use_profile = bool(st.session_state.get("lh_import_profile", True))
        profile_for_search = dict(profile or {})

        if use_profile:
            if not st.session_state.get("lh_pf_inited", False):
                st.session_state["lh_pf_role"] = str(
                    profile_for_search.get("preferred_title")
                    or profile_for_search.get("desired_title")
                    or profile_for_search.get("job_title")
                    or ""
                ).strip()
                st.session_state["lh_pf_skills"] = str(profile_for_search.get("skills") or "").strip()
                st.session_state["lh_pf_interests"] = str(profile_for_search.get("interests") or "").strip()
                st.session_state["lh_pf_experience"] = str(profile_for_search.get("experience") or "").strip()
                st.session_state["lh_pf_education"] = str(profile_for_search.get("education") or "").strip()

                # Desired industries may live on profile or session; prefer profile
                di = profile_for_search.get("desired_industries") or st.session_state.get("consumer_desired_industries") or ""
                if isinstance(di, list):
                    st.session_state["lh_pf_inds"] = ", ".join([str(x) for x in di if str(x).strip()])
                else:
                    st.session_state["lh_pf_inds"] = str(di or "").strip()

                st.session_state["lh_pf_inited"] = True

            with st.expander("Profile info used for Live Hunt (editable)", expanded=True):
                st.caption("These fields help Live Hunt build a better query. Edit them, or reset from your saved profile.")
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.text_input("Preferred job title / target role", key="lh_pf_role", placeholder="e.g., Training Manager, Product Ops, Education Engineer")
                    st.text_area("Skills (comma-separated)", key="lh_pf_skills", height=80, placeholder="e.g., LMS admin, onboarding, SQL, stakeholder management")
                    st.text_area("Interests (optional)", key="lh_pf_interests", height=60, placeholder="e.g., AI enablement, healthcare tech, customer education")
                with c2:
                    st.text_area("Experience (short summary)", key="lh_pf_experience", height=80, placeholder="e.g., 8+ years leading technical training programs...")
                    st.text_area("Education / certs", key="lh_pf_education", height=60, placeholder="e.g., BAS EE, PMP, Security+")
                    st.text_input("Preferred industries (comma-separated, optional)", key="lh_pf_inds", placeholder="e.g., Healthcare, Education, Technology")

                r1, r2 = st.columns([0.55, 0.45], gap="small")
                with r2:
                    if st.button("Reset from saved profile", use_container_width=True, key="lh_pf_reset"):
                        st.session_state["lh_pf_inited"] = False
                        st.rerun()

            # Rebuild profile_for_search from editable inputs
            profile_for_search = dict(profile_for_search)
            profile_for_search["preferred_title"] = str(st.session_state.get("lh_pf_role") or "").strip()
            profile_for_search["skills"] = str(st.session_state.get("lh_pf_skills") or "").strip()
            profile_for_search["interests"] = str(st.session_state.get("lh_pf_interests") or "").strip()
            profile_for_search["experience"] = str(st.session_state.get("lh_pf_experience") or "").strip()
            profile_for_search["education"] = str(st.session_state.get("lh_pf_education") or "").strip()
            profile_for_search["desired_industries"] = _parse_industry_list(st.session_state.get("lh_pf_inds") or "")

        q_default = build_live_query_from_profile(profile_for_search)

        # If Match & Explore results exist, append top industries/roles into query
        if me_results:
            inds = []
            roles = []
            for r in me_results[:8]:
                if isinstance(r, dict):
                    if r.get("industry"):
                        inds.append(str(r.get("industry")))
                    if r.get("role"):
                        roles.append(str(r.get("role")))
            inds = _parse_industry_list(inds)
            roles = [x.strip() for x in roles if x and x.strip()][:6]
            extra = " ".join(roles + inds)
            if extra:
                q_default = (q_default + " " + extra).strip()

    # --- Live Hunt field persistence rules
    # Keep user-entered values across tab switches/reruns.
    # - Keywords start BLANK (recommended: add keywords for better results).
    # - If Keywords are blank, we still run a profile-derived query under the hood (q_default).
    if "lh_q" not in st.session_state:
        st.session_state["lh_q"] = ""
    if "lh_loc" not in st.session_state:
        # Default location from saved profile ONLY on first load (user can clear it and it will stay cleared).
        st.session_state["lh_loc"] = (
            str(profile.get("location") or "").strip() if st.session_state.get("lh_import_profile") else ""
        )

    st.info("Tip: Adding keywords usually improves Live Hunt results. You can also search without keywords (using your imported profile signals).")

    # Allow user to restore fields from saved profile after clearing/editing them.
    if st.session_state.get("lh_import_profile"):
        if st.button("Reset fields from saved profile", key="lh_reset_fields_profile", use_container_width=True):
            st.session_state["lh_loc"] = str(profile.get("location") or "").strip()
            # Do NOT overwrite keywords; keep them user-controlled and blank by default.
            st.rerun()

    q = st.text_input("Keywords", value=(st.session_state.get("lh_last_query") or ""), key="lh_q")
    loc = st.text_input("Location (optional)", value=(st.session_state.get("lh_last_location") or loc_default), key="lh_loc")
    q_effective = (q or '').strip() or (q_default or '').strip()
    loc_effective = (loc or '').strip()


    go = st.button("Search jobs", use_container_width=True, key="lh_go")

    # Auto-run is triggered by import actions (profile / M&E). One-shot to avoid rerun loops.
    autorun = bool(st.session_state.pop("lh_autorun", False))

    # If no new search was requested, keep last results visible
    if (not go) and (not autorun) and st.session_state.get("lh_results"):
        _render_results(st.session_state.get("lh_results") or [])
        return

    # Run search
    if not (go or autorun):
        return

    if not q_effective:
        st.warning("Add keywords or import profile/match signals to run Live Hunt.")
        return

    with st.spinner("Searching live jobs..."):
        results: List[Dict[str, Any]] = []

        path = m.get("consumer_live_hunt", "")
        if path:
            res = api_post(path, {"q": q_effective, "location": loc_effective, "profile": profile_for_search}, timeout=60)
            if res.ok:
                results = res.data.get("results") or []
                _lh_set_debug("backend", q_effective, loc_effective)
            else:
                st.error(f"Live Hunt backend failed: {res.error}")
                return
        else:
            # Fallback: SERP-based helper (provided elsewhere in this file)
            results = serp_ai_search(q_effective, loc_effective)

    st.session_state["lh_results"] = results
    st.session_state["lh_last_query"] = q
    st.session_state["lh_last_location"] = loc
    st.session_state["lh_autorun"] = False

    if not results:
        # If the search provider failed (missing key / quota / HTTP error), show that instead of a generic message.
        lh_err = st.session_state.get("lh_last_error") or ""
        if lh_err:
            st.error(f"Live Hunt search did not return results because: {lh_err}")
            details = st.session_state.get("lh_last_details") or {}
            if details:
                with st.expander("Live Hunt debug details"):
                    st.write({
                        "provider": st.session_state.get("lh_last_provider"),
                        "q": st.session_state.get("lh_last_q"),
                        "location": st.session_state.get("lh_last_loc"),
                        **details,
                    })
        else:
            st.info("No results found. Try broader keywords, remove location, or add a job title.")
            with st.expander("Live Hunt debug"):
                st.write({
                    "provider": st.session_state.get("lh_last_provider"),
                    "q": st.session_state.get("lh_last_q") or q,
                    "location": st.session_state.get("lh_last_loc") or loc,
                })
        return

    _render_results(results)


def render_files_page(mode: str) -> None:
    card("Files", "Upload your own files + keep saved searches and outputs. No JSON.")
    user_id = st.session_state["user_id"]

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Upload a file</div>", unsafe_allow_html=True)
    up = st.file_uploader("Add any file (stored demo-locally)", type=None, key=f"file_up_{mode}")
    if up is not None:
        raw = read_upload_bytes(up)
        demo_save_upload(mode, user_id, up.name, up.type or "application/octet-stream", raw)
        st.success("Uploaded to Files.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    uploads = demo_list_uploads(mode, user_id)
    if uploads:
        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Your uploads</div>", unsafe_allow_html=True)
        for row in uploads[:30]:
            st.markdown(f"<div style='font-weight:900'>{row['filename']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='pl-muted'>{row['mimetype']} Ã¢â‚¬Â¢ {row['size_bytes']} bytes Ã¢â‚¬Â¢ {row['created_at']}</div>", unsafe_allow_html=True)

            c1, c2 = st.columns([0.6, 0.4], gap="small")
            with c1:
                fn, mt, blob = demo_get_upload_blob(int(row["id"]))
                st.download_button("Download", data=blob, file_name=fn, mime=mt, use_container_width=True, key=f"dl_{mode}_{row['id']}")
            with c2:
                if st.button("Delete", use_container_width=True, key=f"delup_{mode}_{row['id']}"):
                    demo_delete_upload(int(row["id"]))
                    st.rerun()

            st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    items = demo_list_artifacts(mode, user_id)
    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Saved outputs</div>", unsafe_allow_html=True)
    if not items:
        st.info("No saved outputs yet.")
    else:
        for row in items[:40]:
            st.markdown(f"<div style='font-weight:900'>{row['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='pl-muted'>{row['kind']} Ã¢â‚¬Â¢ {row['created_at']}</div>", unsafe_allow_html=True)
            if st.button("Delete", use_container_width=True, key=f"del_art_{mode}_{row['id']}"):
                demo_delete_artifact(int(row["id"]))
                st.rerun()
            st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Employer (Talent Studio): Company Profile + JD Studio + Comp Builder
# =============================================================================

def get_enterprise_profile() -> Dict[str, Any]:
    m = endpoint_map()
    gp = m.get("enterprise_profile_get", "")
    if gp:
        res = api_get(gp)
        if res.ok and res.data:
            return res.data.get("profile", res.data)

    # Local fallback from last saved
    default = {
        "company_name": "",
        "profile_picture_upload_id": "",
        "portfolio_text": "",
        "hq_location": "",
        "industry": "",
        "company_size": "",
        "about": "",
        "values": "",
        "successes": "",
        "benefits": "",
        "work_model": "",
        "eeo_statement": "",
        "compliance_notes": "",
        "brand_voice": "",
    }
    try:
        items = demo_list_artifacts(MODE_TALENT, st.session_state.get("user_id", "demo"), kind="company_profile_local")
        if items:
            payload = json.loads(items[0]["payload_json"])
            if isinstance(payload, dict):
                default.update({k: payload.get(k, default.get(k)) for k in default.keys()})
    except Exception:
        pass
    return default


def save_enterprise_profile(profile: Dict[str, Any]) -> Tuple[bool, str]:
    m = endpoint_map()
    sp = m.get("enterprise_profile_save", "")
    if sp:
        res = api_post(sp, {"profile": profile})
        if res.ok:
            return True, ""
        res2 = api_post(sp, profile)
        if res2.ok:
            return True, ""
        return False, res.error or res2.error or "Unknown backend error"

    demo_save_artifact(MODE_TALENT, st.session_state["user_id"], "company_profile_local", "Company profile (local)", profile)
    return True, ""


def render_talent_company_profile_page() -> None:
    profile = get_enterprise_profile()
    card("Company Profile", "Specialized fields used across JD Studio + Comp Builder.")
    updated = dict(profile)

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Brand assets (optional)</div>", unsafe_allow_html=True)

    pic = st.file_uploader("Company profile picture (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"], key="cp_pic_up")
    if pic is not None:
        raw = read_upload_bytes(pic)
        demo_save_upload(MODE_TALENT, st.session_state["user_id"], pic.name, pic.type or "application/octet-stream", raw)
        ups = demo_list_uploads(MODE_TALENT, st.session_state["user_id"])
        if ups:
            updated["profile_picture_upload_id"] = str(ups[0]["id"])
        try:
            st.image(raw, caption="Profile picture preview", use_container_width=True)
        except Exception:
            pass

    portfolio = st.file_uploader("Company portfolio / one-pager (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="cp_portfolio_up")
    if portfolio is not None:
        raw = read_upload_bytes(portfolio)
        demo_save_upload(MODE_TALENT, st.session_state["user_id"], portfolio.name, portfolio.type or "application/octet-stream", raw)
        extracted = try_extract_text_from_bytes(portfolio.name, portfolio.type or "", raw)
        if extracted.strip():
            updated["portfolio_text"] = extracted.strip()
            st.success("Portfolio text extracted. You can auto-fill fields below.")
        else:
            st.warning("Saved to Files, but could not extract text locally. Paste portfolio text below.")

    updated["portfolio_text"] = st.text_area(
        "Portfolio text (optional)",
        value=updated.get("portfolio_text") or "",
        height=120,
        placeholder="Paste company overview, case studies, products, awards, culture notes...",
    )

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    if st.button("Auto-fill profile from portfolio text", use_container_width=True, key="cp_autofill"):
        if not (updated.get("portfolio_text") or "").strip():
            st.warning("Add portfolio text first.")
        elif not openai_available():
            st.warning("Set OPENAI_API_KEY to enable auto-fill.")
        else:
            with st.spinner("Extracting company profile details..."):
                ok, text = openai_generate(
                    system="You extract structured company profile information from messy text. Return readable sections (not JSON).",
                    user=f"""From the portfolio text below, extract:

1) A crisp About section (3Ã¢â‚¬â€œ5 sentences)
2) Company values (5Ã¢â‚¬â€œ8 bullets)
3) Company successes (5Ã¢â‚¬â€œ8 bullets)
4) Benefits/perks (bulleted list)
5) Brand voice guidance for job descriptions (tone + style)

Return in labeled sections so the user can paste it into fields.

PORTFOLIO TEXT:
{updated.get("portfolio_text","")}
""",
                    max_tokens=900,
                )
            if ok and text.strip():
                st.session_state["cp_autofill_text"] = text
                parsed = _parse_company_profile_autofill(text)
                if parsed:
                    st.session_state["enterprise_profile_draft"] = parsed
                    st.success("Auto-filled the form fields below. Review and save when ready.")
                    st.rerun()
                st.success("Generated suggestions. Review and paste into fields below.")
            else:
                st.error(text)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("cp_autofill_text"):
        st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Auto-fill suggestions</div>", unsafe_allow_html=True)
        callout("Autofill draft", st.session_state["cp_autofill_text"], tone="teal")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    updated["company_name"] = st.text_input("Company name", value=updated.get("company_name") or "")
    updated["hq_location"] = st.text_input("HQ location", value=updated.get("hq_location") or "")
    updated["industry"] = st.text_input("Industry", value=updated.get("industry") or "")
    updated["company_size"] = st.text_input("Company size (e.g., 51Ã¢â‚¬â€œ200, 1000+)", value=updated.get("company_size") or "")
    updated["work_model"] = st.selectbox("Work model", ["", "Remote", "Hybrid", "On-site"], index=0, key="cp_work_model")
    updated["about"] = st.text_area("About the company", value=updated.get("about") or "", height=90)
    updated["values"] = st.text_area("Values / culture", value=updated.get("values") or "", height=80)
    updated["successes"] = st.text_area(
        "Successes / proof points",
        value=updated.get("successes") or "",
        height=80,
        placeholder="Awards, metrics, major launches, notable partners, impact statements",
    )
    updated["benefits"] = st.text_area("Benefits / perks", value=updated.get("benefits") or "", height=80)
    updated["brand_voice"] = st.text_area(
        "Brand voice for JDs (tone + style)",
        value=updated.get("brand_voice") or "",
        height=70,
        placeholder="e.g., direct, inclusive, outcome-focused, minimal jargon",
    )
    updated["eeo_statement"] = st.text_area("EEO statement (optional)", value=updated.get("eeo_statement") or "", height=70)
    updated["compliance_notes"] = st.text_area("Compliance notes (ADA, security, background checks, etc.)", value=updated.get("compliance_notes") or "", height=70)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    if st.button("Save company profile", use_container_width=True):
        with st.spinner("Saving..."):
            ok, err = save_enterprise_profile(updated)
        if ok:
            st.success("Saved.")
        else:
            st.error(err)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_jd_studio_page() -> None:
    m = endpoint_map()
    company = get_enterprise_profile()

    _about = (company.get("about") or "").strip()
    _values = (company.get("values") or "").strip()
    _successes = (company.get("successes") or "").strip()
    company_brand_default = "\n\n".join([p for p in [_about, _values, _successes] if p]).strip()


    card("JD Studio", "JD Inspector + Rewrite Studio merged into one system. No JSON.")
    render_backend_debug_panel()

    jd = st.text_area("Paste job description", height=280, key="jd_studio_text")
    role_target = st.text_input("Role target (optional)", placeholder="e.g., Senior Data Analyst", key="jd_studio_role")
    tone = st.selectbox("Tone", ["Modern & inclusive", "Direct & technical", "Warm & human", "Executive"], index=0)
    include_eeo = st.checkbox("Include EEO/ADA language in rewrites", value=True)

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    analyze = st.button("Analyze JD", use_container_width=True, key="jd_studio_analyze")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        path = m.get("enterprise_jd_studio_analyze", "")
        payload = {"jd_text": jd, "company_profile": company, "role_target": role_target, "tone": tone}
        report_text = ""
        if path:
            with st.spinner('Analyzing JD...'):
                res = api_post(path, payload, timeout=90)
            if not res.ok:
                st.error(res.error)
                return
            # backend can return either report sections or a text field
            rep = res.data.get("report") or res.data
            report_text = rep.get("analysis_text") or rep.get("text") or ""
            if not report_text:
                # lightweight rendering if backend returns sections
                sections = rep.get("sections") or []
                lines: List[str] = []
                for s in sections:
                    if isinstance(s, dict):
                        t = s.get("title") or "Section"
                        lines.append(f"{t}\n")
                        bullets_list = s.get("bullets") or []
                        for b in bullets_list[:10]:
                            lines.append(f"- {b}")
                        lines.append("")
                report_text = "\n".join(lines).strip()

        else:
            if not openai_available():
                st.error("No backend analyze route and OPENAI_API_KEY is missing.")
                return
            ok, text = openai_generate(
                system="You are a JD auditor. Return structured sections (not JSON) with headings and bullet points.",
                user=(
                    "Analyze this JD with maximum granularity:\n"
                    "1) Bias risk areas and why they are bias risks\n"
                    "2) Clarity issues and confusing language\n"
                    "3) Unnecessary requirements and degree inflation\n"
                    "4) Inclusive rewrite suggestions by section\n"
                    "5) Missing compensation/benefits transparency\n"
                    "6) Compliance notes\n\n"
                    f"Company profile (for context):\n"
                    f"Company name: {company.get('company_name','')}\n"
                    f"Industry: {company.get('industry','')}\n"
                    f"Values: {company.get('values','')}\n"
                    f"Brand voice: {company.get('brand_voice','')}\n\n"
                    f"JD:\n{jd}"
                ),
                max_tokens=1200,
            )
            if not ok:
                st.error(text)
                return
            report_text = text

        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Analysis</div>", unsafe_allow_html=True)
        callout("Analysis", report_text or "Analysis complete.", tone="violet")
        st.markdown("</div>", unsafe_allow_html=True)

        demo_save_artifact(MODE_TALENT, st.session_state["user_id"], "jd_studio_analysis", f"JD Studio analysis {_utc_now_iso()}", {"role_target": role_target})

    # Rewrite controls under dropdown (as requested)
    with st.expander("Rewrite options", expanded=False):
        action = st.selectbox(
            "Choose rewrite type",
            ["Full AI rewrite", "Partial rewrite (selected section)", "Manual rewrite suggestions"],
            index=0,
            key="jd_studio_action",
        )
        section = st.text_input("Section focus (for partial rewrite)", placeholder="e.g., Requirements, About the role, Responsibilities", key="jd_studio_section")

        st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
        go = st.button("Run selected option", use_container_width=True, key="jd_studio_do")
        st.markdown("</div>", unsafe_allow_html=True)

        if go:
            rewrite_path = m.get("enterprise_jd_studio_rewrite", "")
            payload = {
                "jd_text": jd,
                "company_profile": company,
                "role_target": role_target,
                "tone": tone,
                "include_eeo": include_eeo,
                "mode": action,
                "section_focus": section,
            }

            out_text = ""

            if rewrite_path:
                with st.spinner('Generating output...'):
                    res = api_post(rewrite_path, payload, timeout=100)
                if not res.ok:
                    st.error(res.error)
                    return
                out_text = res.data.get("text") or res.data.get("rewrite") or res.data.get("result_text") or ""
            else:
                if not openai_available():
                    st.error("No backend rewrite route and OPENAI_API_KEY is missing.")
                    return

                if action == "Full AI rewrite":
                    prompt = (
                        "Rewrite the entire JD to be inclusive, clear, and outcome-driven. "
                        "Give granular rationale notes first (what changed and why), then the rewritten JD. "
                        "Remove unnecessary requirements. "
                        f"{'Include EEO/ADA language at the end.' if include_eeo else 'Do not include EEO/ADA language.'}"
                    )
                elif action == "Partial rewrite (selected section)":
                    prompt = (
                        f"Rewrite ONLY the section '{section}' of this JD. "
                        "Provide (1) why it is risky/unclear, (2) revised section, (3) 5 bullet suggestions for manual edits elsewhere."
                    )
                else:
                    prompt = (
                        "Give manual rewrite suggestions: provide 10 specific line-level edits with before/after examples, "
                        "and explain the bias/clarity reason for each."
                    )

                ok, text = openai_generate(
                    system="You are an expert JD writer for enterprise hiring teams. Return readable sections, not JSON.",
                    user=(
                        f"{prompt}\n\n"
                        f"Company profile:\n"
                        f"Name: {company.get('company_name','')}\n"
                        f"Industry: {company.get('industry','')}\n"
                        f"Values: {company.get('values','')}\n"
                        f"Brand voice: {company.get('brand_voice','')}\n\n"
                        f"JD:\n{jd}"
                    ),
                    max_tokens=1400,
                )
                if not ok:
                    st.error(text)
                    return
                out_text = text

            if out_text.strip():
                st.markdown('<div class="pl-card">', unsafe_allow_html=True)
                st.markdown("<div class='pl-h2'>Output</div>", unsafe_allow_html=True)
                callout("Output", out_text, tone="violet")
                st.markdown("</div>", unsafe_allow_html=True)
                demo_save_artifact(MODE_TALENT, st.session_state["user_id"], "jd_studio_output", f"JD Studio output {_utc_now_iso()}", {"action": action})


def render_comp_builder_page() -> None:
    m = endpoint_map()
    company = get_enterprise_profile()

    card("Comp Builder", "Build role requirements + budget. Backend validates comp via parquet when available. No JSON.")
    render_backend_debug_panel()

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    role_title = st.text_input("Role title (draft)", placeholder="e.g., Senior Instructional Designer")
    level = st.selectbox("Level", ["Entry", "Mid", "Senior", "Lead", "Manager", "Director"], index=2)
    location = st.text_input("Location", placeholder="Remote / City, State")
    work_model = st.selectbox("Work model", ["Remote", "Hybrid", "On-site"], index=0)
    employment_type = st.selectbox("Employment type", ["Full-time", "Part-time", "Contract", "Temp-to-hire"], index=0)

    budget_min = st.number_input("Budget min", min_value=0, value=90000, step=5000)
    budget_max = st.number_input("Budget max", min_value=0, value=130000, step=5000)
    bonus = st.text_input("Bonus / variable comp (optional)", placeholder="e.g., 10% target bonus")
    equity = st.text_input("Equity (optional)", placeholder="e.g., RSUs, options, equity range")
    benefits_pkg = st.text_area("Benefits package (optional)", height=70, placeholder="Health, retirement match, PTO, learning stipend, etc.")
    include_benefits_in_jd = st.checkbox("Include salary/benefits transparency in JD draft", value=True)

    st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)

    must_have = st.text_area("Must-have requirements", height=90, placeholder="Hard requirements only. Keep it lean.")
    nice = st.text_area("Nice-to-have (optional)", height=70)

    responsibilities = st.text_area("Core responsibilities", height=90, placeholder="What success looks like in 90 days / 6 months helps a lot.")
    tools = st.text_area("Tools / stack", height=70, placeholder="e.g., Jira, Confluence, SQL, Tableau, Figma")
    years = st.text_input("Years of experience (optional)", placeholder="e.g., 5+")
    education = st.text_input("Education requirement (optional)", placeholder="e.g., BA preferred, equivalent experience accepted")
    certifications = st.text_input("Certifications (optional)", placeholder="e.g., PMP, Security+")
    reports_to = st.text_input("Reports to (optional)", placeholder="e.g., Director of Product Enablement")
    team_context = st.text_input("Team context (optional)", placeholder="e.g., join a team of 4 enablement specialists")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
    go = st.button("Generate comp + JD guidance", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if not go:
        return

    payload = {
        "company_profile": company,
        "role": {
            "role_title": role_title,
            "level": level,
            "location": location,
            "work_model": work_model,
            "employment_type": employment_type,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "bonus": bonus,
            "equity": equity,
            "benefits_package": benefits_pkg,
            "include_benefits_in_jd": include_benefits_in_jd,
            "must_have": must_have,
            "nice_to_have": nice,
            "responsibilities": responsibilities,
            "tools": tools,
            "years_experience": years,
            "education": education,
            "certifications": certifications,
            "reports_to": reports_to,
            "team_context": team_context,
        },
    }

    path = m.get("enterprise_comp_builder", "")
    out: Dict[str, Any] = {}

    if path:
        with st.spinner('Generating comp + JD guidance...'):
            res = api_post(path, payload, timeout=110)
        if not res.ok:
            st.error(res.error)
            return
        out = res.data
    else:
        if not openai_available():
            st.error("No backend comp builder route and OPENAI_API_KEY missing.")
            return
        ok, text = openai_generate(
            system="You are an enterprise compensation and JD builder. Return readable sections, not JSON.",
            user=(
                "Create:\n"
                "1) Recommended official title\n"
                "2) Salary guidance based on role scope (explain assumptions)\n"
                "3) Is budget realistic? If not, suggest alternative titles or reduced requirements\n"
                "4) A full JD draft\n\n"
                f"Company profile:\n"
                f"Name: {company.get('company_name','')}\n"
                f"Industry: {company.get('industry','')}\n"
                f"Values: {company.get('values','')}\n"
                f"Brand voice: {company.get('brand_voice','')}\n\n"
                f"Role inputs:\n{payload['role']}\n\nInclude salary/benefits in the JD if role inputs include include_benefits_in_jd=True."
            ),
            max_tokens=1500,
        )
        if not ok:
            st.error(text)
            return
        out = {"summary_text": text}

    st.markdown('<div class="pl-card">', unsafe_allow_html=True)
    st.markdown("<div class='pl-h2'>Result</div>", unsafe_allow_html=True)

    title = out.get("recommended_title") or out.get("title") or ""
    if title:
        st.markdown(f"<div style='font-weight:900; font-size:16px; color:{INK};'>Recommended title: {title}</div>", unsafe_allow_html=True)

    budget_eval = out.get("budget_evaluation") or out.get("budget_summary") or ""
    if budget_eval:
        st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Budget vs requirements</div>", unsafe_allow_html=True)
        callout("Budget evaluation", budget_eval, tone="amber")

    alternatives = out.get("alternatives") or []
    if alternatives:
        st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Alternative roles or requirement reductions</div>", unsafe_allow_html=True)
        for a in alternatives[:8]:
            name = a.get("role") or a.get("title") or ""
            why = a.get("why") or a.get("note") or ""
            if name:
                st.markdown(f"<div style='font-weight:900; color:{INK};'>{name}</div>", unsafe_allow_html=True)
            if why:
                st.markdown(f"<div class='pl-muted'>{why}</div>", unsafe_allow_html=True)

    jd_text = out.get("jd_text") or out.get("jd") or ""
    if jd_text:
        st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>JD draft</div>", unsafe_allow_html=True)
        callout("Job description", jd_text, tone="teal")

    if out.get("summary_text"):
        st.markdown("<div class='pl-divider'></div>", unsafe_allow_html=True)
        callout("Summary", out.get("summary_text") or "", tone="teal")

    st.markdown("</div>", unsafe_allow_html=True)

    demo_save_artifact(MODE_TALENT, st.session_state["user_id"], "comp_builder", f"Comp Builder {_utc_now_iso()}", {"recommended_title": title or ""})


# =============================================================================
# Top bars
# =============================================================================

def explorer_topbar() -> None:
    email = current_user_email()
    st.markdown('<div class="pl-topbar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.22, 0.58, 0.20], gap="small")

    with c1:
        st.markdown(f"<div class='pl-brand'>{APP_BRAND}</div><div class='pl-mode'>{MODE_EXPLORER}</div>", unsafe_allow_html=True)

    with c2:
        page = st.session_state.get("page", "explorer.match")
        st.text_input(
            "Search",
            key="pl_global_search",
            placeholder="Search (coming soon)Ã¢â‚¬Â¦",
            label_visibility="collapsed",
        )
        st.write("")

        tabs = [("Match & Explore", "explorer.match"), ("JD Pulse", "explorer.pulse"), ("Live Hunt", "explorer.live")]
        tcols = st.columns(len(tabs), gap="small")
        for i, (label, target) in enumerate(tabs):
            with tcols[i]:
                if st.button(label, use_container_width=True, key=f"ex_tab_{target}"):
                    nav_to(target)
                    st.rerun()

    with c3:
        with st.popover("Account", use_container_width=True):
            st.markdown(f"<div class='pl-muted'>{email}</div>", unsafe_allow_html=True)
            st.write("")
            if st.button("Profile", use_container_width=True):
                nav_to("explorer.account.profile")
                st.rerun()
            if st.button("Files", use_container_width=True):
                nav_to("explorer.files")
                st.rerun()
            if st.button("Help", use_container_width=True):
                nav_to("explorer.account.help")
                st.rerun()
            st.write("---")
            if st.button("Sign out", use_container_width=True):
                reset_auth()
                st.session_state["mode"] = ""
                st.session_state["page"] = ""
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def talent_topbar() -> None:
    email = current_user_email()
    st.markdown('<div class="pl-topbar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.22, 0.58, 0.20], gap="small")

    with c1:
        st.markdown(f"<div class='pl-brand'>{APP_BRAND}</div><div class='pl-mode'>{MODE_TALENT}</div>", unsafe_allow_html=True)

    with c2:
        page = st.session_state.get("page", "talent.jd_studio")
        tabs = [("JD Studio", "talent.jd_studio"), ("Comp Builder", "talent.comp")]
        tcols = st.columns(len(tabs), gap="small")
        for i, (label, target) in enumerate(tabs):
            with tcols[i]:
                if st.button(label, use_container_width=True, key=f"tal_tab_{target}"):
                    nav_to(target)
                    st.rerun()

    with c3:
        with st.popover("Account", use_container_width=True):
            st.markdown(f"<div class='pl-muted'>{email}</div>", unsafe_allow_html=True)
            st.write("")
            if st.button("Company Profile", use_container_width=True):
                nav_to("talent.account.profile")
                st.rerun()
            if st.button("Files", use_container_width=True):
                nav_to("talent.files")
                st.rerun()
            if st.button("Help", use_container_width=True):
                nav_to("talent.account.help")
                st.rerun()
            st.write("---")
            if st.button("Sign out", use_container_width=True):
                reset_auth()
                st.session_state["mode"] = ""
                st.session_state["page"] = ""
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Help
# =============================================================================

def render_help_page(mode: str) -> None:
    card("Help", "Quick guidance.")
    if mode == MODE_EXPLORER:
        bullets([
            "Match & Explore pulls from parquet-backed datasets via the backend. Use desired industries + context fields to shape results.",
            "Live Hunt uses SerpAPI by default, and can later be re-ranked by backend embeddings/FAISS when that endpoint exists.",
            "JD Pulse runs via backend if available, otherwise uses OpenAI fallback (if configured).",
            "Files stores your uploads and saved outputs for later reuse.",
        ])
    else:
        bullets([
            "Company Profile feeds JD Studio and Comp Builder.",
            "JD Studio merges inspector + rewrite into one place. Use Analyze, then pick a rewrite option.",
            "Comp Builder compares budget vs scope when the backend endpoint exists. Otherwise it uses AI fallback.",
            "Files stores uploads and outputs for later reuse.",
        ])


# =============================================================================
# Landing + Login + Shell routers
# =============================================================================

def render_main_two_buttons() -> None:
    st.markdown(
        f"""
<div class="pl-card">
  <div class="pl-title">{APP_BRAND}</div>
  <div class="pl-subtitle">Choose your environment.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
        if st.button("Explorer", use_container_width=True):
            st.session_state["mode"] = MODE_EXPLORER
            st.session_state["page"] = "explorer.match"
            st.session_state["lh_gate_done"] = False
            # Persist mode in the URL so external links (Wix) and refreshes stay in the same experience
            try:
                st.query_params["mode"] = "explorer"  # Streamlit >= 1.27
            except Exception:
                try:
                    st.experimental_set_query_params(mode="explorer")  # older Streamlit
                except Exception:
                    pass
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
        if st.button("Talent Studio", use_container_width=True):
            st.session_state["mode"] = MODE_TALENT
            st.session_state["page"] = "talent.jd_studio"
            # Persist mode in the URL so external links (Wix) and refreshes stay in the same experience
            try:
                st.query_params["mode"] = "talent"  # Streamlit >= 1.27
            except Exception:
                try:
                    st.experimental_set_query_params(mode="talent")  # older Streamlit
                except Exception:
                    pass
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_login(mode: str) -> None:
    st.markdown(
        f"""
<div class="pl-card">
  <div class="pl-title">{APP_BRAND} {mode}</div>
  <div class="pl-subtitle">{'Built for you, not for selling you.' if mode==MODE_EXPLORER else 'Built for hiring teams with serious control.'}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown(
            f"<div class='pl-h2'>{'Explorer access' if mode==MODE_EXPLORER else 'Talent Studio access'}</div>",
            unsafe_allow_html=True,
        )

        tabs = st.tabs(["Sign in", "Create account"])

        # --------------------------
        # Sign in
        # --------------------------
        with tabs[0]:
            email = st.text_input("Email", key=f"login_email_{mode}", placeholder="you@example.com")
            password = st.text_input("Password", key=f"login_pass_{mode}", type="password", placeholder="Ã¢â‚¬Â¢Ã¢â‚¬Â¢Ã¢â‚¬Â¢Ã¢â‚¬Â¢Ã¢â‚¬Â¢Ã¢â‚¬Â¢Ã¢â‚¬Â¢Ã¢â‚¬Â¢")

            st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
            if st.button("Sign in", use_container_width=True, key=f"btn_login_{mode}"):
                m = endpoint_map()
                login_path = m.get("auth_login", "") or "/auth/login"
                with st.spinner("Signing in..."):
                    res = api_post(login_path, {"email": email, "password": password})
                if res.ok:
                    token = res.data.get("access_token", "") or res.data.get("token", "")
                    st.session_state["auth_token"] = token

                    # Try to fetch a canonical user object (preferred), otherwise fall back to what we have.
                    me = api_get("/auth/me")
                    if me.ok and me.data:
                        st.session_state["user"] = me.data.get("user", {}) or {}
                        uid = str((st.session_state["user"] or {}).get("user_id") or (st.session_state["user"] or {}).get("id") or "")
                        if uid:
                            st.session_state["user_id"] = uid
                    else:
                        st.session_state["user"] = res.data.get("user", {}) or {"email": email}
                        st.session_state["user_id"] = str(res.data.get("user_id") or email or "demo")

                    st.session_state["authed_mode"] = mode
                    st.success("Signed in.")
                    st.rerun()
                else:
                    st.error(f"Login failed: {res.error}")
            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------
        # Register
        # --------------------------
        with tabs[1]:
            reg_email = st.text_input("Email", key=f"reg_email_{mode}", placeholder="you@example.com")
            reg_password = st.text_input("Password", key=f"reg_pass_{mode}", type="password", placeholder="Create a password")
            reg_password2 = st.text_input("Confirm password", key=f"reg_pass2_{mode}", type="password", placeholder="Repeat password")

            st.markdown('<div class="pl-primary">', unsafe_allow_html=True)
            if st.button("Create account", use_container_width=True, key=f"btn_register_{mode}"):
                if not reg_email.strip() or not reg_password:
                    st.warning("Enter an email and password.")
                elif reg_password != reg_password2:
                    st.warning("Passwords do not match.")
                else:
                    with st.spinner("Creating account..."):
                        rreg = api_post("/auth/register", {"email": reg_email, "password": reg_password})
                    if not rreg.ok:
                        st.error(f"Register failed: {rreg.error}")
                    else:
                        with st.spinner("Signing you in..."):
                            rlog = api_post("/auth/login", {"email": reg_email, "password": reg_password})
                        if not rlog.ok:
                            st.error(f"Login failed: {rlog.error}")
                        else:
                            token = rlog.data.get("access_token", "") or rlog.data.get("token", "")
                            st.session_state["auth_token"] = token
                            me = api_get("/auth/me")
                            if me.ok and me.data:
                                st.session_state["user"] = me.data.get("user", {}) or {"email": reg_email}
                                st.session_state["user_id"] = str((st.session_state["user"] or {}).get("user_id") or (st.session_state["user"] or {}).get("id") or rlog.data.get("user_id") or reg_email or "demo")
                            else:
                                st.session_state["user"] = {"email": reg_email}
                                st.session_state["user_id"] = str(rlog.data.get("user_id") or reg_email or "demo")

                            st.session_state["authed_mode"] = mode
                            st.success("Account created and signed in.")
                            st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Back", use_container_width=True):
            reset_auth()
            st.session_state["mode"] = ""
            st.session_state["page"] = ""
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="pl-card">', unsafe_allow_html=True)
        st.markdown("<div class='pl-h2'>Keys & data</div>", unsafe_allow_html=True)
        st.write("Backend loads parquets + embeddings/FAISS at startup (authoritative).")
        st.write("")
        st.markdown(f"- Backend: `{API_BASE}`")
        st.markdown(f"- OpenAI: `{'configured' if openai_available() else 'missing'}`")
        st.markdown(f"- SerpAPI: `{'configured' if serpapi_available() else 'missing'}`")
        st.markdown("</div>", unsafe_allow_html=True)

        render_backend_debug_panel()

def render_explorer_shell() -> None:
    explorer_topbar()
    st.write("")
    page = st.session_state.get("page", "explorer.match")

    if page == "explorer.match":
        render_match_explore_page()
    elif page == "explorer.pulse":
        render_jd_pulse_page()
    elif page == "explorer.live":
        render_live_hunt_page()
    elif page == "explorer.files":
        render_files_page(MODE_EXPLORER)
    elif page == "explorer.account.profile":
        render_explorer_profile_page()
    elif page == "explorer.account.help":
        render_help_page(MODE_EXPLORER)
    else:
        render_match_explore_page()


def render_talent_shell() -> None:
    talent_topbar()
    st.write("")
    page = st.session_state.get("page", "talent.jd_studio")

    if page == "talent.jd_studio":
        render_jd_studio_page()
    elif page == "talent.comp":
        render_comp_builder_page()
    elif page == "talent.files":
        render_files_page(MODE_TALENT)
    elif page == "talent.account.profile":
        render_talent_company_profile_page()
    elif page == "talent.account.help":
        render_help_page(MODE_TALENT)
    else:
        render_jd_studio_page()


def _read_query_param(name: str) -> str | None:
    """Best-effort query param read across Streamlit versions."""
    try:
        v = st.query_params.get(name)
        # st.query_params can return list-like or scalar depending on version
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            vals = qp.get(name)
            return vals[0] if vals else None
        except Exception:
            return None


def main() -> None:
    st.set_page_config(
        page_title="Pathlight",
        page_icon="Ã¢Å“Â¨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    # ---- session state defaults (do not overwrite existing values)
    if "mode" not in st.session_state:
        st.session_state["mode"] = ""
    if "page" not in st.session_state:
        st.session_state["page"] = ""

    # ---- Deep-link support (Wix): ?mode=explorer | ?mode=talent
    mode_param = (
        _read_query_param("mode")
        or _read_query_param("m")
        or ""
    ).strip().lower()
    mode_token = re.sub(r"[^a-z]", "", mode_param or "")

    # ---- Fallback: if Wix strips query params, infer mode from URL path (/explorer or /talent)
    try:
        hdrs = st.context.headers  # Streamlit >= 1.27-ish
        bits = [
            hdrs.get("X-Original-URI") or "",
            hdrs.get("X-Forwarded-Uri") or "",
            hdrs.get("X-Forwarded-URI") or "",
            hdrs.get("Referer") or "",
            hdrs.get("Origin") or "",

        ]
        path = (" ".join(bits)).lower()

        if not mode_token:
            if "/explorer" in path or "/consumer" in path:
                mode_token = "explorer"
            elif "/talent" in path or "/enterprise" in path or "/studio" in path:
                mode_token = "talent"
    except Exception:
        pass

# DEBUG DISABLED:     st.write("DEBUG normalized mode_param =", mode_param)

    if mode_token in ("explorer", "consumer"):
        st.session_state["mode"] = MODE_EXPLORER
    elif mode_token in ("talent", "talentstudio", "talent_studio", "enterprise", "studio"):
        st.session_state["mode"] = MODE_TALENT

# DEBUG DISABLED:     st.write("DEBUG normalized mode =", st.session_state.get("mode"))
# DEBUG DISABLED:     st.write("DEBUG session mode =", st.session_state.get("mode"))
# DEBUG DISABLED:     st.write("DEBUG before computing final mode =", st.session_state.get("mode"))

    mode = st.session_state.get("mode") or (
        MODE_EXPLORER
        if mode_token in ("explorer", "consumer")
        else MODE_TALENT
        if mode_token in ("talent", "talentstudio", "enterprise", "studio")
        else ""
    )

    # ---- No mode yet abc123 landing chooser
    if not mode:
        render_main_two_buttons()
        return

    # ---- Auth gate
    if not ensure_authed_for_mode(mode):
        render_login(mode)
        return

    # ---- Enter app shell
    if mode == MODE_EXPLORER:
        render_explorer_shell()
    else:
        render_talent_shell()

if __name__ == "__main__":
    main()