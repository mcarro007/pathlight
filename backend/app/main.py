from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------
# Optional FAISS
# -------------------------------------------------
try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None  # type: ignore
    _faiss_import_error = repr(e)

# -------------------------------------------------
# Internal services
# -------------------------------------------------
from backend.app.api.data_routes import router as data_router
from backend.app.services.sqlite_db import init_db
from backend.app.services.auth_service import (
    register_user,
    login_user,
    logout_token,
    resolve_token,
)
from backend.app.services.profile_service import (
    get_profile,
    upsert_profile,
    parse_resume,
)
from backend.app.services.jd_eval import evaluate_job_description
from backend.app.services.live_jobs import fetch_live_jobs
from backend.app.services.corporate_audit import corporate_jd_audit
from backend.app.services.corporate_jd import generate_or_rewrite
from backend.app.services.openai_status import openai_status


# -------------------------------------------------
# JSON safety (kept for future use)
# -------------------------------------------------
def _json_safe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    try:
        return str(x)
    except Exception:
        return None


# -------------------------------------------------
# Auth helpers
# -------------------------------------------------
def _bearer_token(authorization: Optional[str]) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return ""


def require_user_id(authorization: Optional[str]) -> int:
    """
    Returns the user_id for a valid Bearer token.
    Raises a proper FastAPI HTTPException(401) if missing/invalid.
    """
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    uid = resolve_token(token)
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return int(uid)

def get_current_user_id(authorization: Optional[str] = Header(default=None)) -> int:
    return require_user_id(authorization)

# -------------------------------------------------
# Request models
# -------------------------------------------------
class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfilePatch(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    salary_min: Optional[float] = None
    skills: Optional[str] = None
    interests: Optional[str] = None
    experience: Optional[str] = None
    education: Optional[str] = None
    resume_text: Optional[str] = None


class ResumeParseRequest(BaseModel):
    resume_text: str
    mode: str = "auto"


class ConsumerSearchRequest(BaseModel):
    skills: str = ""
    interests: str = ""
    experience: str = ""
    education: str = ""
    location: str = ""
    salary_min: Optional[float] = None
    k: int = Field(10, ge=1, le=50)


class ConsumerEvaluateJDRequest(BaseModel):
    jd_text: str


class LiveJobsRequest(BaseModel):
    query: str
    location: str = ""
    salary_min: Optional[float] = None
    limit: int = 20


class CorporateAnalyzeJDRequest(BaseModel):
    jd_text: str


class EnterpriseJDRequest(BaseModel):
    existing_jd: str = ""
    requirements: Dict[str, Any] = {}
    use_openai: bool = True

class IndustrySuggestRequest(BaseModel):
    query: str = Field("", description="Free-text query for industry suggestions")


class IndustrySuggestResponse(BaseModel):
    ok: bool = True
    suggestions: List[str] = []


class JDPulseRequest(BaseModel):
    jd_text: str = Field("", description="Job description text to analyze")
    title: Optional[str] = None
    company: Optional[str] = None


class JDPulseResponse(BaseModel):
    ok: bool = True
    pulse: Dict[str, Any] = {}


class CompBuilderRequest(BaseModel):
    title: str = Field(..., description="Role title")
    location: Optional[str] = None
    level: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None


class CompBuilderResponse(BaseModel):
    ok: bool = True
    comp: Dict[str, Any] = {}

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Job Analyzer API", version="0.7.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Startup
# -------------------------------------------------
@app.on_event("startup")
def startup() -> None:
    init_db()


# -------------------------------------------------
# Health & system
# -------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/system/openai-status")
def system_openai():
    return openai_status()

@app.get("/system/build")
def system_build():
    return {
        "ok": True,
        "version": "0.7.2",
        "commit_hint": "routes-live-check"
    }

# -------------------------------------------------
# Register data routes
# -------------------------------------------------
app.include_router(data_router, prefix="/data", tags=["data"])


# -------------------------------------------------
# Auth endpoints
# -------------------------------------------------
@app.post("/auth/register")
def auth_register(req: RegisterRequest):
    uid = register_user(req.email, req.password)
    return {"ok": True, "user_id": uid}


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    token, uid = login_user(req.email, req.password)
    return {"ok": True, "token": token, "user_id": uid}


@app.post("/auth/logout")
def auth_logout(authorization: Optional[str] = Header(default=None)):
    token = _bearer_token(authorization)
    if token:
        logout_token(token)
    return {"ok": True}

@app.get("/auth/me")
def auth_me(user_id: int = Depends(get_current_user_id)):
    from backend.app.services.sqlite_db import connect

    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()

    return {
        "ok": True,
        "user": {
            "user_id": user_id,
            "email": row["email"] if row else None,
        },
    }


# -------------------------------------------------
# Profile endpoints
# -------------------------------------------------
@app.get("/profile")
def profile_get(authorization: Optional[str] = Header(default=None)):
    uid = require_user_id(authorization)
    return {"ok": True, "profile": get_profile(uid)}


@app.post("/profile")
def profile_post(req: ProfilePatch, authorization: Optional[str] = Header(default=None)):
    uid = require_user_id(authorization)
    return {"ok": True, "profile": upsert_profile(uid, req.model_dump(exclude_none=True))}


@app.post("/consumer/profile/parse-resume")
def profile_parse(req: ResumeParseRequest, authorization: Optional[str] = Header(default=None)):
    require_user_id(authorization)
    return parse_resume(req.resume_text, mode=req.mode)


# -------------------------------------------------
# Consumer endpoints
# -------------------------------------------------
@app.post("/consumer/search")
def consumer_search(req: ConsumerSearchRequest):
    # Placeholder until full FAISS/OpenAI hybrid search is wired
    return {"ok": True, "results": []}


@app.post("/consumer/evaluate-jd")
def consumer_eval(req: ConsumerEvaluateJDRequest):
    return evaluate_job_description(req.jd_text)

@app.post("/consumer/industry-suggest", response_model=IndustrySuggestResponse)
def consumer_industry_suggest(req: IndustrySuggestRequest):
    q = (req.query or "").strip().lower()
    if not q:
        return {"ok": True, "suggestions": []}

    seeds = [
        "Healthcare",
        "Fintech",
        "GovTech",
        "EdTech",
        "Cybersecurity",
        "AI/ML",
        "Biotech",
        "Retail",
        "Climate Tech",
        "Manufacturing",
    ]
    suggestions = [s for s in seeds if q in s.lower()] or seeds[:6]
    return {"ok": True, "suggestions": suggestions}


@app.post("/consumer/jd-pulse", response_model=JDPulseResponse)
def consumer_jd_pulse(req: JDPulseRequest):
    text = (req.jd_text or "").strip()
    if not text:
        return {"ok": True, "pulse": {"error": "empty jd_text"}}

    pulse = {
        "word_count": len(text.split()),
        "has_salary": (
            "$" in text
            or "salary" in text.lower()
            or "compensation" in text.lower()
        ),
        "notes": "Placeholder JD Pulse endpoint is live. Replace with scoring/LLM later.",
    }
    return {"ok": True, "pulse": pulse}


@app.post("/consumer/live-jobs")
def consumer_live(req: LiveJobsRequest):
    return fetch_live_jobs(
        req.query,
        location=req.location,
        salary_min=req.salary_min,
        limit=req.limit,
    )


# -------------------------------------------------
# Enterprise endpoints
# -------------------------------------------------
@app.post("/corporate/analyze-jd")
def corporate_analyze(req: CorporateAnalyzeJDRequest):
    return corporate_jd_audit(req.jd_text)


@app.post("/enterprise/rewrite-jd")
def enterprise_rewrite(req: EnterpriseJDRequest):
    return generate_or_rewrite(req.requirements, req.existing_jd, req.use_openai, mode="rewrite")


@app.post("/enterprise/generate-jd")
def enterprise_generate(req: EnterpriseJDRequest):
    return generate_or_rewrite(req.requirements, "", req.use_openai, mode="generate")

@app.post("/enterprise/comp-builder", response_model=CompBuilderResponse)
def enterprise_comp_builder(req: CompBuilderRequest):
    comp = {
        "title": req.title,
        "location": req.location,
        "level": req.level,
        "budget_min": req.budget_min,
        "budget_max": req.budget_max,
        "notes": "Placeholder Comp Builder endpoint is live. Replace with market data + AI later.",
    }
    return {"ok": True, "comp": comp}
