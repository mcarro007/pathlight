from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.db import init_db
from app.api.consumer_persist import router as consumer_persist_router
from app.api.corporate_persist import router as corporate_persist_router

app = FastAPI(title="Job Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
def _startup():
    init_db()

# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}

# -------------------------------------------------------------------
# Auth helper (PATCH 1)
# -------------------------------------------------------------------

def require_user_id(authorization: str | None) -> int:
    """
    Extract user id from Authorization header.
    PATCH 1: Raise HTTPException instead of RuntimeError
    so unauthenticated access returns 401 instead of 500.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    # Placeholder logic — adjust if you later decode JWT/session
    try:
        return int(authorization)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization token",
        )

# -------------------------------------------------------------------
# Profile (PATCH 1)
# -------------------------------------------------------------------

@app.get("/profile")
def profile_get(authorization: str | None = Header(default=None)):
    """
    PATCH 1:
    Unauthenticated requests now return 401 cleanly
    instead of crashing the app.
    """
    uid = require_user_id(authorization)
    return {
        "user_id": uid,
        "profile": {},
    }

# -------------------------------------------------------------------
# Missing endpoints for Streamlit wiring
# -------------------------------------------------------------------

class IndustrySuggestRequest(BaseModel):
    query: str = Field(..., description="Free-text query for industry suggestions")

class IndustrySuggestResponse(BaseModel):
    ok: bool = True
    suggestions: List[str] = []

class JDPulseRequest(BaseModel):
    jd_text: str = Field(..., description="Job description text to analyze")
    title: Optional[str] = None
    company: Optional[str] = None

class JDPulseResponse(BaseModel):
    ok: bool = True
    pulse: Dict[str, Any] = {}

class CompBuilderRequest(BaseModel):
    title: str
    location: Optional[str] = None
    level: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None

class CompBuilderResponse(BaseModel):
    ok: bool = True
    comp: Dict[str, Any] = {}


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
        "has_salary": ("$" in text) or ("salary" in text.lower()) or ("compensation" in text.lower()),
        "notes": "Placeholder JD Pulse endpoint is live. Replace with AI scoring later.",
    }
    return {"ok": True, "pulse": pulse}


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

# -------------------------------------------------------------------
# Routers
# -------------------------------------------------------------------

app.include_router(
    consumer_persist_router,
    prefix="/consumer",
    tags=["consumer-persist"],
)

app.include_router(
    corporate_persist_router,
    prefix="/corporate",
    tags=["corporate-persist"],
)
