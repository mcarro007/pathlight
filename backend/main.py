from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware

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
