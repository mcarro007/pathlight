from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ConsumerSearchRequest(BaseModel):
    query: str
    k: int = 10
    location_filter: str | None = None

class ConsumerJDRequest(BaseModel):
    jd_text: str

@router.post("/search")
def search_jobs(req: ConsumerSearchRequest):
    return {"rows": [], "message": "Consumer search endpoint is live"}

@router.post("/evaluate-jd")
def evaluate_jd(req: ConsumerJDRequest):
    return {
        "scores": {"bias_risk": 35, "requirements_intensity": 62, "clarity": 70},
        "flags": ["10+ years experience for mid-level role", "Aggressive culture language"],
        "guidance": ["Consider applying if you meet core skills", "Expect heavy screening"],
        "summary": "JD appears inflated relative to responsibilities."
    }
