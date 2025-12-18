from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class RewriteRequest(BaseModel):
    jd_text: str

class RoleRequest(BaseModel):
    skills: List[str]
    level: str
    location: str | None = None

@router.post("/rewrite-jd")
def rewrite_jd(req: RewriteRequest):
    return {
        "title": "Senior Data Analyst",
        "salary_band": "$110k–$140k",
        "jd": "Rewritten, bias-reduced job description goes here."
    }

@router.post("/build-role")
def build_role(req: RoleRequest):
    return {
        "title": "Data Platform Analyst",
        "salary_band": "$95k–$125k",
        "responsibilities": ["Build dashboards", "Partner with stakeholders", "Ensure data quality"],
        "requirements": ["Python", "SQL", "3–5 years experience"]
    }
