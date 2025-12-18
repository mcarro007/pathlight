@'
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, Optional

from app.db import upsert_profile, get_profile, add_shortlist, list_shortlist, add_screen

router = APIRouter()

class ProfileIn(BaseModel):
    skills: str = ""
    interests: str = ""
    experience: str = "Any"
    location: str = ""
    salary_need: str = ""

@router.post("/profile/save")
def profile_save(p: ProfileIn):
    upsert_profile(p.model_dump())
    return {"ok": True}

@router.post("/profile/get")
def profile_get():
    p = get_profile()
    return p or {}

class ShortlistIn(BaseModel):
    job: Dict[str, Any]

@router.post("/shortlist/add")
def shortlist_add(x: ShortlistIn):
    rid = add_shortlist(x.job)
    return {"ok": True, "id": rid}

@router.post("/shortlist/list")
def shortlist_list(limit: int = 100):
    return {"rows": list_shortlist(limit=limit)}

class ScreenIn(BaseModel):
    jd_text: str
    result: Dict[str, Any]

@router.post("/screens/save")
def screens_save(x: ScreenIn):
    rid = add_screen(x.jd_text, x.result)
    return {"ok": True, "id": rid}
'@ | Set-Content -Encoding utf8 .\backend\app\api\consumer_persist.py
