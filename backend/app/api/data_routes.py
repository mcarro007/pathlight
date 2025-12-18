from __future__ import annotations

from fastapi import APIRouter
from backend.app.services import job_data

router = APIRouter()

@router.get("/status")
def data_status() -> dict:
    return {"ok": True, "data": job_data.status_dict()}

@router.get("/preview")
def data_preview() -> dict:
    return {"ok": True, "rows": job_data.get_master_preview(5)}
