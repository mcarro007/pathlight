@'
import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, Optional

from app.db import save_role, add_version

router = APIRouter()

class RoleSaveIn(BaseModel):
    intake: Dict[str, Any]
    current: Optional[Dict[str, Any]] = None

@router.post("/roles/save")
def roles_save(x: RoleSaveIn):
    role_id = str(uuid.uuid4())
    save_role(role_id, x.intake, x.current)
    return {"ok": True, "role_id": role_id}

class VersionAddIn(BaseModel):
    role_id: str
    version: Dict[str, Any]

@router.post("/roles/version/add")
def roles_version_add(x: VersionAddIn):
    rid = add_version(x.role_id, x.version)
    return {"ok": True, "id": rid}
'@ | Set-Content -Encoding utf8 .\backend\app\api\corporate_persist.py
