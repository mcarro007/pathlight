from __future__ import annotations

import os
from typing import Any, Dict

def openai_status() -> Dict[str, Any]:
    key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return {
        "ok": True,
        "openai_api_key_present": key_present,
        "note": "Key presence indicates OpenAI can be called. Consumer search uses embeddings; Corporate AI features require Use OpenAI = true."
    }
