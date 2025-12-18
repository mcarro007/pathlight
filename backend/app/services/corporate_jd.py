from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = _strip_code_fences(text)

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _eeo_block() -> str:
    return (
        "Equal Employment Opportunity (EEO): We are an equal opportunity employer and value a diverse workforce. "
        "All qualified applicants will receive consideration without regard to race, color, religion, sex, national origin, age, disability, veteran status, or any other protected status."
    )


def _ada_block() -> str:
    return (
        "ADA Accommodation: If you require reasonable accommodation to complete any part of the application process, please let us know. "
        "We will work with you to provide reasonable accommodations."
    )


def template_variants(requirements: Dict[str, Any], existing_jd: str = "", mode: str = "generate") -> Dict[str, Any]:
    title = requirements.get("target_title") or f"{requirements.get('role_level','Role')} {requirements.get('job_family','Team')}".strip()
    include_eeo = bool(requirements.get("include_eeo", True))
    include_ada = bool(requirements.get("include_ada", True))

    overview = requirements.get("overview") or "This role supports measurable outcomes for the business and customers."
    must_have = requirements.get("must_have") or "Relevant experience and core skills for the role."
    nice = requirements.get("nice_to_have") or "Bonus skills that can be learned on the job."
    salary = requirements.get("salary_range") or "(suggested by system)"

    compliance_lines = []
    if include_eeo:
        compliance_lines.append(_eeo_block())
    if include_ada:
        compliance_lines.append(_ada_block())

    base = f"""
{title}

About the Role
- {overview}

What You'll Do
- Deliver outcomes tied to KPIs and stakeholder needs
- Partner cross-functionally to plan, execute, and iterate
- Communicate clearly through documentation and status updates

Required Qualifications
- {must_have}

Preferred Qualifications
- {nice}

Compensation
- Target range: {salary}

Benefits
- Competitive benefits package, flexible time off, and learning/development support

Compliance
- {' '.join(compliance_lines) if compliance_lines else 'Add EEO/ADA language as required.'}

What Success Looks Like in 90 Days
- Deliver 1-2 measurable wins aligned to KPIs
- Establish stakeholder alignment and operating cadence
- Document workflows, systems, and decision logs
""".strip()

    return {
        "ok": True,
        "openai_used": False,
        "mode": mode,
        "title": title,
        "salary_guidance": salary,
        "variants": [
            {"label": "Balanced", "job_description": base},
            {"label": "Concise", "job_description": base.replace("What You'll Do", "Key Responsibilities").replace("Required Qualifications", "Requirements")},
            {"label": "Detailed", "job_description": base + "\n\nAdditional Details\n- Add team size, tooling, reporting lines, and leveling guidance.\n"},
        ],
    }


def ai_variants(requirements: Dict[str, Any], existing_jd: str = "", mode: str = "generate") -> Dict[str, Any]:
    if not _has_openai_key():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment to use OpenAI for Enterprise JD creation.")

    from openai import OpenAI  # type: ignore
    client = OpenAI()

    task = "Generate a new JD" if mode == "generate" else "Rewrite the existing JD"

    prompt = f"""
You are an expert HR + hiring enablement writer.

Task: {task}. Produce 3 variants: Balanced, Concise, Detailed.

Return STRICT JSON ONLY (no markdown, no code fences) with fields:
- title (string)
- salary_guidance (string)
- variants (array) of objects {{ label (string), job_description (string) }}

Requirements:
{requirements}

Existing JD (if any):
{existing_jd or ""}

Constraints:
- Use neutral, inclusive language
- Include EEO language if include_eeo is true
- Include ADA accommodation language if include_ada is true
- Include clear headings and bullet points
- Include a 'What Success Looks Like in 90 Days' section
"""

    model = requirements.get("model", "gpt-4.1-mini")
    resp = client.responses.create(model=model, input=prompt)
    raw = resp.output_text or ""

    parsed = _extract_json_object(raw)
    if not parsed or "variants" not in parsed:
        raise RuntimeError("OpenAI response was not valid strict JSON with a 'variants' field. Try again or switch models.")

    return {"ok": True, "openai_used": True, "mode": mode, **parsed}


def generate_or_rewrite(requirements: Dict[str, Any], existing_jd: str = "", use_openai: bool = True, mode: str = "generate") -> Dict[str, Any]:
    if use_openai:
        return ai_variants(requirements, existing_jd=existing_jd, mode=mode)
    return template_variants(requirements, existing_jd=existing_jd, mode=mode)
