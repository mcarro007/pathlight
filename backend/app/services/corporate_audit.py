from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.app.services.jd_eval import detect_bias_findings, detect_unrealistic, basic_clarity_flags

EEO_HINTS = [
    "equal opportunity", "eeo", "equal employment", "all qualified applicants",
    "without regard to", "protected status"
]

ADA_HINTS = [
    "reasonable accommodation", "accommodation", "ada", "disability", "accessible"
]

STRUCTURE_HINTS = [
    "responsibilities", "requirements", "qualifications", "about the role",
    "what you'll do", "what you will do", "preferred", "benefits", "compensation"
]

def _has_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)

def _salary_present(text: str) -> bool:
    t = text or ""
    if re.search(r"\$\s*\d", t):
        return True
    tl = t.lower()
    return ("salary" in tl) or ("compensation" in tl) or ("pay range" in tl)

def _severity_counts(bias_findings: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for f in bias_findings:
        sev = f.get("severity", "Low")
        if sev in counts:
            counts[sev] += 1
    return counts

def corporate_jd_audit(jd_text: str, require_eeo: bool = True, require_ada: bool = True) -> Dict[str, Any]:
    jd = jd_text or ""

    bias_findings = detect_bias_findings(jd)
    unrealistic = detect_unrealistic(jd)
    clarity = basic_clarity_flags(jd)

    has_eeo = _has_any(jd, EEO_HINTS)
    has_ada = _has_any(jd, ADA_HINTS)
    has_structure = _has_any(jd, STRUCTURE_HINTS)
    has_salary = _salary_present(jd)

    compliance_gaps = []
    if require_eeo and not has_eeo:
        compliance_gaps.append({
            "gap": "EEO language not detected.",
            "why_it_matters": "EEO language is commonly required for enterprise hiring standards and reduces compliance risk.",
            "suggestion": "Add a standard EEO statement near the end of the JD."
        })
    if require_ada and not has_ada:
        compliance_gaps.append({
            "gap": "ADA accommodation language not detected.",
            "why_it_matters": "Accommodation language supports inclusive hiring and reduces risk.",
            "suggestion": "Add a reasonable accommodation statement near the end of the JD."
        })

    quality_flags = []
    if not has_structure:
        quality_flags.append({
            "issue": "JD structure headings not detected.",
            "why_it_matters": "Candidates scan quickly; missing structure reduces comprehension and conversion.",
            "suggestion": "Add headings: About the Role, Responsibilities, Requirements, Preferred, Compensation, Benefits."
        })
    if not has_salary:
        quality_flags.append({
            "issue": "Salary/compensation not detected.",
            "why_it_matters": "Salary transparency improves candidate quality and reduces drop-off.",
            "suggestion": "Add a realistic range and pay period; include location/level notes if needed."
        })
    if unrealistic:
        quality_flags.append({
            "issue": "Potentially rigid requirements detected.",
            "why_it_matters": "Overly strict requirements reduce qualified applicants and harm diversity.",
            "suggestion": "Separate 'Required' vs 'Preferred' clearly; reduce 'must have' density where possible."
        })

    sev_counts = _severity_counts(bias_findings)

    # Overall scoring (heuristic, conservative)
    score = 1.0
    score -= 0.18 * len(compliance_gaps)
    score -= 0.10 * len(quality_flags)
    score -= 0.05 * sev_counts.get("High", 0)
    score -= 0.03 * sev_counts.get("Medium", 0)
    score -= 0.01 * sev_counts.get("Low", 0)
    score = max(0.0, min(1.0, score))

    band = "Strong" if score >= 0.80 else ("Medium" if score >= 0.55 else "Needs work")

    # Recommended edits checklist (actionable)
    recommended_edits: List[Dict[str, Any]] = []
    for f in bias_findings:
        recommended_edits.append({
            "type": "language",
            "priority": f.get("severity","Low"),
            "change": f"Replace '{f.get('term')}'",
            "why": f.get("rationale"),
            "suggested_replacement": f.get("suggested_replacement"),
            "snippet": f.get("snippet"),
        })
    for u in unrealistic:
        recommended_edits.append({
            "type": "requirements",
            "priority": "Medium",
            "change": u.get("finding","Review requirement"),
            "why": "Rigid requirements can narrow the pool and decrease qualified applicants.",
            "suggested_replacement": "Move to Preferred or clarify equivalency (e.g., 'or equivalent experience').",
            "snippet": u.get("snippet",""),
        })
    for q in quality_flags:
        recommended_edits.append({
            "type": "structure",
            "priority": "Medium",
            "change": q.get("issue","Improve structure"),
            "why": q.get("why_it_matters",""),
            "suggested_replacement": q.get("suggestion",""),
            "snippet": "",
        })
    for c in compliance_gaps:
        recommended_edits.append({
            "type": "compliance",
            "priority": "High",
            "change": c.get("gap","Add compliance language"),
            "why": c.get("why_it_matters",""),
            "suggested_replacement": c.get("suggestion",""),
            "snippet": "",
        })

    # Sort edits by priority
    pr = {"High": 0, "Medium": 1, "Low": 2}
    recommended_edits.sort(key=lambda x: pr.get(x.get("priority","Low"), 2))

    return {
        "overall": {
            "score": score,
            "band": band,
            "disclaimer": "Heuristic audit. Use HR/legal review for final compliance decisions."
        },
        "compliance": {
            "eeo_required": require_eeo,
            "ada_required": require_ada,
            "eeo_detected": has_eeo,
            "ada_detected": has_ada,
            "gaps": compliance_gaps
        },
        "quality": {
            "structure_detected": has_structure,
            "salary_detected": has_salary,
            "clarity_flags": clarity,
            "unrealistic_requirements": unrealistic,
            "bias_findings": bias_findings,
            "severity_counts": sev_counts,
            "quality_flags": quality_flags
        },
        "recommended_edits": recommended_edits
    }
