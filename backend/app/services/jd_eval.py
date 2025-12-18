from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Bias term library (enterprise-safe defaults)
# -----------------------------
# We intentionally avoid ambiguous words (e.g., "strong", "understanding") that create false positives.
# Terms here are higher-signal and commonly cited as biased-coded in recruiting contexts.

MASCULINE_CODED = [
    "rockstar", "ninja", "dominant", "aggressive", "fearless", "assertive",
    "competitive", "decisive", "autonomous", "commanding",
]

FEMININE_CODED = [
    "nurturing", "empathetic", "supportive", "interpersonal",
]

AGE_CODED = [
    "digital native", "recent graduate", "young", "energetic",
    "work hard play hard",
]

# "ableist" is extremely context-dependent.
# We DO NOT flag "stand" or "walk" by themselves.
# We only flag if it is clearly a *physical requirement* (which should be in an ADA-compliant format).
PHYSICAL_REQUIREMENT_TRIGGERS = [
    r"must be able to\s+(stand|walk|lift|carry|bend|kneel)",
    r"ability to\s+(stand|walk|lift|carry|bend|kneel)",
    r"stand for\s+\d+\s*(hours|hrs)",
    r"lift\s+\d+\s*(pounds|lbs)",
]

EXCLUSIONARY = [
    "native english", "native speaker", "no accent",
]

# Suggestions (replace with neutral language)
SUGGESTIONS = {
    "rockstar": "high-performing contributor",
    "ninja": "experienced specialist",
    "dominant": "high-impact",
    "aggressive": "proactive",
    "fearless": "confident",
    "assertive": "clear and direct communicator",
    "competitive": "results-oriented",
    "decisive": "able to make timely decisions",
    "autonomous": "self-directed",
    "commanding": "confident presenter",
    "nurturing": "supportive",
    "empathetic": "customer-focused",
    "supportive": "collaborative",
    "interpersonal": "strong communication",
    "digital native": "comfortable learning new tools",
    "recent graduate": "early-career candidates welcome",
    "work hard play hard": "fast-moving environment with team collaboration",
    "native english": "professional proficiency in English",
    "native speaker": "professional proficiency in English",
    "no accent": "clear verbal communication",
}

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _find_snippet(text: str, start: int, end: int, pad: int = 60) -> str:
    t = text or ""
    a = max(0, start - pad)
    b = min(len(t), end + pad)
    snippet = t[a:b]
    return _normalize_ws(snippet)

def _word_boundary_find(text: str, term: str) -> List[Tuple[int, int]]:
    # Word boundary for single words; phrase search for multi-word terms.
    t = text or ""
    if " " in term:
        matches = []
        idx = 0
        lower_t = t.lower()
        lower_term = term.lower()
        while True:
            j = lower_t.find(lower_term, idx)
            if j == -1:
                break
            matches.append((j, j + len(term)))
            idx = j + len(term)
        return matches

    pat = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    return [(m.start(), m.end()) for m in pat.finditer(t)]

def _physical_requirement_hits(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    t = text or ""
    hits: List[Tuple[str, Tuple[int, int]]] = []
    for raw in PHYSICAL_REQUIREMENT_TRIGGERS:
        pat = re.compile(raw, flags=re.IGNORECASE)
        for m in pat.finditer(t):
            hits.append((m.group(0), (m.start(), m.end())))
    return hits

def _severity_for(category: str) -> str:
    # Conservative severity
    if category in ("exclusionary", "ableist_physical_requirement"):
        return "High"
    if category in ("age_coded",):
        return "Medium"
    return "Low"

def _confidence_for(category: str) -> str:
    if category in ("ableist_physical_requirement", "exclusionary"):
        return "High"
    if category in ("masculine_coded", "age_coded"):
        return "Medium"
    return "Medium"

def detect_bias_findings(text: str) -> List[Dict[str, Any]]:
    """
    Returns granular findings:
      - term / phrase
      - category
      - snippet
      - rationale
      - severity, confidence
      - suggestions
    """
    t = text or ""
    findings: List[Dict[str, Any]] = []

    # Masculine-coded
    for term in MASCULINE_CODED:
        for (s, e) in _word_boundary_find(t, term):
            findings.append({
                "category": "masculine_coded",
                "term": term,
                "span": {"start": s, "end": e},
                "snippet": _find_snippet(t, s, e),
                "rationale": "This term is commonly interpreted as masculine-coded and can discourage some candidates. Consider neutral phrasing.",
                "severity": _severity_for("masculine_coded"),
                "confidence": _confidence_for("masculine_coded"),
                "suggested_replacement": SUGGESTIONS.get(term, "Use neutral role language"),
            })

    # Feminine-coded
    for term in FEMININE_CODED:
        for (s, e) in _word_boundary_find(t, term):
            findings.append({
                "category": "feminine_coded",
                "term": term,
                "span": {"start": s, "end": e},
                "snippet": _find_snippet(t, s, e),
                "rationale": "This term can be interpreted as gender-coded. Neutral skills-based language is safer for broad candidate appeal.",
                "severity": _severity_for("feminine_coded"),
                "confidence": _confidence_for("feminine_coded"),
                "suggested_replacement": SUGGESTIONS.get(term, "Use skills-based phrasing"),
            })

    # Age-coded
    for term in AGE_CODED:
        for (s, e) in _word_boundary_find(t, term):
            findings.append({
                "category": "age_coded",
                "term": term,
                "span": {"start": s, "end": e},
                "snippet": _find_snippet(t, s, e),
                "rationale": "Age-coded language can imply a preference for younger candidates. Use experience-neutral phrasing.",
                "severity": _severity_for("age_coded"),
                "confidence": _confidence_for("age_coded"),
                "suggested_replacement": SUGGESTIONS.get(term, "Use experience-neutral phrasing"),
            })

    # Exclusionary language
    for term in EXCLUSIONARY:
        for (s, e) in _word_boundary_find(t, term):
            findings.append({
                "category": "exclusionary",
                "term": term,
                "span": {"start": s, "end": e},
                "snippet": _find_snippet(t, s, e),
                "rationale": "This can exclude qualified candidates and may create compliance risk. Prefer proficiency-based requirements.",
                "severity": _severity_for("exclusionary"),
                "confidence": _confidence_for("exclusionary"),
                "suggested_replacement": SUGGESTIONS.get(term, "Use proficiency-based phrasing"),
            })

    # ADA-sensitive physical requirements (context-only)
    phys_hits = _physical_requirement_hits(t)
    for phrase, (s, e) in phys_hits:
        findings.append({
            "category": "ableist_physical_requirement",
            "term": phrase,
            "span": {"start": s, "end": e},
            "snippet": _find_snippet(t, s, e),
            "rationale": "Physical requirements should be stated carefully and paired with an accommodation statement. Ensure this is essential to the role.",
            "severity": _severity_for("ableist_physical_requirement"),
            "confidence": _confidence_for("ableist_physical_requirement"),
            "suggested_replacement": "If essential: describe as 'essential functions' and include accommodation language.",
        })

    # Deduplicate by (category, span)
    seen = set()
    unique = []
    for f in findings:
        key = (f["category"], f["span"]["start"], f["span"]["end"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)

    # Sort by severity then position
    sev_rank = {"High": 0, "Medium": 1, "Low": 2}
    unique.sort(key=lambda x: (sev_rank.get(x.get("severity","Low"), 2), x["span"]["start"]))
    return unique

def detect_unrealistic(text: str) -> List[Dict[str, Any]]:
    t = text or ""
    patterns = [
        (re.compile(r"\b(\d{2,})\+?\s*(years|yrs)\b", re.IGNORECASE), "Very high years-of-experience requirement."),
        (re.compile(r"\b(10\+)\s*(years|yrs)\b", re.IGNORECASE), "10+ years requested (check if realistic for level)."),
        (re.compile(r"\b(phd|doctorate)\b", re.IGNORECASE), "Doctorate requirement (may narrow candidate pool)."),
        (re.compile(r"\b(master['’]s|ms|m\.s\.)\b", re.IGNORECASE), "Master's requirement (may narrow candidate pool)."),
        (re.compile(r"\b(must have)\b", re.IGNORECASE), "Heavy 'must have' language (may be overly rigid)."),
    ]
    out: List[Dict[str, Any]] = []
    for pat, msg in patterns:
        m = pat.search(t)
        if m:
            out.append({
                "finding": msg,
                "snippet": _find_snippet(t, m.start(), m.end()),
            })
    return out

def basic_clarity_flags(text: str) -> List[str]:
    t = _normalize_ws(text)
    flags = []
    if len(t) < 200:
        flags.append("JD is very short; may be missing responsibilities, requirements, benefits, or leveling.")
    lowered = t.lower()
    if not any(h in lowered for h in ["responsibilities", "requirements", "qualifications", "what you'll do", "benefits", "compensation"]):
        flags.append("Headings not detected (Responsibilities/Requirements/Benefits). Add clear sections.")
    if "salary" not in lowered and "compensation" not in lowered and "$" not in t:
        flags.append("No salary/compensation detected. Consider adding a range to improve conversion and trust.")
    if not any(x in lowered for x in ["remote", "hybrid", "on-site", "onsite"]):
        flags.append("Work location model not clearly stated (remote/hybrid/on-site).")
    return flags

def _flesch_reading_ease(text: str) -> Optional[float]:
    # Simple heuristic; keeps deps minimal
    t = re.sub(r"[^a-zA-Z\s\.\!\?]", " ", text or "")
    sentences = re.split(r"[.!?]+", t)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"[a-zA-Z]+", t)
    if not sentences or not words:
        return None

    def syllables(word: str) -> int:
        w = word.lower()
        groups = re.findall(r"[aeiouy]+", w)
        count = len(groups)
        if w.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    total_sentences = len(sentences)
    total_words = len(words)
    total_syllables = sum(syllables(w) for w in words)
    return 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

def heuristic_callback_likelihood(match_score: float, bias_findings: List[Dict[str, Any]], unrealistic_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Heuristic only.
    score = float(match_score or 0.0)

    # Penalize only high/medium severity items to avoid over-penalizing
    penalty = 0.0
    for f in bias_findings:
        if f.get("severity") == "High":
            penalty += 0.08
        elif f.get("severity") == "Medium":
            penalty += 0.05
        else:
            penalty += 0.02

    penalty += 0.06 * len(unrealistic_findings)

    adj = max(0.0, min(1.0, score - penalty))
    if adj >= 0.70:
        band = "High (heuristic)"
        note = "Strong skills match. Tailor resume to measurable outcomes and key requirements."
    elif adj >= 0.50:
        band = "Medium (heuristic)"
        note = "Decent match. Improve alignment by mirroring JD keywords and showing proof of impact."
    else:
        band = "Low (heuristic)"
        note = "Weak match or role is rigid. Consider adjacent titles or adjust targeting."

    return {
        "callback_likelihood_band": band,
        "callback_likelihood_score": adj,
        "explanation": note,
        "disclaimer": "Heuristic estimate only. Real outcomes depend on market competition, timing, ATS filters, referrals, and employer process."
    }

def suggest_industries_and_paths_from_titles(titles: List[str]) -> Dict[str, Any]:
    industry_rules = {
        "Healthcare": ["clinical", "health", "hospital", "pharma", "medical"],
        "Finance": ["bank", "trading", "fintech", "insurance"],
        "Retail": ["retail", "ecommerce", "merchandising"],
        "Cybersecurity": ["security", "soc", "threat", "vulnerability"],
        "Data/AI": ["data", "ml", "ai", "machine learning", "analytics", "scientist"],
        "Gov/Defense": ["federal", "dod", "clearance", "government"],
        "SaaS/Tech": ["platform", "cloud", "saas", "software", "devops"],
    }
    scores = {k: 0 for k in industry_rules.keys()}
    for t in titles or []:
        tl = (t or "").lower()
        for ind, kws in industry_rules.items():
            if any(kw in tl for kw in kws):
                scores[ind] += 1
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = [k for k, v in ranked if v > 0][:5]
    paths = [
        "Data Analyst → Senior Data Analyst → Analytics Manager",
        "Data Analyst → Data Scientist → Senior Data Scientist",
        "Data Scientist → Applied ML Engineer → ML Lead",
        "BI Analyst → BI Engineer → Data Platform Specialist",
        "Healthcare Data Analyst → Clinical Data Scientist → Health AI Specialist",
    ]
    return {
        "suggested_industries": top if top else ["Data/AI", "SaaS/Tech"],
        "suggested_career_paths": paths,
        "how_generated": "Derived from keyword signals in top matched job titles (local index) and standard role progressions."
    }

def evaluate_job_description(jd_text: str, match_score: float, matched_titles: Optional[List[str]] = None) -> Dict[str, Any]:
    jd = jd_text or ""
    bias_findings = detect_bias_findings(jd)
    unrealistic_findings = detect_unrealistic(jd)
    clarity = basic_clarity_flags(jd)
    readability = _flesch_reading_ease(jd)

    worth_applying = True
    reasons: List[str] = []

    if unrealistic_findings and float(match_score or 0.0) < 0.45:
        worth_applying = False
        reasons.append("Requirements appear rigid and match score is low. Consider targeting adjacent roles or upskilling gaps.")

    if any(f.get("severity") == "High" for f in bias_findings):
        reasons.append("High-severity language or compliance risk detected. Consider whether the employer is aligned with inclusive hiring practices.")

    if readability is not None and readability < 30:
        reasons.append("JD is hard to read and may be unclear. Unclear roles can lead to mismatch and churn.")

    callback = heuristic_callback_likelihood(match_score, bias_findings, unrealistic_findings)
    paths = suggest_industries_and_paths_from_titles(matched_titles or [])

    return {
        "bias_findings": bias_findings,
        "unrealistic_requirements": unrealistic_findings,
        "clarity_flags": clarity,
        "readability_flesch": readability,
        "match_score": match_score,
        "worth_applying_heuristic": worth_applying,
        "worth_applying_reasons": reasons,
        "callback_estimate": callback,
        "industry_and_paths": paths,
    }
