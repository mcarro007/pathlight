import re
import numpy as np

BIAS_TERMS = [
    "rockstar", "ninja", "guru", "dominant", "aggressive", "fearless",
    "young", "digital native", "recent graduate", "energetic", "cultural fit",
]
EXCLUSION_GATES = ["must have", "required", "no exceptions", "only", "strictly"]
DEGREE_GATES = ["bachelor", "masters", "phd", "degree required", "must have a degree"]
YEARS_PATTERNS = [r"\b(\d+)\+?\s*(years|yrs)\b", r"\bminimum\s+of\s+(\d+)\s*(years|yrs)\b"]

def _count_years_mentions(text: str) -> int:
    tl = (text or "").lower()
    hits = 0
    for pat in YEARS_PATTERNS:
        hits += len(re.findall(pat, tl))
    return hits

def _count_bullets(text: str) -> int:
    lines = (text or "").splitlines()
    bullet_like = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("-", "*", "•")):
            bullet_like += 1
        if re.match(r"^\d+\.", s):
            bullet_like += 1
    return bullet_like

def heuristic_consumer_jd_eval(jd_text: str):
    t = (jd_text or "").strip()
    tl = t.lower()

    bullet_count = _count_bullets(t)
    years_mentions = _count_years_mentions(t)
    degree_hits = sum(1 for k in DEGREE_GATES if k in tl)
    bias_hits = [k for k in BIAS_TERMS if k in tl]
    gate_hits = sum(1 for k in EXCLUSION_GATES if k in tl)

    qual_score = 0
    qual_score += min(40, bullet_count * 2)
    qual_score += min(20, years_mentions * 7)
    qual_score += min(20, degree_hits * 10)
    qual_score += min(20, gate_hits * 5)
    qual_score = int(max(0, min(100, qual_score)))

    bias_score = 0
    bias_score += min(35, len(bias_hits) * 10)
    if "equal opportunity" not in tl and "eeo" not in tl:
        bias_score += 10
    if "reasonable accommodation" not in tl and "accommodation" not in tl:
        bias_score += 5
    if "citizen" in tl and "only" in tl:
        bias_score += 10
    bias_score = int(max(0, min(100, bias_score)))

    length = len(t)
    clarity_penalty = 0
    if length < 800:
        clarity_penalty += 10
    if length > 7000:
        clarity_penalty += 10

    base = 70
    callback = base - (qual_score * 0.45) - (bias_score * 0.25) - clarity_penalty
    callback = int(max(5, min(95, round(callback))))

    flags = []
    if bullet_count >= 20:
        flags.append("Very long list of requirements/responsibilities (often a 'unicorn' listing).")
    if years_mentions >= 2:
        flags.append("Multiple rigid 'years of experience' mentions.")
    if degree_hits >= 1:
        flags.append("Degree gate detected (can exclude strong non-traditional candidates).")
    for bh in bias_hits:
        flags.append(f"Potentially coded term found: '{bh}'")

    suggestions = []
    if bullet_count >= 18:
        suggestions.append("If you apply, focus your resume on the top 6–10 requirements and mirror the language.")
    if years_mentions >= 1:
        suggestions.append("Treat years-of-experience as flexible if you can show equivalent projects and impact.")
    if degree_hits >= 1:
        suggestions.append("If your experience is strong, apply anyway and frame 'equivalent experience' clearly.")
    if len(bias_hits) > 0:
        suggestions.append("Coded terms can signal culture-fit bias. Look for structured evaluation criteria or avoid.")

    return {
        "bias_risk_score": bias_score,
        "qualification_pressure_score": qual_score,
        "estimated_callback_likelihood": callback,
        "evidence": {
            "bullet_count": bullet_count,
            "years_mentions": years_mentions,
            "degree_gate_mentions": degree_hits,
            "coded_terms": bias_hits,
            "exclusion_gate_hits": gate_hits,
            "char_length": length,
        },
        "flags": flags,
        "suggestions": suggestions,
        "disclaimer": "Callback likelihood is a heuristic estimate based only on posting text. Not a promise.",
    }
