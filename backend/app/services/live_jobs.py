from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import urllib.parse
import requests

def _tokenize(q: str) -> List[str]:
    q = (q or "").strip()
    parts = re.split(r"[\s,;/\|\(\)\[\]\{\}\-]+", q)
    toks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= 1:
            continue
        toks.append(p.lower())
    return toks[:14]

def _score_text(hay: str, tokens: List[str]) -> int:
    if not hay:
        return 0
    h = hay.lower()
    return sum(1 for t in tokens if t in h)

def _make_search_links(query: str, location: str = "", salary_min: Optional[float] = None) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    loc = (location or "").strip()
    sal = f" salary {int(salary_min)}" if salary_min else ""
    q_full = f"{q} {loc}{sal}".strip()

    targets = [
        ("Google Jobs search", f"https://www.google.com/search?q={urllib.parse.quote(q_full)}&ibp=htl;jobs"),
        ("Bing job search", f"https://www.bing.com/jobs?q={urllib.parse.quote(q_full)}"),
        ("Indeed search", f"https://www.indeed.com/jobs?q={urllib.parse.quote(q)}&l={urllib.parse.quote(loc)}"),
        ("ZipRecruiter search", f"https://www.ziprecruiter.com/jobs-search?search={urllib.parse.quote(q)}&location={urllib.parse.quote(loc)}"),
        ("SimplyHired search", f"https://www.simplyhired.com/search?q={urllib.parse.quote(q)}&l={urllib.parse.quote(loc)}"),
    ]
    return [
        {"source": "search_link", "title": name, "company": "", "location": loc or "", "url": url, "tags": [], "job_type": "", "published_at": ""}
        for name, url in targets
    ]

def fetch_live_jobs(
    query: str,
    limit: int = 15,
    location: str = "",
    salary_min: Optional[float] = None,
    filter_query: str = "",
) -> Dict[str, Any]:
    raw_query = (query or "").strip()
    filter_query = (filter_query or "").strip()
    location = (location or "").strip()

    blended = raw_query
    if filter_query:
        blended = f"{raw_query} {filter_query}".strip() if raw_query else filter_query

    tokens = _tokenize(blended)
    results: List[Dict[str, Any]] = []

    # ---- Remotive (remote-heavy) ----
    try:
        r = requests.get("https://remotive.com/api/remote-jobs", timeout=25)
        r.raise_for_status()
        data = r.json()
        jobs = data.get("jobs", []) or []

        scored = []
        for job in jobs:
            title = job.get("title", "") or ""
            company = job.get("company_name", "") or ""
            loc = job.get("candidate_required_location", "") or "Remote"
            desc = job.get("description", "") or ""
            s = _score_text(title, tokens) + _score_text(desc, tokens) + _score_text(company, tokens) + _score_text(loc, tokens)
            if tokens and s == 0:
                continue
            scored.append((s, job))
        scored.sort(key=lambda x: x[0], reverse=True)

        for s, job in scored[:limit]:
            results.append({
                "source": "remotive",
                "title": job.get("title", ""),
                "company": job.get("company_name", ""),
                "location": job.get("candidate_required_location", "") or "Remote",
                "url": job.get("url", ""),
                "tags": job.get("tags", []),
                "job_type": job.get("job_type", ""),
                "published_at": job.get("publication_date", ""),
                "_match_score": s
            })
            if len(results) >= limit:
                break
    except Exception:
        pass

    # ---- Arbeitnow (EU/remote mix) ----
    if len(results) < limit:
        try:
            r = requests.get("https://www.arbeitnow.com/api/job-board-api", timeout=25)
            r.raise_for_status()
            data = r.json()
            jobs = data.get("data", []) or []

            scored = []
            for job in jobs:
                title = job.get("title", "") or ""
                company = job.get("company_name", "") or ""
                loc = job.get("location", "") or ""
                desc = job.get("description", "") or ""
                s = _score_text(title, tokens) + _score_text(desc, tokens) + _score_text(company, tokens) + _score_text(loc, tokens)
                if tokens and s == 0:
                    continue
                scored.append((s, job))
            scored.sort(key=lambda x: x[0], reverse=True)

            for s, job in scored[: max(0, limit - len(results))]:
                results.append({
                    "source": "arbeitnow",
                    "title": job.get("title", ""),
                    "company": job.get("company_name", ""),
                    "location": job.get("location", ""),
                    "url": job.get("url", ""),
                    "tags": job.get("tags", []),
                    "job_type": "Remote" if job.get("remote") else "",
                    "published_at": "",
                    "_match_score": s
                })
        except Exception:
            pass

    if not results:
        return {
            "ok": True,
            "mode": "search_links_fallback",
            "query": raw_query,
            "filter_query": filter_query,
            "results": _make_search_links(blended or "jobs", location=location, salary_min=salary_min)[:limit],
        }

    return {
        "ok": True,
        "mode": "direct_listings",
        "query": raw_query,
        "filter_query": filter_query,
        "results": results[:limit],
    }
