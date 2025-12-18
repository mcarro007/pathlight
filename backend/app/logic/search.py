import re
import numpy as np
import pandas as pd
from openai import OpenAI

def _tokenize(q: str):
    return [w for w in re.split(r"\W+", (q or "").lower()) if len(w) > 2]

def keyword_search(jobs: pd.DataFrame, query: str, k: int = 20, location_filter: str = "") -> pd.DataFrame:
    if jobs is None or len(jobs) == 0:
        return pd.DataFrame()

    q = (query or "").strip().lower()
    if not q:
        out = jobs.head(k).copy()
        out["match_score"] = 0.0
        return out

    cols = [c for c in ["title", "company", "location", "jd_text"] if c in jobs.columns]
    if not cols:
        out = jobs.head(k).copy()
        out["match_score"] = 0.0
        return out

    blob = jobs[cols].astype("string").fillna("").agg(" ".join, axis=1).str.lower()
    tokens = _tokenize(q)
    if not tokens:
        out = jobs.head(k).copy()
        out["match_score"] = 0.0
        return out

    pattern = "|".join([re.escape(t) for t in tokens])
    score = blob.str.count(pattern)

    out = jobs.copy()
    out["match_score"] = score.astype(float)

    if location_filter.strip() and "location" in out.columns:
        out = out[out["location"].astype("string").str.contains(location_filter, case=False, na=False)]

    out = out.sort_values("match_score", ascending=False)
    return out.head(k).copy()

def embed_query(client: OpenAI, text: str, embed_model: str) -> np.ndarray:
    resp = client.embeddings.create(model=embed_model, input=[text])
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def semantic_search(
    client: OpenAI,
    jobs: pd.DataFrame,
    emb: np.ndarray | None,
    index,
    query: str,
    embed_model: str,
    k: int = 20,
    location_filter: str = "",
) -> pd.DataFrame:
    if jobs is None or len(jobs) == 0:
        return pd.DataFrame()

    if emb is None:
        return keyword_search(jobs=jobs, query=query, k=k, location_filter=location_filter)

    qv = embed_query(client, query, embed_model)

    if index is not None:
        scores, idxs = index.search(qv, k * 5)
        idxs = idxs.flatten().tolist()
        scores = scores.flatten().tolist()
    else:
        sims = (emb @ qv.T).reshape(-1)
        idxs = np.argsort(-sims)[: k * 5].tolist()
        scores = sims[idxs].tolist()

    rows = jobs.iloc[idxs].copy()
    rows["match_score"] = scores

    if location_filter.strip() and "location" in rows.columns:
        rows = rows[rows["location"].astype("string").str.contains(location_filter, case=False, na=False)]

    return rows.head(k).copy()

def salary_from_similar(similar_rows: pd.DataFrame):
    if similar_rows is None or len(similar_rows) == 0:
        return None
    if "salary_annual_clean" not in similar_rows.columns:
        return None

    s = pd.to_numeric(similar_rows["salary_annual_clean"], errors="coerce").dropna()
    if len(s) < 10:
        return None

    lo = float(np.percentile(s, 25))
    hi = float(np.percentile(s, 75))
    lo = round(lo / 1000) * 1000
    hi = round(hi / 1000) * 1000
    if lo <= 0 or hi <= 0 or hi < lo:
        return None
    return lo, hi
