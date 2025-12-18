from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


MASTER_PATH = _env_path("JOB_DATA_MASTER_PARQUET", "data/merged_model_ready_with_text.parquet")
AUDIT_PATH = _env_path("JOB_DATA_AUDIT_PARQUET", "data/ui_hr_audit_sample.parquet")
REWRITE_PATH = _env_path("JOB_DATA_REWRITE_PARQUET", "data/ui_hr_rewrites_200.parquet")

_MASTER_DF: Optional[pd.DataFrame] = None
_AUDIT_DF: Optional[pd.DataFrame] = None
_REWRITE_DF: Optional[pd.DataFrame] = None

_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF = None  # sparse matrix
_SEARCH_TEXT_COL = "search_text"


@dataclass
class LoadStatus:
    ok: bool
    message: str
    master_loaded: bool
    audit_loaded: bool
    rewrite_loaded: bool
    master_rows: int = 0
    audit_rows: int = 0
    rewrite_rows: int = 0


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        return pd.read_parquet(path)
    except Exception:
        return None


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def ensure_loaded() -> LoadStatus:
    global _MASTER_DF, _AUDIT_DF, _REWRITE_DF, _VECTORIZER, _TFIDF

    if _MASTER_DF is None:
        df = _safe_read_parquet(MASTER_PATH)
        if df is not None and len(df) > 0:
            # Ensure expected text cols exist
            for col in ["title", "company", "location", "jd_text"]:
                if col not in df.columns:
                    df[col] = ""
                df[col] = df[col].fillna("").astype(str)

            # Build a searchable text field. If jd_text is empty for many rows, title/company still helps.
            df[_SEARCH_TEXT_COL] = (
                df["title"].map(_norm)
                + " "
                + df["company"].map(_norm)
                + " "
                + df["location"].map(_norm)
                + " "
                + df["jd_text"].map(_norm)
            )

            # Build TF-IDF index (yes, this is heavy for 1.9M rows, but it will work on a capable machine;
            # if it’s too slow, we’ll switch to chunked search or a smaller sampled index).
            vec = TfidfVectorizer(
                max_features=200_000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
            )
            tfidf = vec.fit_transform(df[_SEARCH_TEXT_COL].tolist())

            _MASTER_DF = df
            _VECTORIZER = vec
            _TFIDF = tfidf

    if _AUDIT_DF is None:
        df = _safe_read_parquet(AUDIT_PATH)
        if df is not None and len(df) > 0:
            _AUDIT_DF = df.copy()

    if _REWRITE_DF is None:
        df = _safe_read_parquet(REWRITE_PATH)
        if df is not None and len(df) > 0:
            _REWRITE_DF = df.copy()

    ok = _MASTER_DF is not None and _VECTORIZER is not None and _TFIDF is not None
    msg = "Loaded master parquet + TF-IDF index." if ok else f"Master parquet not loaded. Check JOB_DATA_MASTER_PARQUET: {MASTER_PATH}"

    return LoadStatus(
        ok=ok,
        message=msg,
        master_loaded=_MASTER_DF is not None,
        audit_loaded=_AUDIT_DF is not None,
        rewrite_loaded=_REWRITE_DF is not None,
        master_rows=int(len(_MASTER_DF)) if _MASTER_DF is not None else 0,
        audit_rows=int(len(_AUDIT_DF)) if _AUDIT_DF is not None else 0,
        rewrite_rows=int(len(_REWRITE_DF)) if _REWRITE_DF is not None else 0,
    )


def status_dict() -> Dict[str, Any]:
    s = ensure_loaded()
    return {
        "ok": s.ok,
        "message": s.message,
        "paths": {"master": str(MASTER_PATH), "audit": str(AUDIT_PATH), "rewrite": str(REWRITE_PATH)},
        "loaded": {"master": s.master_loaded, "audit": s.audit_loaded, "rewrite": s.rewrite_loaded},
        "rows": {"master": s.master_rows, "audit": s.audit_rows, "rewrite": s.rewrite_rows},
        "master_columns": list(_MASTER_DF.columns) if _MASTER_DF is not None else [],
    }


def get_master_preview(n: int = 5) -> List[Dict[str, Any]]:
    ensure_loaded()
    if _MASTER_DF is None:
        return []
    n = max(1, min(20, int(n)))
    return _MASTER_DF.head(n).to_dict(orient="records")


def search_master(
    query: str,
    k: int = 10,
    location: str = "",
    salary_min: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Returns rows shaped for your Streamlit cards:
      title, company, location, _score plus salary_annual_clean if available.
    """
    s = ensure_loaded()
    if not s.ok or _MASTER_DF is None or _VECTORIZER is None or _TFIDF is None:
        return []

    df = _MASTER_DF
    k = max(1, min(50, int(k)))

    q = (query or "").strip()
    if not q:
        q = "general jobs"

    q_vec = _VECTORIZER.transform([_norm(q)])
    sims = cosine_similarity(q_vec, _TFIDF).ravel()

    mask = np.ones(len(df), dtype=bool)

    if location:
        loc = location.strip().lower()
        loc_series = df["location"].str.lower()
        # keep remote or matching location
        mask &= loc_series.str.contains(re.escape(loc), na=False) | loc_series.str.contains("remote", na=False)

    if salary_min is not None and "salary_annual_clean" in df.columns:
        try:
            sal = pd.to_numeric(df["salary_annual_clean"], errors="coerce")
            mask &= sal.isna() | (sal >= float(salary_min))
        except Exception:
            pass

    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    top = idx[np.argsort(sims[idx])[::-1][:k]]

    out: List[Dict[str, Any]] = []
    for i in top:
        r = df.iloc[int(i)]
        out.append(
            {
                "title": r.get("title", ""),
                "company": r.get("company", ""),
                "location": r.get("location", ""),
                "_score": float(sims[int(i)]),
                "salary_annual_clean": None
                if pd.isna(r.get("salary_annual_clean", np.nan))
                else float(r.get("salary_annual_clean")),
                "_source_file": r.get("_source_file", ""),
                "has_salary_clean": bool(r.get("has_salary_clean", False)),
                "has_jd_text": bool(r.get("has_jd_text", False)),
            }
        )
    return out
