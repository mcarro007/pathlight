import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ============================================================
# Optional: FAISS for faster semantic search (works without it)
# ============================================================
try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False


# ============================================================
# CONFIG / PATHS
# ============================================================
PROJECT_ROOT = r"C:\Users\micha\job-analyzer"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

AUDIT_PATH = os.path.join(DATA_DIR, "ui_hr_audit_sample.parquet")
REWRITE_PATH = os.path.join(DATA_DIR, "ui_hr_rewrites_200.parquet")  # optional
MASTER_PATH = os.path.join(DATA_DIR, "merged_model_ready_with_text.parquet")

JOBS_META_PATH = os.path.join(DATA_DIR, "jobs_meta_for_search.parquet")
JOBS_EMB_PATH = os.path.join(DATA_DIR, "jobs_emb.npy")

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

st.set_page_config(page_title="Job Analyzer (Enterprise)", layout="wide")


# ============================================================
# OPENAI CLIENT
# ============================================================
def get_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None, False
    return OpenAI(api_key=api_key), True


# ============================================================
# LOADERS
# ============================================================
@st.cache_data
def load_audit():
    if not os.path.exists(AUDIT_PATH):
        return pd.DataFrame()
    return pd.read_parquet(AUDIT_PATH)

@st.cache_data
def load_rewrites():
    if not os.path.exists(REWRITE_PATH):
        return pd.DataFrame()
    return pd.read_parquet(REWRITE_PATH)

@st.cache_data
def load_jobs_meta():
    if not os.path.exists(JOBS_META_PATH):
        return pd.DataFrame()
    return pd.read_parquet(JOBS_META_PATH)

@st.cache_resource
def load_embeddings_and_index():
    if not os.path.exists(JOBS_EMB_PATH):
        return None, None

    emb = np.load(JOBS_EMB_PATH).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    if FAISS_OK:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return emb, index

    return emb, None


# ============================================================
# SAFE HELPERS
# ============================================================
def ensure_list(x):
    """Safely coerce list-like / numpy / pandas / string / NaN into a Python list."""
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)

    if hasattr(x, "tolist"):
        try:
            v = x.tolist()
            if isinstance(v, list):
                return v
            return [v]
        except Exception:
            pass

    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []

    return [str(x)]

def clean_list_strings(items):
    out = []
    for it in ensure_list(items):
        s = str(it).strip()
        if not s or s.lower() == "nan":
            continue
        out.append(s)
    return out

def extract_first_json_object(text: str):
    if not text or not text.strip():
        return None, "Empty response"
    t = text.strip().replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None, "No JSON found"
    blob = m.group(0).strip()
    try:
        return json.loads(blob), None
    except Exception as e:
        return None, str(e)

def clamp_int(x, lo=0, hi=100, default=0):
    try:
        v = int(round(float(x)))
        return max(lo, min(hi, v))
    except Exception:
        return default


# ============================================================
# SEARCH / EMBEDDINGS
# ============================================================
def embed_query(text: str) -> np.ndarray:
    client, ok = get_client()
    if not ok:
        st.error("OPENAI_API_KEY is not set. Set it and restart Streamlit.")
        st.stop()

    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def keyword_search(query: str, k: int = 20, location_filter: str = "") -> pd.DataFrame:
    jobs = load_jobs_meta()
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
    tokens = [w for w in re.split(r"\W+", q) if len(w) > 2]
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

def semantic_search(query: str, k: int = 20, location_filter: str = "") -> pd.DataFrame:
    jobs = load_jobs_meta()
    emb, index = load_embeddings_and_index()

    if emb is None or len(jobs) == 0:
        return keyword_search(query=query, k=k, location_filter=location_filter)

    qv = embed_query(query)

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


# ============================================================
# CONSUMER JD HEURISTICS (ENTERPRISE STYLE)
# ============================================================
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
    qual_score = max(0, min(100, qual_score))

    bias_score = 0
    bias_score += min(35, len(bias_hits) * 10)
    if "equal opportunity" not in tl and "eeo" not in tl:
        bias_score += 10
    if "reasonable accommodation" not in tl and "accommodation" not in tl:
        bias_score += 5
    if "citizen" in tl and "only" in tl:
        bias_score += 10
    bias_score = max(0, min(100, bias_score))

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
        "disclaimer": "This callback likelihood is a heuristic estimate based only on posting text. It is not a promise.",
    }


# ============================================================
# PROMPTS (ENTERPRISE)
# ============================================================
CONSUMER_JD_LLM_SYSTEM = """You are a careful hiring analyst and career coach.
Evaluate a pasted job description for bias risk, qualification overload, and applicant friction.

Return ONLY valid JSON:
{
  "bias_risk_score": number,
  "qualification_pressure_score": number,
  "estimated_callback_likelihood": number,
  "summary": "string",
  "top_flags": ["string","..."],
  "what_to_do": ["string","..."],
  "quick_rewrite_suggestions": ["string","..."],
  "disclaimer": "string"
}

Rules:
- All scores must be numbers 0-100.
- JSON only. No markdown. No extra keys.
"""

REWRITE_SYSTEM_V2 = """You are an HR writing assistant.
Rewrite job descriptions to be inclusive, unbiased, and clear.
Preserve responsibilities and required qualifications unless they are explicitly biased or exclusionary.
Use plain language. Do not add new requirements. Do not remove compliance language.

Return ONLY valid JSON:
{
  "rewritten_jd": "string",
  "change_log": ["string","..."],
  "risks": ["string","..."]
}
Rules: JSON only. No markdown. No extra keys.
"""

ROLE_NAMER_SYSTEM = """You are an HR org design specialist.
Given what the company needs done, propose the best-fit role title and level.

Return ONLY valid JSON:
{
  "recommended_job_title": "string",
  "recommended_level": "Entry|Mid|Senior|Lead|Manager",
  "alternative_titles": ["string","..."],
  "why_this_title": "string"
}
Rules: JSON only. No extra keys.
"""

JOB_BUILDER_SYSTEM = """You are an expert HR job description writer.
Create a clear, inclusive job posting based on inputs.

Return ONLY valid JSON:
{
  "job_title": "string",
  "level": "string",
  "salary_range": {"min": number, "max": number, "currency": "USD"},
  "job_description": "string",
  "assumptions": ["string","..."]
}
Rules:
- JSON only. No markdown. No extra keys.
- salary_range min/max must be numeric.
"""

def make_consumer_llm_prompt(jd_text: str, user_skills: str, experience: str):
    return f"""User:
Skills: {user_skills}
Experience level: {experience}

Job description:
\"\"\"{jd_text}\"\"\"

IMPORTANT:
- Output must start with {{ and end with }}.
"""

def make_rewrite_user_prompt(jd_text: str, flags):
    cleaned = clean_list_strings(flags)
    flags_str = ", ".join(cleaned) if len(cleaned) > 0 else "none detected"
    return f"""Rewrite this job description.

Bias flags detected: {flags_str}

Job description:
\"\"\"{jd_text}\"\"\"

IMPORTANT:
- Output must start with {{ and end with }}.
- change_log must be a JSON array of strings.
- risks must be a JSON array of strings.

Return only JSON.
"""

def make_role_namer_user_prompt(team_context, responsibilities, required_skills, preferred_skills, location, remote_policy):
    return f"""Company needs:

Team context:
{team_context}

Responsibilities:
{responsibilities}

Required skills:
{required_skills}

Preferred skills:
{preferred_skills}

Location: {location}
Remote policy: {remote_policy}

Return only JSON.
"""

def make_job_builder_user_prompt(job_title, level, location, remote_policy, team_context,
                                 responsibilities, required_skills, preferred_skills,
                                 compliance, extra_instructions, market_comp, salary_anchor_text):
    return f"""Build a job posting using these inputs.

Job title: {job_title}
Seniority level: {level}
Location: {location}
Remote policy: {remote_policy}
Market competitiveness: {market_comp}

Team context:
{team_context}

Responsibilities:
{responsibilities}

Required skills:
{required_skills}

Preferred skills:
{preferred_skills}

Compliance / constraints:
{compliance}

Extra instructions:
{extra_instructions}

Salary guidance:
{salary_anchor_text}

Return only JSON.
"""


# ============================================================
# UI: GLOBAL CSS + SIDEBAR NAV + SETTINGS
# ============================================================
client, api_ok = get_client()
emb_exists = os.path.exists(JOBS_EMB_PATH)
meta_exists = os.path.exists(JOBS_META_PATH)

# Sidebar toggles first (so CSS can react)
with st.sidebar:
    st.markdown("## Job Analyzer")
    st.markdown("<div style='opacity:0.8;font-size:13px;'>Enterprise UI build</div>", unsafe_allow_html=True)
    st.divider()

    mode = st.radio("Workspace", ["Consumer", "Corporate (HR)"], index=0, key="nav_mode")

    st.markdown("### System status")
    st.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;"
        f"border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.04);margin-right:6px;"
        f"{'border-color: rgba(46,204,113,0.45);' if api_ok else 'border-color: rgba(231,76,60,0.55);'}'>"
        f"API {'OK' if api_ok else 'Missing'}</span>"
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;"
        f"border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.04);margin-right:6px;"
        f"{'border-color: rgba(46,204,113,0.45);' if meta_exists else 'border-color: rgba(231,76,60,0.55);'}'>"
        f"Meta {'OK' if meta_exists else 'Missing'}</span>"
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;"
        f"border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.04);margin-right:6px;"
        f"{'border-color: rgba(46,204,113,0.45);' if emb_exists else 'border-color: rgba(241,196,15,0.55);'}'>"
        f"Embeddings {'OK' if emb_exists else 'Not ready'}</span>"
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;"
        f"border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.04);margin-right:6px;"
        f"{'border-color: rgba(46,204,113,0.45);' if FAISS_OK else 'border-color: rgba(241,196,15,0.55);'}'>"
        f"FAISS {'OK' if FAISS_OK else 'Off'}</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### UI preferences")
    compact = st.toggle("Compact layout", value=False, key="ui_compact")
    show_debug = st.toggle("Show debug panels", value=False, key="ui_debug")

    st.divider()
    st.markdown("### Saved items")
    if "saved_jobs" not in st.session_state:
        st.session_state["saved_jobs"] = []

    if st.session_state["saved_jobs"]:
        st.write(f"Saved: **{len(st.session_state['saved_jobs'])}**")
        if st.button("Clear saved", key="clear_saved"):
            st.session_state["saved_jobs"] = []
            st.rerun()
    else:
        st.caption("No saved jobs yet.")

# Global CSS (reacts to compact)
container_max = "980px" if st.session_state.get("ui_compact") else "1200px"
pad_top = "0.75rem" if st.session_state.get("ui_compact") else "1.25rem"

st.markdown(
    f"""
<style>
.block-container {{
  padding-top: {pad_top};
  padding-bottom: 2rem;
  max-width: {container_max};
}}

.card {{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 14px;
}}

.h2 {{ font-size: 22px; font-weight: 750; margin: 0 0 6px 0; }}
.h3 {{ font-size: 18px; font-weight: 700; margin: 0 0 6px 0; }}
.subtle {{ opacity: 0.82; font-size: 13px; }}

.hr {{
  height: 1px;
  background: rgba(255,255,255,0.08);
  border: 0;
  margin: 14px 0;
}}

.small-note {{
  opacity: 0.80;
  font-size: 12px;
}}

.kpi {{
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 10px 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h2'>Job Analyzer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>Enterprise prototype: Consumer decision support + HR audit, rewrite, and job creation workflows.</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.get("ui_debug"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>Debug</div>", unsafe_allow_html=True)
    st.code(
        "\n".join([
            f"PROJECT_ROOT: {PROJECT_ROOT}",
            f"DATA_DIR: {DATA_DIR}",
            f"AUDIT_PATH exists: {os.path.exists(AUDIT_PATH)}",
            f"JOBS_META_PATH exists: {os.path.exists(JOBS_META_PATH)}",
            f"JOBS_EMB_PATH exists: {os.path.exists(JOBS_EMB_PATH)}",
            f"FAISS_OK: {FAISS_OK}",
            f"API OK: {api_ok}",
        ]),
        language="text"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SAVED ITEMS HELPERS (ENTERPRISE WORKFLOW)
# ============================================================
def _job_key(row: pd.Series):
    return (
        str(row.get("title", "")),
        str(row.get("company", "")),
        str(row.get("location", "")),
        str(row.get("_source_file", "")),
    )

def add_saved_from_df(df: pd.DataFrame, selected_mask: pd.Series):
    if df is None or len(df) == 0:
        return
    if selected_mask is None:
        return

    saved = st.session_state.get("saved_jobs", [])
    existing_keys = set()
    for x in saved:
        existing_keys.add((x.get("title"), x.get("company"), x.get("location"), x.get("_source_file")))

    for idx, sel in selected_mask.items():
        if not bool(sel):
            continue
        row = df.loc[idx]
        key = (
            str(row.get("title", "")),
            str(row.get("company", "")),
            str(row.get("location", "")),
            str(row.get("_source_file", "")),
        )
        if key in existing_keys:
            continue
        saved.append(
            {
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "location": str(row.get("location", "")),
                "salary_annual_clean": row.get("salary_annual_clean", None),
                "_source_file": str(row.get("_source_file", "")),
            }
        )
        existing_keys.add(key)

    st.session_state["saved_jobs"] = saved


def render_saved_panel():
    saved = st.session_state.get("saved_jobs", [])
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>Saved items</div>", unsafe_allow_html=True)

    if not saved:
        st.caption("Save jobs from results tables to build a shortlist here.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    sdf = pd.DataFrame(saved)
    show_cols = [c for c in ["title", "company", "location", "salary_annual_clean", "_source_file"] if c in sdf.columns]
    st.dataframe(sdf[show_cols], use_container_width=True, height=240)

    csv = sdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download saved list (CSV)",
        data=csv,
        file_name="saved_jobs.csv",
        mime="text/csv",
        key="saved_dl_csv_main",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CONSUMER VIEW (ENTERPRISE)
# ============================================================
def consumer_view():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>Consumer workspace</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Search roles, build a shortlist, and analyze a posting before you apply.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Section: Skill-based job suggestions ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>Skill-based job suggestions</div>", unsafe_allow_html=True)
    st.caption("Semantic search uses embeddings (if present). If embeddings are missing, it falls back to keyword match.")

    colA, colB = st.columns([3, 2], gap="large")
    with colA:
        skills = st.text_area(
            "Your skillset",
            height=110,
            value="Python, data analysis, dashboards, stakeholder communication",
            key="consumer_skills",
        )
        interests = st.text_area(
            "Interests (optional)",
            height=80,
            value="healthcare, public sector, research",
            key="consumer_interests",
        )

    with colB:
        experience = st.selectbox(
            "Experience level",
            ["Any", "Entry", "Mid", "Senior"],
            index=0,
            key="consumer_exp",
        )
        location_pref = st.text_input("Location filter (optional)", value="", key="consumer_loc")
        k = st.slider("Number of suggestions", 5, 30, 15, key="consumer_k")

        query = f"Skills: {skills}\nInterests: {interests}\nExperience: {experience}".strip()

        st.write("")
        do_jobs = st.button("Run job search", key="consumer_do_jobs")

    if do_jobs:
        rows = semantic_search(query=query, k=k, location_filter=location_pref)
        st.session_state["consumer_job_rows"] = rows

    rows = st.session_state.get("consumer_job_rows", pd.DataFrame())

    # KPI summary bar
    if isinstance(rows, pd.DataFrame) and len(rows) > 0:
        total = len(rows)
        salary_col = "salary_annual_clean" if "salary_annual_clean" in rows.columns else None
        salary_count = int(pd.to_numeric(rows[salary_col], errors="coerce").notna().sum()) if salary_col else 0
        top_loc = ""
        if "location" in rows.columns:
            vc = rows["location"].astype("string").fillna("").value_counts()
            top_loc = str(vc.index[0]) if len(vc) else ""
        med_salary = None
        if salary_col:
            s = pd.to_numeric(rows[salary_col], errors="coerce").dropna()
            if len(s):
                med_salary = float(np.median(s))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Results", total)
        c2.metric("With salary", salary_count)
        c3.metric("Top location", top_loc if top_loc else "—")
        c4.metric("Median salary", f"${med_salary:,.0f}" if med_salary else "—")

        # Editable table with save checkbox
        st.markdown("<hr class='hr' />", unsafe_allow_html=True)
        st.markdown("**Results**")

        display_cols = [c for c in ["match_score", "title", "company", "location", "salary_annual_clean", "_source_file"] if c in rows.columns]
        view_df = rows[display_cols].copy()
        view_df.insert(0, "Save", False)

        edited = st.data_editor(
            view_df,
            use_container_width=True,
            height=340,
            key="consumer_results_editor",
            column_config={
                "Save": st.column_config.CheckboxColumn("Save", help="Add to saved shortlist"),
                "match_score": st.column_config.NumberColumn("Score", format="%.3f"),
            },
            disabled=[c for c in view_df.columns if c != "Save"],
        )

        col_save, col_dl = st.columns([1, 2])
        with col_save:
            if st.button("Add selected to saved", key="consumer_add_saved"):
                add_saved_from_df(rows, edited["Save"])
                st.success("Saved list updated.")
        with col_dl:
            csv = rows.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results (CSV)",
                data=csv,
                file_name="consumer_job_suggestions.csv",
                mime="text/csv",
                key="consumer_dl_csv",
            )

    else:
        st.markdown("<div class='small-note'>Run a search to see results and KPIs.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end card

    # --------- Section: Career paths + Application plan ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>Career paths and application plan</div>", unsafe_allow_html=True)
    st.caption("These use OpenAI. If API key is missing, the buttons will be disabled.")

    query = f"Skills: {st.session_state.get('consumer_skills','')}\nInterests: {st.session_state.get('consumer_interests','')}\nExperience: {st.session_state.get('consumer_exp','')}".strip()
    location_pref = st.session_state.get("consumer_loc", "")
    k = int(st.session_state.get("consumer_k", 15))

    b1, b2, b3 = st.columns(3)
    do_careers = b1.button("Suggest careers / paths", key="consumer_do_careers")
    do_plan = b2.button("Generate application plan", key="consumer_do_plan")
    show_saved = b3.button("Show saved list", key="consumer_show_saved")

    if show_saved:
        st.session_state["consumer_show_saved_panel"] = True

    if do_careers:
        if not api_ok:
            st.error("OPENAI_API_KEY is not set. Set it and restart Streamlit.")
        else:
            sample = semantic_search(query=query, k=max(k, 25), location_filter=location_pref)
            titles = sample["title"].astype("string").fillna("").tolist() if "title" in sample.columns else []
            payload = "\n".join([f"- {t}" for t in titles[:30]])

            system = "You are a career coach. Suggest realistic career paths based on the user's skills and example job matches."
            user = f"""User info:
{query}

Example matched job titles:
{payload}

Return JSON only:
{{
  "career_paths": [
    {{
      "career_name": "string",
      "why_it_fits": "string",
      "roles_to_search": ["string","..."]
    }}
  ]
}}
Rules:
- JSON only
- Provide 6 to 10 career_paths
"""
            resp = client.responses.create(
                model=LLM_MODEL,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            raw = (resp.output_text or "").strip()
            data, err = extract_first_json_object(raw)
            if err:
                st.error("Could not parse JSON: " + err)
                st.code(raw[:2000])
            else:
                st.session_state["consumer_careers"] = data

    careers_data = st.session_state.get("consumer_careers", None)
    if careers_data:
        st.markdown("**Career paths**")
        for c in careers_data.get("career_paths", []):
            st.markdown(f"**{c.get('career_name','')}**")
            st.write(c.get("why_it_fits", ""))
            roles = ensure_list(c.get("roles_to_search"))
            if roles:
                st.write("Roles to search:")
                for r in roles:
                    st.write("• " + str(r))

    if do_plan:
        if not api_ok:
            st.error("OPENAI_API_KEY is not set. Set it and restart Streamlit.")
        else:
            sample = semantic_search(query=query, k=25, location_filter=location_pref)
            targets = []
            for _, r in sample.head(12).iterrows():
                targets.append(
                    {
                        "title": str(r.get("title", "")),
                        "company": str(r.get("company", "")),
                        "location": str(r.get("location", "")),
                    }
                )

            system = "You are a practical job-search strategist. Produce a focused application plan."
            user = f"""User info:
{query}

Suggested target roles:
{json.dumps(targets, indent=2)}

Return JSON only:
{{
  "top_targets": [
    {{
      "priority": 1,
      "target_title": "string",
      "why_this": "string",
      "search_terms": ["string", "..."],
      "next_steps": ["string", "..."]
    }}
  ],
  "weekly_plan": ["string","..."]
}}
Rules:
- JSON only
- 8 to 12 top_targets
- weekly_plan should be 6 to 10 bullet-like strings
"""
            resp = client.responses.create(
                model=LLM_MODEL,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            raw = (resp.output_text or "").strip()
            data, err = extract_first_json_object(raw)
            if err:
                st.error("Could not parse JSON: " + err)
                st.code(raw[:2000])
            else:
                st.session_state["consumer_plan"] = data

    plan = st.session_state.get("consumer_plan", None)
    if plan:
        st.markdown("**Application plan**")
        for t in plan.get("top_targets", []):
            st.markdown(f"**Priority {t.get('priority','')} — {t.get('target_title','')}**")
            st.write(t.get("why_this", ""))
            st.write("Search terms:")
            for s in ensure_list(t.get("search_terms")):
                st.write("• " + str(s))
            st.write("Next steps:")
            for s in ensure_list(t.get("next_steps")):
                st.write("• " + str(s))

        st.markdown("**Weekly plan**")
        for s in ensure_list(plan.get("weekly_plan")):
            st.write("• " + str(s))

    st.markdown("</div>", unsafe_allow_html=True)

    # --------- Section: Paste JD to analyze ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>Check a job description before you apply</div>", unsafe_allow_html=True)
    st.caption("Heuristic scorecard + optional OpenAI deep analysis. Includes an estimated callback likelihood (heuristic).")

    jd_in = st.text_area("Paste the job description", height=240, key="consumer_jd_in")
    use_llm = st.checkbox("Use OpenAI deep analysis (recommended)", value=True, key="consumer_use_llm")

    colX, colY = st.columns([1, 2])
    with colX:
        analyze = st.button("Analyze posting", key="consumer_analyze_jd")
    with colY:
        st.caption("Tip: For best results, paste the full posting including requirements and responsibilities.")

    if analyze:
        if not jd_in.strip():
            st.warning("Paste a job description first.")
        else:
            baseline = heuristic_consumer_jd_eval(jd_in)
            st.session_state["consumer_jd_baseline"] = baseline

            data_llm, llm_err = None, None
            if use_llm and api_ok:
                prompt = make_consumer_llm_prompt(
                    jd_text=jd_in,
                    user_skills=st.session_state.get("consumer_skills", ""),
                    experience=st.session_state.get("consumer_exp", "Any"),
                )
                resp = client.responses.create(
                    model=LLM_MODEL,
                    input=[{"role": "system", "content": CONSUMER_JD_LLM_SYSTEM}, {"role": "user", "content": prompt}],
                )
                raw = (resp.output_text or "").strip()
                data_llm, llm_err = extract_first_json_object(raw)
                if llm_err:
                    data_llm = None
                else:
                    data_llm["bias_risk_score"] = clamp_int(data_llm.get("bias_risk_score", 0))
                    data_llm["qualification_pressure_score"] = clamp_int(data_llm.get("qualification_pressure_score", 0))
                    data_llm["estimated_callback_likelihood"] = clamp_int(data_llm.get("estimated_callback_likelihood", 0))
            elif use_llm and not api_ok:
                llm_err = "OPENAI_API_KEY is not set. Showing heuristic analysis only."

            st.session_state["consumer_jd_llm"] = data_llm
            st.session_state["consumer_jd_llm_err"] = llm_err

            sim = semantic_search(query=f"Job description:\n{jd_in[:2500]}", k=15, location_filter=st.session_state.get("consumer_loc", ""))
            st.session_state["consumer_jd_similar"] = sim

    baseline = st.session_state.get("consumer_jd_baseline", None)
    if baseline:
        c1, c2, c3 = st.columns(3)
        c1.metric("Bias risk (0–100)", baseline["bias_risk_score"])
        c2.metric("Qualification pressure (0–100)", baseline["qualification_pressure_score"])
        c3.metric("Estimated callback likelihood", baseline["estimated_callback_likelihood"])
        st.caption(baseline.get("disclaimer", ""))

        with st.expander("Evidence + recommendations (heuristic)", expanded=False):
            st.write("Evidence:", baseline.get("evidence", {}))
            st.write("Flags:")
            for f in baseline.get("flags", []):
                st.warning(f)
            st.write("Suggested actions:")
            for s in baseline.get("suggestions", []):
                st.info(s)

    llm_err = st.session_state.get("consumer_jd_llm_err", None)
    if llm_err:
        st.info(llm_err)

    llm_data = st.session_state.get("consumer_jd_llm", None)
    if llm_data:
        st.markdown("**OpenAI deep analysis**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Bias risk", llm_data.get("bias_risk_score", 0))
        d2.metric("Qualification pressure", llm_data.get("qualification_pressure_score", 0))
        d3.metric("Callback likelihood", llm_data.get("estimated_callback_likelihood", 0))
        st.write(llm_data.get("summary", ""))

        st.write("Top flags:")
        for f in ensure_list(llm_data.get("top_flags")):
            st.warning(str(f))

        st.write("What to do:")
        for x in ensure_list(llm_data.get("what_to_do")):
            st.info(str(x))

        st.write("Quick rewrite suggestions:")
        for x in ensure_list(llm_data.get("quick_rewrite_suggestions")):
            st.write("• " + str(x))

        st.caption(llm_data.get("disclaimer", ""))

    sim = st.session_state.get("consumer_jd_similar", pd.DataFrame())
    if isinstance(sim, pd.DataFrame) and len(sim) > 0:
        st.markdown("<hr class='hr' />", unsafe_allow_html=True)
        st.markdown("**Similar postings from your dataset**")

        show_cols = [c for c in ["title", "company", "location", "salary_annual_clean", "_source_file"] if c in sim.columns]
        sim_view = sim[show_cols].copy()
        sim_view.insert(0, "Save", False)

        edited_sim = st.data_editor(
            sim_view,
            use_container_width=True,
            height=260,
            key="consumer_sim_editor",
            column_config={"Save": st.column_config.CheckboxColumn("Save")},
            disabled=[c for c in sim_view.columns if c != "Save"],
        )

        if st.button("Add selected similar postings to saved", key="consumer_add_saved_sim"):
            add_saved_from_df(sim, edited_sim["Save"])
            st.success("Saved list updated.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Saved panel on main page
    if st.session_state.get("consumer_show_saved_panel") or st.session_state.get("saved_jobs"):
        render_saved_panel()


# ============================================================
# HR VIEW (ENTERPRISE)
# ============================================================
def hr_view():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h2'>Corporate (HR) workspace</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Audit browser + rewrite studio + role creation workflows.</div>", unsafe_allow_html=True)
    if not api_ok:
        st.warning("OPENAI_API_KEY is not set. Rewrite and generation will be disabled until you set it and restart Streamlit.")
    st.markdown("</div>", unsafe_allow_html=True)

    hr_tab1, hr_tab2, hr_tab3 = st.tabs([
        "Audit + Rewrite Existing JD",
        "Create JD + Salary (Direct Input)",
        "Role Finder + JD Builder (From Needs)"
    ])

    # ---------------- HR: Audit + Rewrite ----------------
    with hr_tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h3'>Audit browser and rewrite studio</div>", unsafe_allow_html=True)
        st.caption("Filter audited postings by bias score, open a row, and generate a rewritten JD.")

        audit = load_audit()
        if audit is None or len(audit) == 0:
            st.info("Audit dataset not found or empty. Check your file in /data.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            left, right = st.columns([1, 2], gap="large")

            with left:
                min_score = st.slider("Minimum bias score", 0, 100, 10, key="hr_min_score")
                # guard if missing column
                if "bias_score" in audit.columns:
                    view = audit[audit["bias_score"] >= min_score].sort_values("bias_score", ascending=False).head(300)
                else:
                    view = audit.head(300).copy()

                cols_to_show = [c for c in ["title", "company", "bias_score", "bias_flags", "_source_file"] if c in view.columns]
                st.dataframe(view[cols_to_show], height=520, use_container_width=True)

                pick_idx = st.number_input("Pick row index (0..)", min_value=0, value=0, step=1, key="hr_pick_idx")

            with right:
                if len(view) == 0:
                    st.info("No rows match that filter.")
                else:
                    pick_idx = min(int(pick_idx), len(view) - 1)
                    row = view.iloc[pick_idx]

                    st.markdown(f"### {row.get('title','')}")
                    st.write(
                        f"**Company:** {row.get('company','')}  |  "
                        f"**Location:** {row.get('location','')}  |  "
                        f"**Bias score:** {row.get('bias_score','')}"
                    )

                    flags_str = ", ".join(clean_list_strings(row.get("bias_flags", []))) or "none"
                    st.write(f"**Flags:** {flags_str}")

                    st.markdown("#### Original JD")
                    st.text_area("Original JD", value=row.get("jd_text", ""), height=240, key="hr_orig_jd")

                    st.markdown("#### Rewrite this JD")
                    if not api_ok:
                        st.info("Set OPENAI_API_KEY to enable rewrites.")
                    else:
                        if st.button("Rewrite with OpenAI", key="hr_rewrite_btn"):
                            resp = client.responses.create(
                                model=LLM_MODEL,
                                input=[
                                    {"role": "system", "content": REWRITE_SYSTEM_V2},
                                    {"role": "user", "content": make_rewrite_user_prompt(row.get("jd_text", ""), row.get("bias_flags", []))},
                                ],
                            )
                            raw = (resp.output_text or "").strip()
                            data, err = extract_first_json_object(raw)
                            if err:
                                st.error("Could not parse JSON: " + err)
                                st.code(raw[:2000])
                            else:
                                st.markdown("#### Rewritten JD")
                                st.text_area("Rewritten JD", value=data.get("rewritten_jd", ""), height=280, key="hr_rewritten")

                                st.markdown("#### Change log")
                                for item in ensure_list(data.get("change_log")):
                                    st.write("• " + str(item))

                                st.markdown("#### Risks / notes")
                                for item in ensure_list(data.get("risks")):
                                    st.write("• " + str(item))

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- HR: Create JD + Salary (Direct Input) ----------------
    with hr_tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h3'>Create JD + salary (direct input)</div>", unsafe_allow_html=True)
        st.caption("Enter job details and generate a posting with salary grounded from your dataset when possible.")

        c1, c2 = st.columns(2)
        with c1:
            job_title = st.text_input("Job title", value="Data Analyst", key="hr2_job_title")
            level = st.selectbox("Seniority level", ["Entry", "Mid", "Senior", "Lead", "Manager"], index=1, key="hr2_level")
            location = st.text_input("Location", value="Remote", key="hr2_location")
            remote_policy = st.selectbox("Remote policy", ["Remote", "Hybrid", "On-site"], index=0, key="hr2_remote")
            market_comp = st.selectbox("Market competitiveness", ["low", "standard", "high"], index=1, key="hr2_comp")

        with c2:
            team_context = st.text_area("Team / mission context", height=120, value="This role supports stakeholders with reporting and insights.", key="hr2_team")
            compliance = st.text_area("Compliance / constraints", height=120, value="Include EEO language. Avoid exclusionary phrasing.", key="hr2_compliance")

        responsibilities = st.text_area("Responsibilities", height=140, value="- Build dashboards\n- Analyze trends\n- Present insights", key="hr2_resp")
        required_skills = st.text_area("Required skills", height=110, value="SQL, Excel, data visualization, communication", key="hr2_req")
        preferred_skills = st.text_area("Preferred skills", height=110, value="Python, Tableau/Power BI, statistics", key="hr2_pref")
        extra_instructions = st.text_area("Extra instructions", height=110, value="Tone: modern and clear. Include benefits placeholder. Keep it realistic.", key="hr2_extra")

        similar = semantic_search(
            query=f"Title: {job_title}\nLevel: {level}\nLocation: {location}\nResponsibilities: {responsibilities}\nRequired: {required_skills}\nPreferred: {preferred_skills}",
            k=60,
            location_filter=location if location.strip() else "",
        )
        salary_hint = salary_from_similar(similar)

        # KPI bar
        s1, s2, s3 = st.columns(3)
        s1.metric("Similar roles sampled", len(similar) if isinstance(similar, pd.DataFrame) else 0)
        if salary_hint:
            s2.metric("Grounded salary (25–75%)", f"${salary_hint[0]:,.0f}–${salary_hint[1]:,.0f}")
            salary_anchor_text = f"Use this grounded salary range: min={salary_hint[0]}, max={salary_hint[1]}, currency=USD."
        else:
            s2.metric("Grounded salary (25–75%)", "—")
            salary_anchor_text = "No grounded salary range available. Estimate conservatively and state assumptions."
        s3.metric("Remote policy", remote_policy)

        if salary_hint:
            st.success("Salary guidance grounded from your dataset.")
        else:
            st.warning("Could not ground salary from dataset (not enough salary signal). Will estimate with assumptions.")

        if not api_ok:
            st.info("Set OPENAI_API_KEY to enable generation.")
        else:
            if st.button("Generate JD + Salary", key="hr2_generate"):
                user_prompt = make_job_builder_user_prompt(
                    job_title=job_title,
                    level=level,
                    location=location,
                    remote_policy=remote_policy,
                    team_context=team_context,
                    responsibilities=responsibilities,
                    required_skills=required_skills,
                    preferred_skills=preferred_skills,
                    compliance=compliance,
                    extra_instructions=extra_instructions,
                    market_comp=market_comp,
                    salary_anchor_text=salary_anchor_text,
                )
                resp = client.responses.create(
                    model=LLM_MODEL,
                    input=[{"role": "system", "content": JOB_BUILDER_SYSTEM}, {"role": "user", "content": user_prompt}],
                )
                raw = (resp.output_text or "").strip()
                data, err = extract_first_json_object(raw)
                if err:
                    st.error("Could not parse JSON: " + err)
                    st.code(raw[:2000])
                else:
                    st.session_state["hr2_out"] = data

        out = st.session_state.get("hr2_out", None)
        if out:
            st.markdown("<hr class='hr' />", unsafe_allow_html=True)
            st.subheader("Generated job posting")
            st.write(f"**Job title:** {out.get('job_title','')}  |  **Level:** {out.get('level','')}")
            sr = out.get("salary_range", {}) or {}
            st.write(f"**Salary range:** {sr.get('min','')} – {sr.get('max','')} {sr.get('currency','USD')}")
            st.text_area("Job description", value=out.get("job_description", ""), height=380, key="hr2_jd_out")
            st.write("Assumptions:")
            for a in ensure_list(out.get("assumptions")):
                st.write("• " + str(a))

            st.download_button(
                "Download output (JSON)",
                data=json.dumps(out, indent=2).encode("utf-8"),
                file_name="hr_generated_job_posting.json",
                mime="application/json",
                key="hr2_dl_json",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- HR: Role Finder + Builder (From Needs) ----------------
    with hr_tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h3'>Role finder + job builder (from needs)</div>", unsafe_allow_html=True)
        st.caption("Describe what you need done. System recommends a title/level, grounds salary, and generates a full JD.")

        c1, c2 = st.columns(2)
        with c1:
            location = st.text_input("Location", value="Remote", key="hr3_location")
            remote_policy = st.selectbox("Remote policy", ["Remote", "Hybrid", "On-site"], index=0, key="hr3_remote")
            market_comp = st.selectbox("Market competitiveness", ["low", "standard", "high"], index=1, key="hr3_comp")

        with c2:
            compliance = st.text_area("Compliance / constraints", height=110, value="Include EEO language. Avoid exclusionary phrasing.", key="hr3_compliance")
            extra_instructions = st.text_area("Extra instructions", height=110, value="Tone: modern, clear. Include benefits placeholder.", key="hr3_extra")

        team_context = st.text_area("Team / mission context", height=110, value="We need someone to support reporting and insights across stakeholders.", key="hr3_team")
        responsibilities = st.text_area("What needs to get done (responsibilities)", height=150, value="- Build dashboards\n- Analyze trends\n- Present insights", key="hr3_resp")
        required_skills = st.text_area("Required skills", height=110, value="SQL, Excel, data visualization, communication", key="hr3_req")
        preferred_skills = st.text_area("Preferred skills", height=110, value="Python, Tableau/Power BI, statistics", key="hr3_pref")

        if not api_ok:
            st.info("Set OPENAI_API_KEY to enable role naming and job generation.")
        else:
            cA, cB = st.columns(2)
            do_name = cA.button("Suggest role title + level", key="hr3_name")
            do_build = cB.button("Generate JD + Salary", key="hr3_build")

            if do_name:
                resp = client.responses.create(
                    model=LLM_MODEL,
                    input=[
                        {"role": "system", "content": ROLE_NAMER_SYSTEM},
                        {"role": "user", "content": make_role_namer_user_prompt(
                            team_context=team_context,
                            responsibilities=responsibilities,
                            required_skills=required_skills,
                            preferred_skills=preferred_skills,
                            location=location,
                            remote_policy=remote_policy,
                        )},
                    ],
                )
                raw = (resp.output_text or "").strip()
                data, err = extract_first_json_object(raw)
                if err:
                    st.error("Could not parse JSON: " + err)
                    st.code(raw[:2000])
                else:
                    st.session_state["hr3_role"] = data

            role = st.session_state.get("hr3_role", None)
            if role:
                st.markdown("<hr class='hr' />", unsafe_allow_html=True)
                st.subheader("Recommended role")
                st.write(f"**Title:** {role.get('recommended_job_title','')}")
                st.write(f"**Level:** {role.get('recommended_level','')}")
                st.write(role.get("why_this_title",""))
                alts = ensure_list(role.get("alternative_titles"))
                if alts:
                    st.write("Alternative titles:")
                    for t in alts:
                        st.write("• " + str(t))

            if do_build:
                role = st.session_state.get("hr3_role", None)
                if not role:
                    st.warning("Click 'Suggest role title + level' first.")
                else:
                    job_title = role.get("recommended_job_title", "")
                    level = role.get("recommended_level", "Mid")

                    similar = semantic_search(
                        query=f"Title: {job_title}\nLevel: {level}\nLocation: {location}\nResponsibilities: {responsibilities}\nRequired: {required_skills}\nPreferred: {preferred_skills}",
                        k=60,
                        location_filter=location if location.strip() else "",
                    )
                    salary_hint = salary_from_similar(similar)

                    if salary_hint:
                        salary_anchor_text = f"Use this grounded salary range: min={salary_hint[0]}, max={salary_hint[1]}, currency=USD."
                    else:
                        salary_anchor_text = "No grounded salary range available. Estimate conservatively and state assumptions."

                    user_prompt = make_job_builder_user_prompt(
                        job_title=job_title,
                        level=level,
                        location=location,
                        remote_policy=remote_policy,
                        team_context=team_context,
                        responsibilities=responsibilities,
                        required_skills=required_skills,
                        preferred_skills=preferred_skills,
                        compliance=compliance,
                        extra_instructions=extra_instructions,
                        market_comp=market_comp,
                        salary_anchor_text=salary_anchor_text,
                    )

                    resp = client.responses.create(
                        model=LLM_MODEL,
                        input=[
                            {"role": "system", "content": JOB_BUILDER_SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    raw = (resp.output_text or "").strip()
                    data, err = extract_first_json_object(raw)
                    if err:
                        st.error("Could not parse JSON: " + err)
                        st.code(raw[:2000])
                    else:
                        st.session_state["hr3_out"] = data

        out = st.session_state.get("hr3_out", None)
        if out:
            st.markdown("<hr class='hr' />", unsafe_allow_html=True)
            st.subheader("Generated job posting")
            st.write(f"**Job title:** {out.get('job_title','')}  |  **Level:** {out.get('level','')}")
            sr = out.get("salary_range", {}) or {}
            st.write(f"**Salary range:** {sr.get('min','')} – {sr.get('max','')} {sr.get('currency','USD')}")
            st.text_area("Job description", value=out.get("job_description", ""), height=420, key="hr3_jd_out")
            st.write("Assumptions:")
            for a in ensure_list(out.get("assumptions")):
                st.write("• " + str(a))

            st.download_button(
                "Download output (JSON)",
                data=json.dumps(out, indent=2).encode("utf-8"),
                file_name="hr_role_finder_job_posting.json",
                mime="application/json",
                key="hr3_dl_json",
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ROUTER (SIDEBAR NAV)
# ============================================================
if st.session_state.get("nav_mode", "Consumer") == "Consumer":
    consumer_view()
else:
    hr_view()
