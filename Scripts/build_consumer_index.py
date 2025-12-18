import os
import time
import random
import numpy as np
import pandas as pd
from openai import OpenAI

PROJECT_ROOT = r"C:\Users\micha\job-analyzer"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MASTER_PATH = os.path.join(DATA_DIR, "merged_model_ready_with_text.parquet")

OUT_META_PATH = os.path.join(DATA_DIR, "jobs_meta_for_search.parquet")
OUT_EMB_PATH = os.path.join(DATA_DIR, "jobs_emb.npy")
OUT_PROGRESS_PATH = os.path.join(DATA_DIR, "jobs_emb_progress.txt")

EMBED_MODEL = "text-embedding-3-small"

# Smaller batches help with flaky TLS connections
BATCH_SIZE = 32
# Start smaller for stability; raise later once it's working end-to-end
MAX_ROWS = 50_000

MAX_RETRIES = 10


def make_search_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "")
    company = str(row.get("company", "") or "")
    location = str(row.get("location", "") or "")
    jd = str(row.get("jd_text", "") or "")
    jd = jd[:1200]
    return (
        f"Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Job Description: {jd}"
    )


def load_resume_index() -> int:
    if not os.path.exists(OUT_PROGRESS_PATH):
        return 0
    try:
        with open(OUT_PROGRESS_PATH, "r", encoding="utf-8") as f:
            s = (f.read() or "").strip()
        return int(s) if s else 0
    except Exception:
        return 0


def save_resume_index(i: int) -> None:
    with open(OUT_PROGRESS_PATH, "w", encoding="utf-8") as f:
        f.write(str(i))


def embed_with_retries(client: OpenAI, chunk, i0: int) -> np.ndarray:
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return vecs
        except Exception as e:
            last_err = e
            # exponential backoff + jitter (helps when TLS is flaky)
            sleep_s = min(60, (2 ** attempt)) + random.random() * 2
            print(
                f"[WARN] Embed failed at row {i0}. "
                f"Attempt {attempt}/{MAX_RETRIES}. "
                f"Sleeping {sleep_s:.1f}s. "
                f"Error: {type(e).__name__}: {e}"
            )
            time.sleep(sleep_s)

    raise last_err


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in this session.")

    client = OpenAI(api_key=api_key)

    df = pd.read_parquet(MASTER_PATH)

    if "jd_text" in df.columns:
        df = df[df["jd_text"].notna()].copy()

    df = df.head(MAX_ROWS).copy()

    keep_cols = [
        "title",
        "company",
        "location",
        "salary_annual_clean",
        "pay_period_norm",
        "_source_file",
        "jd_text",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["search_text"] = df.apply(make_search_text, axis=1)
    texts = df["search_text"].tolist()

    # Save meta immediately so your Streamlit app can run while embeddings build
    df.to_parquet(OUT_META_PATH, index=False)
    print("Meta saved to:", OUT_META_PATH)

    start_i = load_resume_index()
    if start_i:
        print(f"[RESUME] Continuing from {start_i:,}/{len(texts):,}")

    vectors = []
    if start_i and os.path.exists(OUT_EMB_PATH):
        vectors.append(np.load(OUT_EMB_PATH).astype(np.float32))

    for i in range(start_i, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        vecs = embed_with_retries(client, chunk, i0=i)
        vectors.append(vecs)

        done = min(i + BATCH_SIZE, len(texts))
        emb = np.vstack(vectors)

        # checkpoint every batch
        np.save(OUT_EMB_PATH, emb)
        save_resume_index(done)

        print(f"Embedded {done:,}/{len(texts):,}")

    print("DONE")
    print("Embeddings saved to:", OUT_EMB_PATH, emb.shape)


if __name__ == "__main__":
    main()
