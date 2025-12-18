import os
import numpy as np
import pandas as pd

# Optional FAISS
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    faiss = None
    FAISS_OK = False

class AppStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.jobs_meta_path = os.path.join(data_dir, "jobs_meta_for_search.parquet")
        self.jobs_emb_path = os.path.join(data_dir, "jobs_emb.npy")
        self.audit_path = os.path.join(data_dir, "ui_hr_audit_sample.parquet")

        self.jobs_meta: pd.DataFrame | None = None
        self.emb: np.ndarray | None = None
        self.index = None  # FAISS index if available

    def load_all(self):
        # jobs meta
        if os.path.exists(self.jobs_meta_path):
            self.jobs_meta = pd.read_parquet(self.jobs_meta_path)
        else:
            self.jobs_meta = pd.DataFrame()

        # embeddings + index
        if os.path.exists(self.jobs_emb_path):
            emb = np.load(self.jobs_emb_path).astype(np.float32)

            # normalize for cosine similarity via inner product
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
            self.emb = emb

            if FAISS_OK and emb.ndim == 2 and emb.shape[0] > 0:
                idx = faiss.IndexFlatIP(emb.shape[1])
                idx.add(emb)
                self.index = idx
        else:
            self.emb = None
            self.index = None

    def status(self):
        return {
            "jobs_meta_exists": os.path.exists(self.jobs_meta_path),
            "jobs_emb_exists": os.path.exists(self.jobs_emb_path),
            "faiss_ok": FAISS_OK,
            "jobs_rows": int(0 if self.jobs_meta is None else len(self.jobs_meta)),
            "emb_rows": int(0 if self.emb is None else self.emb.shape[0]),
        }
