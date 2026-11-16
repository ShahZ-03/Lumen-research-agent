# backend/vecstore.py — FAISS index management (Study Partner pattern + job_id namespacing)
import os
import faiss
import numpy as np
import pickle
import logging
import threading
from functools import wraps
from uuid import uuid4

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

logger = logging.getLogger("research_agent")


def _locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class LazyVectorStore:
    """Load the embedding model on first use rather than module import."""
    def __init__(self, path="data/faiss_index"):
        self.path = path
        self._store = None
        self._init_lock = threading.Lock()

    def __getattr__(self, name):
        with self._init_lock:
            if self._store is None:
                self._store = VectorStore(self.path)
        return getattr(self._store, name)


class VectorStore:
    def __init__(self, path="data/faiss_index"):
        from sentence_transformers import SentenceTransformer
        self._lock = threading.RLock()
        self.model = SentenceTransformer(MODEL_NAME)
        self.path = path
        self.index = None
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self._ensure_index()

    def _ensure_index(self):
        if os.path.exists(self.path + ".store"):
            with open(self.path + ".store", "rb") as f:
                index_bytes, self.texts, self.metadata = pickle.load(f)
            self.index = faiss.deserialize_index(index_bytes)
        elif os.path.exists(self.path + ".index"):
            self.index = faiss.read_index(self.path + ".index")
            with open(self.path + ".meta", "rb") as f:
                stored = pickle.load(f)
                self.texts, self.metadata = stored
        else:
            self.index = faiss.IndexFlatL2(EMBED_DIM)
            self.texts = []
            self.metadata = []
        if self.index.ntotal != len(self.texts) or len(self.texts) != len(self.metadata):
            raise RuntimeError("Stored vector index and metadata are not aligned")

    @_locked
    def add_texts(self, texts, metadatas=None):
        """Insert texts with optional per-item metadata. Include ``job_id`` in each
        metadata dict for per-job isolation; :meth:`query` can filter by ``job_id``.
        """
        logger.info(
            "VectorStore.add_texts: adding %d texts (index size before %d)",
            len(texts),
            self.index.ntotal,
        )
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(texts) != len(metadatas):
            raise ValueError("Vector texts and metadata must align")
        if not texts:
            return

        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)

        previous_size = self.index.ntotal
        self.index.add(embs.astype("float32"))
        self.texts.extend(texts)
        self.metadata.extend(metadatas)
        try:
            self._save()
        except Exception:
            self.index.remove_ids(np.arange(previous_size, self.index.ntotal, dtype="int64"))
            del self.texts[previous_size:]
            del self.metadata[previous_size:]
            raise
        del embs
        logger.info("VectorStore.add_texts: index size after %d", self.index.ntotal)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        # A single atomic checkpoint prevents index/meta mismatches on interruption.
        # Existing .index/.meta files remain readable as a legacy import format.
        temporary = self.path + f".{uuid4().hex}.tmp"
        try:
            with open(temporary, "wb") as f:
                pickle.dump((faiss.serialize_index(self.index), self.texts, self.metadata), f)
            os.replace(temporary, self.path + ".store")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @_locked
    def query(self, query_text, top_k=5, job_id: str | None = None):
        """Return up to ``top_k`` hits. When ``job_id`` is set, only chunks whose
        metadata contains that ``job_id`` are returned (retrieves extra neighbors
        from FAISS, then filters).
        """
        if self.index.ntotal == 0 or top_k <= 0:
            return []

        q_emb = self.model.encode(
            [query_text], show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")

        if job_id is None:
            D, I = self.index.search(q_emb, min(top_k, self.index.ntotal))
            results = []
            for idx in I[0]:
                if idx >= 0 and idx < len(self.texts):
                    results.append(
                        {"text": self.texts[idx], "metadata": self.metadata[idx]}
                    )
            del q_emb
            return results

        fetch_k = min(self.index.ntotal, max(top_k * 30, top_k))
        results: list[dict] = []
        seen: set[int] = set()
        while len(results) < top_k and fetch_k <= self.index.ntotal:
            D, I = self.index.search(q_emb, fetch_k)
            for idx in I[0]:
                if idx < 0 or idx >= len(self.texts) or int(idx) in seen:
                    continue
                md = self.metadata[idx]
                if md.get("job_id") != job_id:
                    continue
                seen.add(int(idx))
                results.append({"text": self.texts[idx], "metadata": md})
                if len(results) >= top_k:
                    break
            if fetch_k >= self.index.ntotal:
                break
            fetch_k = min(self.index.ntotal, fetch_k * 2)

        del q_emb
        return results[:top_k]
