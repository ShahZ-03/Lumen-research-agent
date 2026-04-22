# backend/vecstore.py — FAISS index management (Study Partner pattern + job_id namespacing)
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import logging

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

logger = logging.getLogger("research_agent")


class VectorStore:
    def __init__(self, path="data/faiss_index"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.path = path
        self.index = None
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self._ensure_index()

    def _ensure_index(self):
        if os.path.exists(self.path + ".index"):
            self.index = faiss.read_index(self.path + ".index")
            with open(self.path + ".meta", "rb") as f:
                stored = pickle.load(f)
                self.texts, self.metadata = stored
        else:
            self.index = faiss.IndexFlatL2(EMBED_DIM)
            self.texts = []
            self.metadata = []

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

        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)

        self.index.add(embs.astype("float32"))
        self.texts.extend(texts)
        self.metadata.extend(metadatas)
        self._save()
        del embs
        logger.info("VectorStore.add_texts: index size after %d", self.index.ntotal)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        faiss.write_index(self.index, self.path + ".index")
        with open(self.path + ".meta", "wb") as f:
            pickle.dump((self.texts, self.metadata), f)

    def query(self, query_text, top_k=5, job_id: str | None = None):
        """Return up to ``top_k`` hits. When ``job_id`` is set, only chunks whose
        metadata contains that ``job_id`` are returned (retrieves extra neighbors
        from FAISS, then filters).
        """
        if self.index.ntotal == 0:
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
        while len(results) < top_k and fetch_k <= self.index.ntotal:
            D, I = self.index.search(q_emb, fetch_k)
            for idx in I[0]:
                if idx < 0 or idx >= len(self.texts):
                    continue
                md = self.metadata[idx]
                if md.get("job_id") != job_id:
                    continue
                results.append({"text": self.texts[idx], "metadata": md})
                if len(results) >= top_k:
                    break
            if fetch_k >= self.index.ntotal:
                break
            fetch_k = min(self.index.ntotal, fetch_k * 2)

        del q_emb
        return results[:top_k]
