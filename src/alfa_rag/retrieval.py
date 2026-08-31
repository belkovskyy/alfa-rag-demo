from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # allow import for docs/tests without faiss

from sentence_transformers import SentenceTransformer


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure expected columns exist and have sane dtypes."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["score", "web_id", "title", "url", "preview", "text"])
    out = df.copy()
    for c in ["score", "web_id", "title", "url", "preview", "text"]:
        if c not in out.columns:
            out[c] = "" if c != "score" else 0.0
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    return out


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s


def _build_preview(text: str, max_chars: int = 320) -> str:
    t = _clean_text(text)
    if len(t) > max_chars:
        return t[:max_chars].rstrip() + "…"
    return t


@dataclass
class Retriever:
    """
    Dense retriever over chunks with FAISS + doc-level aggregation.

    Expected chunk columns in parquet:
      - web_id (int/str)
      - title (str)
      - url (str)
      - text (str)  # chunk text
    """
    chunks_df: pd.DataFrame
    embedder: SentenceTransformer
    index: Any
    chunk_emb: np.ndarray

    @staticmethod
    def _embed_passages(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
        # E5: passage prefix
        xs = [f"passage: {t}" for t in texts]
        emb = embedder.encode(xs, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype="float32")

    @staticmethod
    def _embed_query(embedder: SentenceTransformer, query: str) -> np.ndarray:
        q = f"query: {query}"
        emb = embedder.encode([q], normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype="float32")[0]

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        *,
        chunks_path: str = "chunks_websites.parquet",
        embed_model: str = "intfloat/multilingual-e5-small",
        device: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> "Retriever":
        data_dir = Path(data_dir)
        cache_dir = Path(cache_dir) if cache_dir else data_dir

        chunks_fp = data_dir / chunks_path
        if not chunks_fp.exists():
            raise FileNotFoundError(f"Missing chunks parquet: {chunks_fp}")

        df = pd.read_parquet(chunks_fp)
        for c in ["web_id", "title", "url", "text"]:
            if c not in df.columns:
                raise ValueError(f"chunks parquet must contain column '{c}'")

        if device is None:
            # SentenceTransformer accepts "cuda"/"cpu"
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        embedder = SentenceTransformer(embed_model, device=device)

        emb_fp = cache_dir / "chunks_emb.npy"
        index_fp = cache_dir / "chunks.index.faiss"

        if faiss is None:
            raise ImportError("faiss is not available. Install faiss-cpu or faiss-gpu.")

        expected_dim = int(embedder.get_sentence_embedding_dimension())
        cache_is_valid = False
        if emb_fp.exists() and index_fp.exists():
            chunk_emb = np.load(emb_fp)
            index = faiss.read_index(str(index_fp))
            cache_is_valid = (
                chunk_emb.shape[0] == len(df)
                and chunk_emb.shape[1] == expected_dim
                and index.d == expected_dim
            )

        if not cache_is_valid:
            texts = df["text"].astype(str).tolist()
            chunk_emb = cls._embed_passages(embedder, texts)
            index = faiss.IndexFlatIP(chunk_emb.shape[1])
            index.add(chunk_emb)
            np.save(emb_fp, chunk_emb)
            faiss.write_index(index, str(index_fp))

        return cls(chunks_df=df, embedder=embedder, index=index, chunk_emb=chunk_emb)

    def search_chunks(self, query: str, *, k: int = 80) -> pd.DataFrame:
        q = (query or "").strip()
        if not q:
            return pd.DataFrame(columns=["score", "web_id", "title", "url", "text"])
        qv = self._embed_query(self.embedder, q).reshape(1, -1)
        scores, idxs = self.index.search(qv, int(k))
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        rows = []
        for i, s in zip(idxs, scores):
            if i < 0:
                continue
            r = self.chunks_df.iloc[int(i)]
            rows.append({
                "score": float(s),
                "web_id": int(r["web_id"]) if str(r["web_id"]).isdigit() else r["web_id"],
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "text": r.get("text", ""),
            })
        return pd.DataFrame(rows)

    def search_docs_dense(self, query: str, *, k_chunks: int = 80, k_docs: int = 20) -> pd.DataFrame:
        """
        Aggregate chunk hits into doc-level ranking (web_id).
        score = max chunk score per doc; preview = top chunk text snippet.
        """
        ch = self.search_chunks(query, k=int(k_chunks))
        if ch is None or len(ch) == 0:
            return pd.DataFrame(columns=["score", "web_id", "title", "url", "preview"])

        ch = ch.copy()
        ch["score"] = pd.to_numeric(ch["score"], errors="coerce").fillna(0.0)
        ch = ch.sort_values("score", ascending=False)

        # pick best chunk per web_id
        best = ch.groupby("web_id", as_index=False).head(1).copy()
        best["preview"] = best["text"].astype(str).apply(_build_preview)
        docs = best[["score", "web_id", "title", "url", "preview"]].head(int(k_docs)).reset_index(drop=True)
        return docs
