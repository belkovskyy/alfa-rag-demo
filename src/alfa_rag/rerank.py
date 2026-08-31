from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder


@dataclass
class Reranker:
    model_name: str
    device: str = "cpu"
    max_chars: int = 900

    def __post_init__(self):
        self.model = CrossEncoder(self.model_name, device=self.device)

    def _doc_text(self, row: pd.Series) -> str:
        title = str(row.get("title", "") or "").strip()
        preview = str(row.get("preview", "") or "").strip()
        t = (title + "\n" + preview).strip()
        if len(t) > self.max_chars:
            t = t[: self.max_chars].rstrip() + "…"
        return t

    def rerank(self, query: str, df: pd.DataFrame, *, topn: int = 40, keep: int = 20, batch_size: int = 32) -> pd.DataFrame:
        """
        Rerank topN candidates (by dense score) and return topK.
        NOTE: in this project rerank is **optional** and default is OFF.
        """
        if df is None or len(df) == 0:
            return df

        dfx = df.copy()
        if "score" in dfx.columns:
            dfx["score"] = pd.to_numeric(dfx["score"], errors="coerce").fillna(0.0)

        cand = dfx.sort_values("score", ascending=False).head(int(topn)).copy()
        pairs = [(query, self._doc_text(r)) for _, r in cand.iterrows()]

        scores = self.model.predict(pairs, batch_size=batch_size)
        cand["rerank_score"] = np.asarray(scores, dtype="float32")

        cand = cand.sort_values(["rerank_score", "score"], ascending=[False, False]).reset_index(drop=True)
        return cand.head(int(keep)).reset_index(drop=True)

# Backward-compatible alias used by the Streamlit app
CrossEncoderReranker = Reranker
