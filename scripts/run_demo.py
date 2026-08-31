from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from alfa_rag.config import E5_MODEL, USE_RERANK, RERANK_MODEL
from alfa_rag.retrieval import Retriever
from alfa_rag.rerank import Reranker
from alfa_rag.service import Pipeline
from alfa_rag.clarify import build_clarify_bank
from alfa_rag.llm import ollama_available


def make_embed_fn(model: SentenceTransformer):
    def embed_fn(texts):
        texts2 = [f"query: {str(t)}" for t in texts]
        vecs = model.encode(texts2, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")
    return embed_fn


def main():
    ap = argparse.ArgumentParser(description="Run a single query through the RAG pipeline")
    ap.add_argument("query")
    ap.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    ap.add_argument("--chunks", default=os.getenv("CHUNKS_PATH", "demo_chunks.parquet"))
    ap.add_argument("--gold", default=os.getenv("GOLD_PATH", "data/gold_labels_sample.csv"))
    ap.add_argument("--no-llm", action="store_true", help="Skip Ollama even if it is running")
    args = ap.parse_args()

    query = args.query
    data_dir = Path(args.data_dir)

    retriever = Retriever.from_data_dir(data_dir, chunks_path=args.chunks, embed_model=E5_MODEL)

    reranker = None
    if USE_RERANK:
        reranker = Reranker(RERANK_MODEL, device="cpu")

    pipeline = Pipeline(retriever=retriever, reranker=reranker)

    # optional clarify bank if a labelled csv is available
    gold_path = Path(args.gold)
    clarify_bank = None
    embed_fn = None
    if gold_path.exists():
        e5 = SentenceTransformer(E5_MODEL)
        embed_fn = make_embed_fn(e5)
        clarify_bank = build_clarify_bank(gold_path, embed_fn=embed_fn)

    use_llm = (not args.no_llm) and ollama_available()
    if not use_llm and not args.no_llm:
        print("Ollama на", os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
              "недоступна — ответ будет без генерации.")

    res = pipeline.ask(
        query,
        clarify_bank=clarify_bank,
        embed_fn=embed_fn,
        use_llm=use_llm,
        llm_for="both",
    )

    print("\n=== QUERY ===")
    print(query)
    print("\n=== STATUS ===")
    print(f"[{res['out']['status']}] {res['final']}")

    print("\n=== TOP DOCS ===")
    df = res["out"]["results"]
    if df is not None and len(df):
        cols = [c for c in ["score","rerank_score","web_id","title","url"] if c in df.columns]
        print(df[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
