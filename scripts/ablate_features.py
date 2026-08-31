"""Вклад каждого признака guardrails в решение: признаки зануляются по очереди,
правила пересчитываются на тех же meta, retrieval повторно не запускается."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from alfa_rag.config import E5_MODEL
from alfa_rag.decision import decision_policy_rules
from alfa_rag.eval import build_gold_runs
from alfa_rag.retrieval import Retriever
from alfa_rag.service import Pipeline

# булевы признаки и intent — то, что можно занулить осмысленно
BOOL_FEATURES = [
    "has_question_mark", "has_question_word",
    "has_product_markers", "has_problem_markers",
    "has_process_markers", "has_time_words", "has_action_markers",
    "has_family_markers", "has_digits", "has_personal_id_context",
    "needs_context", "underspecified", "is_callcenter", "pinned_doc",
]
NEUTRAL = {"intent": "faq"}


def metas_from_runs(runs: pd.DataFrame) -> list[dict]:
    cols = [c for c in runs.columns if c.startswith("meta_")]
    out = []
    for _, r in runs.iterrows():
        out.append({c[len("meta_"):]: r[c] for c in cols})
    return out


def accuracy(metas: list[dict], gold: list[str], *, drop: str | None = None) -> float:
    hits = 0
    for meta, y in zip(metas, gold):
        m = dict(meta)
        if drop is not None:
            m[drop] = NEUTRAL.get(drop, False)
        status, _ = decision_policy_rules(m)
        hits += int(status == y)
    return hits / len(gold) if gold else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature ablation for the decision layer")
    ap.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    ap.add_argument("--chunks", default=os.getenv("CHUNKS_PATH", "demo_chunks.parquet"))
    ap.add_argument("--gold", default=os.getenv("GOLD_PATH", "data/gold_labels_sample.csv"))
    ap.add_argument("--k-chunks", type=int, default=80)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    retriever = Retriever.from_data_dir(args.data_dir, chunks_path=args.chunks,
                                        embed_model=E5_MODEL, device=args.device)
    runs = build_gold_runs(Pipeline(retriever=retriever), gold_path=args.gold,
                           k_docs=20, k_chunks=args.k_chunks, save_parquet=None)

    metas = metas_from_runs(runs)
    gold = runs["gold_status"].astype(str).tolist()
    base = accuracy(metas, gold)
    print(f"\naccuracy со всеми признаками: {base:.3f}, запросов {len(gold)}\n")

    rows = []
    for feat in ["intent"] + BOOL_FEATURES:
        if feat not in metas[0]:
            continue
        acc = accuracy(metas, gold, drop=feat)
        rows.append({"признак": feat, "accuracy без него": round(acc, 3),
                     "изменение": round(acc - base, 3)})

    df = pd.DataFrame(rows).sort_values("изменение")
    print(df.to_string(index=False))

    dead = df[df["изменение"] == 0.0]["признак"].tolist()
    if dead:
        print(f"\nНа этом наборе ничего не меняют: {', '.join(dead)}")


if __name__ == "__main__":
    main()
