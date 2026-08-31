"""Метрики из README: retrieval на ok-подмножестве и accuracy decision layer."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from alfa_rag.config import E5_MODEL, RERANK_MODEL
from alfa_rag.eval import _parse_ids, build_gold_runs, mrr_at_k, recall_at_k
from alfa_rag.rerank import Reranker
from alfa_rag.retrieval import Retriever, normalize_results
from alfa_rag.service import Pipeline

KS = (1, 3, 5, 10, 20)
BOOTSTRAP = 2000
SEED = 42


def mean_ci(values: list[float], n_boot: int = BOOTSTRAP) -> tuple[float, float, float]:
    """Среднее и 95% доверительный интервал по бутстрапу над запросами."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype="float64")
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def retrieval_scores(retriever: Retriever, gold: pd.DataFrame, *, reranker=None,
                     k_chunks: int, k_docs: int) -> dict[str, list[float]]:
    ok = gold[gold["label_status"] == "ok"]
    acc = {f"{m}@{k}": [] for k in KS for m in ("recall", "mrr")}

    for _, r in ok.iterrows():
        res = retriever.search_docs_dense(str(r["query"]), k_chunks=k_chunks, k_docs=k_docs)
        res = normalize_results(res)
        if reranker is not None:
            res = reranker.rerank(str(r["query"]), res)
            sort_col = "rerank_score" if "rerank_score" in res.columns else "score"
        else:
            sort_col = "score"
        res = res.sort_values(sort_col, ascending=False)

        pred = [int(x) for x in res["web_id"].tolist() if str(x).isdigit()]
        for k in KS:
            acc[f"recall@{k}"].append(recall_at_k(r["gold_ids"], pred, k))
            acc[f"mrr@{k}"].append(mrr_at_k(r["gold_ids"], pred, k))

    return acc


def print_markdown_table(rows: dict[str, dict[str, list[float]]], cols: list[str]) -> None:
    print("| режим | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, vals in rows.items():
        cells = []
        for c in cols:
            m, lo, hi = mean_ci(vals[c])
            cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"| {name} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the metrics reported in README")
    ap.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    ap.add_argument("--chunks", default=os.getenv("CHUNKS_PATH", "demo_chunks.parquet"))
    ap.add_argument("--gold", default=os.getenv("GOLD_PATH", "data/gold_labels_sample.csv"))
    ap.add_argument("--k-chunks", type=int, default=80)
    ap.add_argument("--k-docs", type=int, default=40)
    ap.add_argument("--device", default=None, help="cpu или cuda; по умолчанию cuda, если доступна")
    ap.add_argument("--with-rerank", action="store_true")
    ap.add_argument("--dump-errors", metavar="CSV",
                    help="Save misclassified queries for manual error analysis")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold)
    gold["gold_ids"] = gold.get("gold_web_ids", "").apply(_parse_ids)

    retriever = Retriever.from_data_dir(args.data_dir, chunks_path=args.chunks,
                                        embed_model=E5_MODEL, device=args.device)

    corpus_ids = set(retriever.chunks_df["web_id"].astype(int))
    labelled = gold[gold["label_status"] == "ok"]["gold_ids"]
    covered = sum(1 for ids in labelled if set(ids) & corpus_ids)
    if not len(labelled) or covered / len(labelled) < 0.5:
        print(f"\nРазметка {args.gold} почти не пересекается с корпусом {args.chunks}: "
              f"документы найдены только для {covered} из {len(labelled)} размеченных запросов.\n"
              "Демо-корпус и разметка хакатона не связаны между собой — чтобы воспроизвести "
              "числа из README, нужны данные хакатона.")
        return

    n_ok = int((gold["label_status"] == "ok").sum())
    print(f"\nRetrieval, {n_ok} запросов с меткой ok\n")

    rows = {"dense": retrieval_scores(retriever, gold, k_chunks=args.k_chunks, k_docs=args.k_docs)}
    if args.with_rerank:
        rows["dense + CrossEncoder"] = retrieval_scores(
            retriever, gold, reranker=Reranker(RERANK_MODEL, device=args.device or "cpu"),
            k_chunks=args.k_chunks, k_docs=args.k_docs,
        )
    print_markdown_table(rows, [f"recall@{k}" for k in (1, 3, 5)] + [f"mrr@{k}" for k in (1, 3, 5, 10)])

    print(f"\nDecision layer, {len(gold)} запросов\n")
    pipeline = Pipeline(retriever=retriever)
    runs = build_gold_runs(pipeline, gold_path=args.gold, k_docs=20,
                           k_chunks=args.k_chunks, save_parquet=None)

    # то, что система реально отдаёт: выдача после guardrails, а не сырой dense
    ok_runs = runs[runs["gold_status"] == "ok"]
    served = {}
    for k in KS:
        served[f"recall@{k}"] = [recall_at_k(g, p, k)
                                 for g, p in zip(ok_runs["gold_ids"], ok_runs["pred_ids"])]
        served[f"mrr@{k}"] = [mrr_at_k(g, p, k)
                              for g, p in zip(ok_runs["gold_ids"], ok_runs["pred_ids"])]
    print_markdown_table({"полный пайплайн (guardrails)": served},
                         [f"recall@{k}" for k in (1, 3, 5)] + [f"mrr@{k}" for k in (1, 3, 5, 10)])
    print()

    labels = ["ok", "need_clarify", "no_answer"]
    print(classification_report(runs["gold_status"], runs["pred_status"],
                                labels=labels, digits=3, zero_division=0))
    print("confusion matrix (строки — gold, столбцы — pred), порядок:", labels)
    print(confusion_matrix(runs["gold_status"], runs["pred_status"], labels=labels))

    if args.dump_errors:
        bad = runs[runs["gold_status"] != runs["pred_status"]]
        bad.to_csv(args.dump_errors, index=False)
        print(f"\nОшибки сохранены: {args.dump_errors} ({len(bad)} строк)")

    majority = runs["gold_status"].value_counts(normalize=True).iloc[0]
    hits = (runs["gold_status"] == runs["pred_status"]).astype(float).tolist()
    accuracy, lo, hi = mean_ci(hits)
    print(f"\naccuracy правил: {accuracy:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"baseline «всегда самый частый класс»: {majority:.3f}")
    print(f"прирост над baseline: {accuracy - majority:+.3f}")
    print(f"\nИнтервалы — бутстрап по запросам, {BOOTSTRAP} итераций, 95%.")


if __name__ == "__main__":
    main()
