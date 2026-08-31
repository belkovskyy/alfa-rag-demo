from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .service import Pipeline


def _parse_ids(x) -> List[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    return [int(t) for t in re.split(r"[,\s]+", s) if t.strip().isdigit()]


def recall_at_k(gold: List[int], pred: List[int], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(pred[:k])
    return 1.0 if any(g in top for g in gold) else 0.0


def mrr_at_k(gold: List[int], pred: List[int], k: int) -> float:
    if not gold:
        return float("nan")
    top = pred[:k]
    for i, p in enumerate(top, 1):
        if p in set(gold):
            return 1.0 / i
    return 0.0


def _jsonify(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple, set)):
        return json.dumps(list(v) if isinstance(v, set) else v, ensure_ascii=False)
    return v


def build_gold_runs(
    pipeline: Pipeline,
    gold_path: str = "gold_labels.csv",
    *,
    k_docs: int = 20,
    k_chunks: int = 80,
    save_parquet: Optional[str] = "gold_runs.parquet",
) -> pd.DataFrame:
    gold = pd.read_csv(Path(gold_path))
    gold["gold_ids"] = gold.get("gold_web_ids", "").apply(_parse_ids)

    runs = []
    for _, r in gold.iterrows():
        query = str(r["query"])
        out = pipeline.safe_search_docs(query, k_docs=k_docs, k_chunks=k_chunks)

        res = out.get("results")
        pred_ids: List[int] = []
        top_score = 0.0
        if isinstance(res, pd.DataFrame) and len(res):
            tmp = res.copy()
            tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce").fillna(0.0)
            tmp = tmp.sort_values("score", ascending=False)
            pred_ids = [int(x) for x in tmp["web_id"].head(k_docs).tolist() if str(x).isdigit()]
            top_score = float(tmp["score"].iloc[0])

        meta = out.get("meta") or {}
        meta.setdefault("top_score", top_score)

        runs.append({
            "q_id": int(r["q_id"]),
            "query": query,
            "gold_status": r["label_status"],
            "pred_status": out.get("status"),
            "gold_ids": r["gold_ids"],
            "pred_ids": pred_ids,
            "message": out.get("message"),
            **{f"meta_{k}": _jsonify(v) for k, v in meta.items()},
        })

    df = pd.DataFrame(runs)
    if save_parquet:
        df.to_parquet(save_parquet, index=False)
    return df


def eval_runs(
    runs_df: pd.DataFrame,
    *,
    out_errors_csv: Optional[str] = None,
) -> Dict[str, Any]:
    y_true = runs_df["gold_status"].astype(str)
    y_pred = runs_df["pred_status"].astype(str)

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["ok", "need_clarify", "no_answer"])

    # retrieval metrics on ok subset
    ok = runs_df[runs_df["gold_status"] == "ok"].copy()
    recalls = {1: [], 3: [], 5: [], 10: [], 20: []}
    mrrs = {1: [], 3: [], 5: [], 10: [], 20: []}

    for _, r in ok.iterrows():
        gold_ids = list(r["gold_ids"] or [])
        pred_ids = list(r["pred_ids"] or [])
        for k in [1, 3, 5, 10, 20]:
            recalls[k].append(recall_at_k(gold_ids, pred_ids, k))
            mrrs[k].append(mrr_at_k(gold_ids, pred_ids, k))

    retrieval = {f"recall@{k}": float(np.nanmean(recalls[k])) for k in recalls}
    retrieval.update({f"mrr@{k}": float(np.nanmean(mrrs[k])) for k in mrrs})

    if out_errors_csv:
        bad = runs_df[runs_df["gold_status"] != runs_df["pred_status"]].copy()
        bad.to_csv(out_errors_csv, index=False)

    return {
        "report": rep,
        "confusion_matrix": cm.tolist(),
        "retrieval_ok": retrieval,
    }
