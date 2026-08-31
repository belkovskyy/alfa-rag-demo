from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd

ALLOWED_STATUS = {"ok", "need_clarify", "no_answer"}

CLARIFY_CODES = {
    "1": "how_to_check",
    "2": "how_to_do",
    "3": "ask_product",
    "4": "ask_channel",
    "5": "ask_details",
    "6": "route_support",
    "7": "docs_policy",
    "8": "security_urgent",
}


def _backup_csv(path: Path):
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def _parse_ranks(s: str, n: int) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    ranks: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                lo, hi = sorted([int(a), int(b)])
                ranks.extend(list(range(lo, hi + 1)))
        elif p.isdigit():
            ranks.append(int(p))
    return sorted({r for r in ranks if 1 <= r <= n})


def _pick_clarify_codes() -> str:
    print("\nclarify codes:")
    for k, v in CLARIFY_CODES.items():
        print(f"  {k}) {v}")
    s = input("choose codes (e.g. 3,5,6 or ask_product,ask_details) or empty: ").strip().lower()
    if not s:
        return ""
    parts = re.split(r"[,\s]+", s)
    codes = []
    for p in parts:
        if p in CLARIFY_CODES:
            codes.append(CLARIFY_CODES[p])
        elif p in set(CLARIFY_CODES.values()):
            codes.append(p)
    return "|".join(sorted(set(codes)))


def label_eval_set_v2(
    eval_df: pd.DataFrame,
    pipeline,
    *,
    out_csv: str = "gold_labels.csv",
    k_chunks: int = 200,
    k_docs_label: int = 20,
    auto_add_web_id: Optional[int] = None,
):
    """
    Interactive labeling for gold_labels.csv.

    - ok: choose relevant ranks -> gold_web_ids
    - need_clarify: only clarify_note (faster)
    - auto_add_web_id: if this web_id appears in shown top, append it to ok gold_web_ids
    """
    out_path = Path(out_csv)
    done_ids = set()

    if out_path.exists():
        done = pd.read_csv(out_path)
        for col in ["gold_web_ids", "clarify_note"]:
            if col not in done.columns:
                done[col] = ""
                _backup_csv(out_path)
                done.to_csv(out_path, index=False)
        done_ids = set(done["q_id"].astype(int))
        print(f"resume: already labeled {len(done_ids)} queries")
    else:
        print("start: no existing labels")

    i = 0
    while i < len(eval_df):
        row = eval_df.iloc[i]
        q_id = int(row["q_id"])
        q = str(row["query"])

        if q_id in done_ids:
            i += 1
            continue

        print("\n" + "=" * 110)
        print(f"[{i+1}/{len(eval_df)}] q_id={q_id} | {q}")

        resp = pipeline.safe_search_docs(q, k_chunks=k_chunks, k_docs=k_docs_label)
        print("model:", resp["status"], "—", resp["message"])

        res = resp["results"].copy()
        if res is not None and len(res):
            res = res.reset_index(drop=True)
            res["rank"] = res.index + 1
            view = res[["rank", "score", "web_id", "title", "url", "preview"]].head(k_docs_label)
            print(view.to_string(index=False))
        else:
            view = pd.DataFrame(columns=["rank", "score", "web_id", "title", "url", "preview"])
            print("(empty results)")

        while True:
            cmd = input("label status [ok/need_clarify/no_answer] (enter=model, u=undo, s=skip, q=quit): ").strip().lower()

            if cmd in ("q", "quit"):
                print("Stopped. Saved:", out_path)
                return

            if cmd in ("s", "skip"):
                print("Skipped q_id", q_id)
                i += 1
                break

            if cmd in ("u", "undo"):
                if out_path.exists():
                    df = pd.read_csv(out_path)
                    if len(df):
                        _backup_csv(out_path)
                        last = df.tail(1)
                        df = df.iloc[:-1]
                        df.to_csv(out_path, index=False)
                        print("Undone last row:\n", last.to_string(index=False))
                        done_ids = set(df["q_id"].astype(int)) if len(df) else set()
                continue

            label_status = cmd if cmd else resp["status"]
            if label_status not in ALLOWED_STATUS:
                print("bad status, try again")
                continue

            gold_ids: List[int] = []
            clarify_note = ""

            if label_status == "ok" and len(view):
                s = input("relevant ranks for OK (e.g. 1,3,5 or 1-3) or empty: ").strip()
                ranks = _parse_ranks(s, n=min(len(view), k_docs_label))
                gold_ids = [int(view.loc[r - 1, "web_id"]) for r in ranks]
                if auto_add_web_id is not None:
                    top_web_ids = set(map(int, view["web_id"].tolist()))
                    if auto_add_web_id in top_web_ids and auto_add_web_id not in gold_ids:
                        gold_ids.append(auto_add_web_id)
                        print(f"auto_add_web_id: добавил web_id={auto_add_web_id}")
                gold_ids = sorted(set(gold_ids))

            if label_status == "need_clarify":
                codes = _pick_clarify_codes()
                free = input("clarify_note_free (one short line) or empty: ").strip()
                clarify_note = codes
                if free:
                    clarify_note = f"{codes}::{free}" if codes else f"::{free}"

            out_row = {
                "q_id": q_id,
                "query": q,
                "label_status": label_status,
                "gold_web_ids": ",".join(map(str, gold_ids)),
                "clarify_note": clarify_note,
            }
            first_write = not out_path.exists()
            pd.DataFrame([out_row]).to_csv(out_path, mode="a", header=first_write, index=False)

            done_ids.add(q_id)
            i += 1
            break

    print(f"\nDONE. saved to {out_path}")
