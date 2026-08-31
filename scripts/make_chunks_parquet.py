from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(t) -> str:
    return re.sub(r"\s+", " ", str(t or "").replace(" ", " ")).strip()


def split_doc(text: str, *, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text) if text else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a chunks parquet from a documents csv")
    ap.add_argument("--inp", default="data/websites.csv", help="csv with web_id, title, url, text")
    ap.add_argument("--out", default="data/chunks_websites.parquet", help="Output parquet path")
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--no-title", action="store_true",
                    help="Не подставлять заголовок в начало документа перед нарезкой")

    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # required columns in source file
    for c in ["web_id", "title", "url", "text"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {inp}. Columns: {list(df.columns)}")

    rows = []
    for r in df.itertuples(index=False):
        title = clean_text(getattr(r, "title"))
        text = clean_text(getattr(r, "text"))
        doc = text if args.no_title else clean_text(f"{title}\n\n{text}")

        for j, part in enumerate(split_doc(doc, chunk_size=args.chunk_size, overlap=args.overlap)):
            rows.append({
                "web_id": getattr(r, "web_id"),
                "title": title,
                "url": getattr(r, "url"),
                "chunk_no": j,
                "text": part,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out, index=False)
    print(f"Saved: {out} | docs={out_df['web_id'].nunique()} chunks={len(out_df)}")


if __name__ == "__main__":
    main()
