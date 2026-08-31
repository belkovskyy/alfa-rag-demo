import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_chunks_parquet import clean_text, split_doc


def test_clean_text_collapses_whitespace_and_nbsp():
    assert clean_text("  а\n\nб\tв г  ") == "а б в г"


def test_clean_text_on_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_split_doc_respects_chunk_size():
    doc = ". ".join(f"предложение номер {i}" for i in range(200))
    parts = split_doc(doc, chunk_size=300, overlap=50)
    assert len(parts) > 1
    assert all(len(p) <= 300 for p in parts)


def test_split_doc_keeps_short_document_whole():
    assert split_doc("короткий текст", chunk_size=900, overlap=150) == ["короткий текст"]


def test_split_doc_on_empty():
    assert split_doc("", chunk_size=900, overlap=150) == []
