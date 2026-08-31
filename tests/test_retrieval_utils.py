import pandas as pd

from alfa_rag.eval import _parse_ids, mrr_at_k, recall_at_k
from alfa_rag.retrieval import normalize_results


def test_normalize_results_on_empty_input():
    out = normalize_results(pd.DataFrame())
    assert list(out.columns) == ["score", "web_id", "title", "url", "preview", "text"]
    assert len(out) == 0


def test_normalize_results_fills_missing_columns():
    out = normalize_results(pd.DataFrame([{"web_id": 1}]))
    assert out.loc[0, "score"] == 0.0
    assert out.loc[0, "title"] == ""


def test_normalize_results_coerces_bad_scores():
    out = normalize_results(pd.DataFrame([{"web_id": 1, "score": "не число"}]))
    assert out.loc[0, "score"] == 0.0


def test_parse_ids_handles_missing_and_mixed():
    assert _parse_ids(None) == []
    assert _parse_ids("nan") == []
    assert _parse_ids("1, 2 3") == [1, 2, 3]


def test_recall_and_mrr():
    assert recall_at_k([5], [1, 5, 9], 1) == 0.0
    assert recall_at_k([5], [1, 5, 9], 3) == 1.0
    assert mrr_at_k([5], [1, 5, 9], 3) == 0.5
    assert mrr_at_k([7], [1, 5, 9], 3) == 0.0
