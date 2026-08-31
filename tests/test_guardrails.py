import pandas as pd

from alfa_rag import guardrails
from alfa_rag.guardrails import (
    CALLCENTER_PIN_WEB_ID,
    PERSONAL_WEAK_MARKERS,
    PRODUCT_MARKERS,
    _has_marker,
    classify_intent_first,
    extract_keywords,
    filter_noise,
    guardrails_features,
    keyword_overlap_max,
)


def docs(*rows):
    return pd.DataFrame(list(rows))


def test_markers_match_only_at_word_start():
    assert _has_marker("где реквизиты счета", PRODUCT_MARKERS)
    assert not _has_marker("насчет тарифов", PRODUCT_MARKERS)
    assert not _has_marker("сомневаюсь в расчете", PRODUCT_MARKERS)


def test_weak_personal_marker_is_a_whole_word():
    assert _has_marker("мне не пришёл кэшбэк", PERSONAL_WEAK_MARKERS)
    assert not _has_marker("сомневаюсь в этом", PERSONAL_WEAK_MARKERS)


def test_filter_noise_drops_service_pages():
    df = docs(
        {"web_id": 1, "title": "Кэшбэк", "url": "https://x/help/", "preview": "как начисляется"},
        {"web_id": 2, "title": "", "url": "https://x/a/", "preview": "текст"},
        {"web_id": 3, "title": "Вход", "url": "https://x/a/", "preview": "введите код из смс"},
        {"web_id": 4, "title": "Кабинет", "url": "https://private.auth/x", "preview": "текст"},
    )
    assert filter_noise(df)["web_id"].tolist() == [1]


def test_extract_keywords_drops_stopwords_and_brand():
    keys = extract_keywords("Как в Альфа-Банке подключить кэшбэк?")
    assert "как" not in keys
    assert "альфа" not in keys
    assert "кэшбэк" in keys


def test_extract_keywords_keeps_short_but_meaningful():
    assert "жкх" in extract_keywords("не начислен кэшбэк за жкх")


def test_login_trouble_is_personal():
    assert classify_intent_first("не могу войти в приложение") == "personal"


def test_plain_faq_question_is_not_personal():
    assert classify_intent_first("как подключить кэшбэк") != "personal"


def test_keyword_overlap_rewards_matching_documents():
    df = docs(
        {"score": 0.9, "web_id": 1, "title": "Кэшбэк", "url": "", "preview": "как начисляется кэшбэк"},
        {"score": 0.5, "web_id": 2, "title": "Вклады", "url": "", "preview": "ставки по вкладам"},
    )
    assert keyword_overlap_max("кэшбэк", df, topn=2) == 1.0
    assert keyword_overlap_max("ипотека", df, topn=2) == 0.0


def test_features_expose_the_schema_decision_layer_expects():
    df = docs({"score": 0.9, "web_id": 1, "title": "Кэшбэк", "url": "", "preview": "как начисляется"})
    _, meta = guardrails_features("как начисляется кэшбэк", df, df)
    for key in ["intent", "top_score", "overlap", "n_base", "n_keys", "pinned_doc"]:
        assert key in meta
    assert meta["top_score"] == 0.9


def callcenter_docs():
    # guardrails_features ожидает выдачу, уже отсортированную по score
    return docs(
        {"score": 0.9, "web_id": 1, "title": "Кэшбэк", "url": "", "preview": "начисление"},
        {"score": 0.5, "web_id": 862, "title": "Контакты", "url": "", "preview": "телефон поддержки"},
    )


def test_document_pin_is_off_unless_configured():
    assert CALLCENTER_PIN_WEB_ID is None
    df = callcenter_docs()
    ranked, meta = guardrails_features("как позвонить в поддержку", df, df)
    assert meta["is_callcenter"] is True
    assert meta["pinned_doc"] is False
    assert ranked["web_id"].tolist()[0] == 1


def test_configured_pin_lifts_the_document_for_callcenter_queries(monkeypatch):
    monkeypatch.setattr(guardrails, "CALLCENTER_PIN_WEB_ID", 862)
    df = callcenter_docs()
    ranked, meta = guardrails_features("как позвонить в поддержку", df, df)
    assert meta["pinned_doc"] is True
    assert ranked["web_id"].tolist()[0] == 862


def test_configured_pin_does_not_touch_other_queries(monkeypatch):
    monkeypatch.setattr(guardrails, "CALLCENTER_PIN_WEB_ID", 862)
    df = callcenter_docs()
    ranked, meta = guardrails_features("как начисляется кэшбэк", df, df)
    assert meta["is_callcenter"] is False
    assert meta["pinned_doc"] is False
    assert ranked["web_id"].tolist()[0] == 1


def test_empty_retrieval_does_not_crash():
    empty = pd.DataFrame()
    ranked, meta = guardrails_features("что-нибудь", empty, empty)
    assert meta["n_base"] == 0
    assert meta["top_score"] == 0.0
