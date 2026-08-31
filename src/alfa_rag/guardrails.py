from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Shared schema helper (defined in retrieval.py)
from .retrieval import normalize_results

# -------------------------
# Конфиги
# -------------------------
BAD_TITLE = {"", "без названия", "untitled"}
BAD_PREVIEW_PATTERNS = ["подтвердите запрос", "captcha", "введите код", "авторизац"]
BAD_URL_PATTERNS = ["private.auth"]

PERSONAL_WEAK_MARKERS = ["у меня", "мне", "мой", "моя", "моё", "моем", "моём"]
PERSONAL_STRONG_MARKERS = [
    "прошу", "договор", "заявлен", "претенз",
    "не начисли", "не приш", "списал", "верните", "остал", "закрыл",
    "оспор", "чарджбек", "возврат", "жалоб"
]

LOGIN_TROUBLE_STRICT = [
    "не получается войти", "не могу войти", "не удается войти", "не удаётся войти",
    "не входит", "не пускает", "ошибка входа",
    "не приходит код", "код не приходит", "смс не приходит", "sms не приходит",
    "заблокирован", "заблокировали"
]

APPEAL_MARKERS = ["обращен", "обращение", "ответ по обращ", "статус обращ", "заявк", "статус заяв", "претенз", "жалоб"]
APPEAL_PERSONAL_HINTS = ["статус", "ответ", "по обращ", "по заяв", "номер", "мой", "моя", "моё", "мне", "у меня"]

NUMBER_PERSONAL_CONTEXT = ["страх", "полис", "сертификат", "договор", "заявк", "кредит", "ипотек", "карта", "счет", "счёт"]

NOTIF_TROUBLE_STRICT = [
    "не приходит уведом", "не приходят уведом", "уведомления нет", "уведомлений нет",
    "не приходят пуш", "не приходит пуш", "пуш не приходит", "push не приходит",
    "не показываются уведомления", "нет пушей",
    "смс вместо пуш", "только смс", "только sms",
    "в уведомлениях нет", "уведомлениях нет",
]

VAGUE_MARKERS = ["там", "это", "вот", "оно", "такое", "то есть", "по идее"]

STOPWORDS_RU = {
    "что", "как", "где", "когда", "почему", "зачем",
    "это", "вот", "там", "ли", "или",
    "услуга", "услуги", "вопрос", "ответ",
    "откуда", "взялась", "появилась", "можно", "нужно",
    "то", "есть", "по", "идее", "вообще", "просто"
}

QUESTION_WORDS = {"что", "как", "где", "когда", "почему", "зачем", "можно", "нужно", "сколько", "куда"}

BRAND_WORDS = {"альфа", "альфабанк", "alfabank", "альфа-банк"}
SHORT_IMPORTANT = {"жкх", "ип", "ооо", "инн", "кпп", "мсс", "sms", "смс", "пуш", "push", "код", "пин", "pin", "чек", "вк", "лк"}
TOO_GENERIC = {"доход", "доходы", "деньги", "финансы", "банк", "банка", "банке", "прибыль"}

# Маркеры продуктов / проблем
PRODUCT_MARKERS = ["карта", "кредит", "ипотек", "вклад", "счет", "счёт", "договор", "полис", "заявк", "заказ"]
PROBLEM_MARKERS = ["не приш", "не начис", "спис", "верн", "ошиб", "долг", "задолж", "просроч", "не работает", "не открыва", "не отображ", "не показывает", "пропал", "исчез"]

# ВАЖНО: процесс/статус (персонально) отдельно от времени (часто FAQ)
PROCESS_MARKERS = ["статус", "на каком шаге", "этап", "завис", "зависло", "готовы документы", "готовы ли документы"]
TIME_WORDS = ["когда", "через сколько", "сколько ждать", "сегодня", "завтра", "вчера"]

# past tense actions (персональные кейсы)
ACTION_MARKERS = [
    "подал", "подала", "оформил", "оформила", "оформлял", "оформляла",
    "заказал", "заказывал", "заказывала",
    "отправил", "отправила", "создал", "создала",
    "закрыл", "закрыла", "закрывал", "закрывала",
    "подключил", "подключила", "подключал", "подключала",
]
FAMILY_MARKERS = ["муж", "жена", "сын", "дочь", "мама", "папа"]

CALLCENTER_MARKERS = ["консульт", "консультац", "звонок", "позвон", "телефон", "горяч", "колл", "call", "контакт", "оператор"]

# идентификаторы кейса (цифры + контекст)
ID_CONTEXT_MARKERS = ["номер", "заявк", "договор", "обращ", "претенз", "полис", "счет", "счёт", "карта"]

# Документ, который поднимается в выдачу для запросов про звонок в поддержку.
# На данных хакатона это была страница контактов, web_id=862. Пусто — выключено.
_pin = os.getenv("CALLCENTER_PIN_WEB_ID", "").strip()
CALLCENTER_PIN_WEB_ID: Optional[int] = int(_pin) if _pin.isdigit() else None

def _contains_any(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(p in t for p in patterns)

@lru_cache(maxsize=None)
def _marker_re(markers: Tuple[str, ...]):
    return re.compile(r"\b(?:" + "|".join(re.escape(m) for m in markers) + ")")

def _has_marker(text: str, markers: List[str]) -> bool:
    # маркер должен начинаться с начала слова: "счет" не ловится в "насчет"
    return bool(_marker_re(tuple(markers)).search((text or "").lower()))

def _has_question_word(q_low: str) -> bool:
    toks = re.findall(r"[a-zа-яё0-9-]+", q_low)
    return any(t in QUESTION_WORDS for t in toks)

def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    out["title"] = out["title"].fillna("").astype(str)
    out["preview"] = out["preview"].fillna("").astype(str)
    out["url"] = out["url"].fillna("").astype(str)

    out["title_norm"] = out["title"].str.strip().str.lower()
    mask_bad_title = out["title_norm"].isin(BAD_TITLE)

    preview_norm = out["preview"].str.lower()
    mask_bad_preview = preview_norm.apply(lambda x: _contains_any(x, BAD_PREVIEW_PATTERNS))

    url_norm = out["url"].str.lower()
    mask_bad_url = url_norm.apply(lambda x: _contains_any(x, BAD_URL_PATTERNS))

    out = out[~(mask_bad_title | mask_bad_preview | mask_bad_url)].copy()
    return out.drop(columns=["title_norm"], errors="ignore")

def extract_keywords(query: str) -> List[str]:
    q = (query or "").lower()
    toks = re.findall(r"[a-zа-яё0-9-]+", q)
    out: List[str] = []
    for t in toks:
        t = t.strip("-")
        if not t:
            continue
        if t in BRAND_WORDS:
            continue
        if t in STOPWORDS_RU:
            continue
        if len(t) >= 5 or t in SHORT_IMPORTANT:
            out.append(t)
    return sorted(set(out))

def _variants_for_key(k: str) -> Set[str]:
    kk = k.lower()
    variants = {kk}
    if kk.startswith("сообщ") or kk.startswith("уведом"):
        variants |= {"сообщ", "сообщени", "уведом", "уведомлен", "push", "пуш", "sms", "смс", "альфа-чек", "альфа чек"}
    if kk.startswith("кэшб") or kk.startswith("кешб") or "cashback" in kk:
        variants |= {"кэшб", "кешб", "cashback"}
    if kk.startswith("автоплат"):
        variants |= {"автоплат", "автоплатеж", "автоплатёж", "автооплат"}
    return variants

def keyword_overlap_max(query: str, df: pd.DataFrame, topn: int = 5) -> float:
    keys = extract_keywords(query)
    if not keys or df is None or len(df) == 0:
        return 0.0

    dfx = df.sort_values("score", ascending=False).head(min(topn, len(df)))
    best = 0.0
    for _, r in dfx.iterrows():
        blob = (str(r.get("title", "")) + " " + str(r.get("preview", ""))).lower()
        matched = 0
        for k in keys:
            if any(v in blob for v in _variants_for_key(k)):
                matched += 1
        best = max(best, matched / len(keys))
    return float(best)

def classify_intent_first(query: str) -> str:
    """
    Близко к первому intent, но без "когда" как автоматического personal.
    """
    q = (query or "").lower().strip()
    keys = extract_keywords(q)
    words = q.split()

    if _has_marker(q, LOGIN_TROUBLE_STRICT):
        return "personal"
    if (("войти" in q) or ("личный кабинет" in q) or ("в лк" in q) or ("вход" in q)) and (("не" in q) or ("ошибк" in q)):
        return "personal"

    # process markers: почти всегда персональный кейс, если рядом продукт/заявка/счет
    if _has_marker(q, PROCESS_MARKERS) and (_has_marker(q, PRODUCT_MARKERS) or "заяв" in q or "счет" in q or "счёт" in q):
        return "personal"

    # actions: past tense + продукт/заявка
    if _has_marker(q, ACTION_MARKERS) and (_has_marker(q, PRODUCT_MARKERS) or "заяв" in q or "счет" in q or "счёт" in q):
        return "personal"

    # family + продукт
    if _has_marker(q, FAMILY_MARKERS) and _has_marker(q, PRODUCT_MARKERS):
        return "personal"

    if len(words) <= 3:
        toks = set(re.findall(r"[a-zа-яё0-9-]+", q))
        toks = {t for t in toks if t not in STOPWORDS_RU}
        if toks and toks.issubset(TOO_GENERIC):
            return "vague"

    if _has_marker(q, NOTIF_TROUBLE_STRICT):
        return "personal"
    has_notif = ("уведом" in q) or ("пуш" in q) or ("push" in q)
    has_sms = ("смс" in q) or ("sms" in q)
    has_neg = ("нет" in q) or ("не приход" in q) or ("не показыва" in q)
    has_contrast = ("но" in q) or ("вместо" in q) or ("только" in q)
    if (has_notif and has_neg) or (has_notif and has_sms and has_contrast):
        return "personal"

    if _has_marker(q, APPEAL_MARKERS) and (_has_marker(q, APPEAL_PERSONAL_HINTS) or re.search(r"\d{3,}", q)):
        return "personal"

    if "номер" in q and _has_marker(q, NUMBER_PERSONAL_CONTEXT):
        return "personal"

    if _has_marker(q, PERSONAL_STRONG_MARKERS):
        return "personal"

    personal_context = ["спис", "начис", "верн", "деньг", "счет", "счёт", "операц", "платеж", "платёж", "договор", "кредит", "ипотек"]
    if _has_marker(q, PERSONAL_WEAK_MARKERS) and _has_marker(q, personal_context):
        return "personal"

    if len(words) <= 3 and len(keys) == 0:
        return "vague"
    if _has_marker(q, VAGUE_MARKERS) and len(keys) <= 1:
        return "vague"

    return "faq"

def _pin_web_id(df: pd.DataFrame, web_id: int) -> Tuple[pd.DataFrame, bool]:
    if df is None or df.empty:
        return df, False
    if "web_id" not in df.columns:
        return df, False
    m = df["web_id"].astype(str) == str(web_id)
    if not m.any():
        return df, False
    top = df[m].copy()
    rest = df[~m].copy()
    return pd.concat([top, rest], ignore_index=True), True

def guardrails_features(
    query: str,
    ranked_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    *,
    retrieval_error: Optional[str] = None,
    k_docs: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    ranked_df -> для results (после filter_noise/pin)
    dense_df  -> для стабильных мета-фич (top_score/overlap)
    """
    q = (query or "").strip()
    q_low = q.lower()

    ranked_raw = normalize_results(ranked_df)
    ranked_filtered = normalize_results(filter_noise(ranked_raw)) if len(ranked_raw) else ranked_raw
    base_df = ranked_filtered if len(ranked_filtered) else ranked_raw

    # DENSE база для мета (стабильная)
    dense_base = normalize_results(dense_df).sort_values("score", ascending=False) if len(dense_df) else normalize_results(pd.DataFrame())

    keys = extract_keywords(q)
    word_count = len(re.findall(r"[a-zа-яё0-9-]+", q_low))
    has_question_mark = ("?" in q)
    has_question_word = _has_question_word(q_low)

    has_process_markers = _has_marker(q_low, PROCESS_MARKERS)
    has_time_words = _has_marker(q_low, TIME_WORDS)
    has_action_markers = _has_marker(q_low, ACTION_MARKERS)
    has_family_markers = _has_marker(q_low, FAMILY_MARKERS)

    has_product_markers = _has_marker(q_low, PRODUCT_MARKERS)
    has_problem_markers = _has_marker(q_low, PROBLEM_MARKERS)
    has_digits = bool(re.search(r"\d{3,}", q_low))
    has_personal_id_context = bool(has_digits and _has_marker(q_low, ID_CONTEXT_MARKERS))

    is_callcenter = _has_marker(q_low, CALLCENTER_MARKERS)

    intent = classify_intent_first(q)

    # ВАЖНО: top_score/overlap считаем по DENSE-top5, а не по rerank-упорядоченному base_df
    dense_top_score = float(dense_base["score"].iloc[0]) if len(dense_base) else 0.0
    dense_overlap = keyword_overlap_max(q, dense_base, topn=5) if len(dense_base) else 0.0

    pinned_doc = False
    if is_callcenter and CALLCENTER_PIN_WEB_ID is not None:
        base_df, pinned_doc = _pin_web_id(base_df, CALLCENTER_PIN_WEB_ID)

    needs_context = (word_count <= 3) and (not has_question_mark) and (not has_question_word)
    underspecified = (
        needs_context
        or ((word_count <= 4) and (len(keys) <= 1))
        or (has_problem_markers and not has_product_markers)
    )

    meta = {
        "intent": intent,
        "keys": keys,
        "n_keys": int(len(keys)),
        "word_count": int(word_count),
        "has_question_mark": bool(has_question_mark),
        "has_question_word": bool(has_question_word),

        # используем DENSE значения
        "top_score": float(dense_top_score),
        "overlap": float(dense_overlap),

        "n_raw": int(len(ranked_raw)),
        "n_filtered": int(len(ranked_filtered)),
        "n_base": int(len(base_df)),

        "has_product_markers": bool(has_product_markers),
        "has_problem_markers": bool(has_problem_markers),

        "has_process_markers": bool(has_process_markers),
        "has_time_words": bool(has_time_words),
        "has_action_markers": bool(has_action_markers),
        "has_family_markers": bool(has_family_markers),

        "has_digits": bool(has_digits),
        "has_personal_id_context": bool(has_personal_id_context),

        "needs_context": bool(needs_context),
        "underspecified": bool(underspecified),

        "is_callcenter": bool(is_callcenter),
        "pinned_doc": bool(pinned_doc),
        "retrieval_error": retrieval_error,
    }

    return base_df.head(k_docs), meta

