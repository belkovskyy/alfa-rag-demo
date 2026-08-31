from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _parse_clarify_note(note: str) -> Dict[str, Any]:
    """
    note format:
      "ask_details|ask_product::Уточните ..."
      or just "Уточните ..."
    """
    s = (note or "").strip()
    if not s:
        return {"codes": [], "text": ""}

    if "::" in s:
        left, right = s.split("::", 1)
        codes = [c.strip() for c in left.split("|") if c.strip()]
        text = right.strip()
        return {"codes": codes, "text": text}

    return {"codes": [], "text": s}


@dataclass
class ClarifyBank:
    E: np.ndarray
    queries: List[str]
    note_texts: List[str]
    note_codes: List[List[str]]


def build_clarify_bank(gold_path: str | Path, embed_fn) -> ClarifyBank:
    gold = pd.read_csv(Path(gold_path))
    df = gold[(gold["label_status"] == "need_clarify") & gold["clarify_note"].notna()].copy()

    df["query"] = df["query"].astype(str)
    df["clarify_note"] = df["clarify_note"].astype(str)

    parsed = df["clarify_note"].apply(_parse_clarify_note)
    df["note_codes"] = parsed.apply(lambda x: x["codes"])
    df["note_text"]  = parsed.apply(lambda x: x["text"])

    texts = df["query"].tolist()
    E = np.asarray(embed_fn(texts), dtype="float32")

    return ClarifyBank(
        E=E,
        queries=texts,
        note_texts=df["note_text"].tolist(),
        note_codes=df["note_codes"].tolist(),
    )


def pick_clarify_hint_topk(
    query: str,
    bank: ClarifyBank,
    embed_fn,
    *,
    topk: int = 5,
    min_sim: float = 0.70,
    max_examples: int = 3,
) -> Optional[Dict[str, Any]]:
    qE = np.asarray(embed_fn([query]), dtype="float32")[0]
    sims = bank.E @ qE

    idx = np.argsort(-sims)[:max(topk, 1)]
    picked = []
    for i in idx:
        sim = float(sims[i])
        if sim < min_sim:
            continue
        picked.append((int(i), sim))
    if not picked:
        return None

    best_i, best_sim = picked[0]
    best_text = bank.note_texts[best_i]
    best_query = bank.queries[best_i]

    codes: List[str] = []
    for i, _ in picked:
        for c in bank.note_codes[i]:
            if c and c not in codes:
                codes.append(c)

    examples = []
    for i, sim in picked[:max_examples]:
        examples.append({"sim": sim, "matched_query": bank.queries[i]})

    return {
        "codes": codes[:8],
        "examples": examples,
        "top_sim": float(best_sim),
        "best_text": best_text,
        "best_query": best_query,
    }


# --- tag derivation (lightweight, no LLM) ---

MONEY_MISSING_PATTERNS = [
    "где мои деньги", "куда делись деньги", "куда ушли деньги", "пропали деньги",
    "не вижу деньги", "исчезли деньги", "деньги пропали",
    "не пришли деньги", "не поступили деньги",
]
APP_RATE_PATTERNS = [
    "оценить приложение", "приложение оценить", "оценка приложения", "поставить оценку",
    "оставить отзыв", "отзыв о приложении", "рейтинг приложения", "звезды", "звёзды"
]
GOS_NOTIF_PATTERNS = ["госуведомления", "гос уведомления", "гос-уведомления"]

CODE_GUIDE = {
    "ask_what_happened": "что именно произошло (не поступило / списалось / не видно / в обработке)",
    "ask_what_exactly":  "что именно хотите оценить (приложение в целом / конкретную функцию / отзыв в магазине приложений)",
    "ask_product":  "какой продукт/сервис (карта/счёт/вклад/кредит/инвестиции/уведомления и т.д.)",
    "ask_channel":  "где это происходит (приложение/сайт/чат/банкомат) и если про уведомления — какой канал (push/sms/почта)",
    "ask_time":     "когда это случилось (дата/примерное время)",
    "ask_amount":   "на какую сумму (примерно) и в какой валюте",
    "ask_details":  "каких деталей не хватает, чтобы понять вопрос",
    "how_to_do":    "что именно хотите сделать (подключить/отключить/изменить/оформить)",
    "how_to_check": "что и где проверяете (какой экран/раздел/статус)",
    "route_support":"если это персональный случай — уточнить контекст и при необходимости направить в поддержку",
}

# Concrete question templates used to generate concise clarifications.
# These are intentionally short, safe (no PII), and map 1:1 to clarify codes.
CODE_QUESTIONS = {
    "ask_what_happened": "Что именно произошло (не поступило / списалось / не отображается / ошибка)?",
    "ask_what_exactly":  "Что именно вы хотите сделать или уточнить?",
    "ask_product":       "По какому продукту/сервису вопрос: карта, счёт, вклад, кредит, инвестиции, уведомления?",
    "ask_channel":       "Где это происходит: в приложении, на сайте, в чате или в банкомате? Если про уведомления — это push/SMS/почта?",
    "ask_time":          "Когда это случилось (примерно дата/время)?",
    "ask_amount":        "Какая сумма и валюта (примерно)?",
    "ask_details":       "Опишите, пожалуйста, подробнее: что вы делаете и что получается/не получается?",
    "how_to_do":         "Что именно вы хотите сделать (подключить/отключить/перевести/изменить)?",
    "how_to_check":      "Где именно вы это проверяете (какой раздел/экран в приложении или на сайте)?",
    "route_support":     "Это связано с безопасностью или подозрительными сообщениями/операциями?",
}

def derive_clarify_tags(query: str, meta: Dict[str, Any], hint_codes: List[str]) -> List[str]:
    q = (query or "").lower().strip()
    meta = meta or {}
    tags = set(hint_codes or [])

    if not meta.get("has_product_markers", False):
        tags.add("ask_product")
    if meta.get("needs_context", False) or meta.get("underspecified", False):
        tags.add("ask_details")

    if any(p in q for p in MONEY_MISSING_PATTERNS) or ("где" in q and "деньг" in q):
        tags |= {"ask_what_happened", "ask_product", "ask_channel", "ask_time", "ask_amount"}

    if any(p in q for p in APP_RATE_PATTERNS) or ("оцен" in q and "прилож" in q):
        tags |= {"ask_what_exactly", "ask_channel"}

    if any(p in q for p in GOS_NOTIF_PATTERNS):
        tags |= {"ask_product", "ask_channel", "how_to_do"}

    priority = [
        "ask_what_happened", "ask_what_exactly",
        "ask_product", "ask_channel", "ask_time", "ask_amount",
        "ask_details", "how_to_do", "how_to_check", "route_support"
    ]
    ordered = [t for t in priority if t in tags]
    for t in tags:
        if t not in ordered:
            ordered.append(t)
    return ordered


def fallback_clarify(tags: List[str], *, max_bullets: int = 3) -> str:
    questions = build_clarify_questions(tags, max_questions=max_bullets)
    bullets = ["— " + q for q in questions]
    return "Уточните, пожалуйста, пару деталей, чтобы подсказать точнее?\n" + "\n".join(bullets)


def build_clarify_questions(tags: List[str], *, max_questions: int = 3) -> List[str]:
    """Pick 2–3 concrete clarification questions from tags.

    This is the key improvement vs. generic "ask_product/ask_channel" bullets.
    It ensures the user sees actionable short questions.
    """
    tags = tags or []

    # Prefer asking about product+channel first (most useful), then details.
    priority = [
        "ask_what_happened", "ask_what_exactly",
        "ask_product", "ask_channel",
        "how_to_do", "how_to_check",
        "ask_time", "ask_amount",
        "ask_details",
        "route_support",
    ]

    ordered = []
    for t in priority:
        if t in tags and t not in ordered:
            ordered.append(t)
    for t in tags:
        if t not in ordered:
            ordered.append(t)

    out: List[str] = []
    for t in ordered:
        q = CODE_QUESTIONS.get(t)
        if not q:
            continue
        if q not in out:
            out.append(q)
        if len(out) >= max_questions:
            break

    if not out:
        out = [
            CODE_QUESTIONS["ask_details"],
            CODE_QUESTIONS["ask_product"],
            CODE_QUESTIONS["ask_channel"],
        ][:max_questions]

    return out
