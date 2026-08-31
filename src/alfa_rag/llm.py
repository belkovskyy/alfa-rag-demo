from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests

from .config import OLLAMA_MODEL, OLLAMA_URL
from .clarify import build_clarify_questions, derive_clarify_tags, fallback_clarify


DEFAULT_SYSTEM = (
    "Ты аккуратный помощник банка. Не выдумывай факты. "
    "Если данных нет — честно скажи об этом и предложи уточнить."
)

SENSITIVE_LINE_PATTERNS = [
    r"номер\s+(карты|сч[её]та)",
    r"\bcvv\b|\bcvc\b|\bпин\b",
    r"паспорт|снилс|инн|код\s+из\s+смс|одноразов",
    r"логин|парол",
]


def ollama_available(*, timeout: float = 2.0) -> bool:
    try:
        requests.get(OLLAMA_URL, timeout=timeout).raise_for_status()
        return True
    except Exception:
        return False


def ollama_generate(
    prompt: str,
    *,
    model: str = OLLAMA_MODEL,
    system: str = DEFAULT_SYSTEM,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def build_context_from_docs(docs: List[Dict[str, Any]], *, max_docs: int = 6, max_chars: int = 9000) -> str:
    parts = []
    total = 0
    for d in (docs or [])[:max_docs]:
        title = str(d.get("title", "")).strip()
        url = str(d.get("url", "")).strip()
        preview = str(d.get("preview", "")).strip()
        chunk = f"- {title}\n  {url}\n  {preview}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts).strip()


def _to_second_person_ru(t: str) -> str:
    if not t:
        return t
    repl = [
        (r"\bмоим\b", "вашим"),
        (r"\bмоем\b", "вашем"),
        (r"\bмоём\b", "вашем"),
        (r"\bмоей\b", "вашей"),
        (r"\bмоего\b", "вашего"),
        (r"\bмоих\b", "ваших"),
        (r"\bмоему\b", "вашему"),
        (r"\bмоими\b", "вашими"),
        (r"\bмои\b", "ваши"),
        (r"\bмой\b", "ваш"),
        (r"\bмоя\b", "ваша"),
        (r"\bмоё\b", "ваше"),
    ]
    out = t
    for pat, rep in repl:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def _clean_llm_text_need_clarify(txt: str, tags: List[str]) -> str:
    t = (txt or "").strip()
    t = re.sub(r"(?im)^\s*запрос\s*:\s*.*$", "", t).strip()
    t = re.sub(r"(?im)^\s*вопрос\s*:\s*", "", t).strip()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    norm = []
    for ln in lines:
        ln = re.sub(r"^\d+\)\s*", "", ln)
        ln = re.sub(r"^[-•]\s*", "", ln)
        ln = re.sub(r"^—\s*", "", ln)
        ln = ln.strip()
        if not ln:
            continue
        low = ln.lower()
        if "ask_" in low:
            continue
        if any(re.search(p, low) for p in SENSITIVE_LINE_PATTERNS):
            continue
        norm.append(ln)

    if not norm:
        return ""

    qline = norm[0]
    if not qline.endswith("?"):
        qline = qline.rstrip(".") + "?"

    bullets = []
    for b in norm[1:]:
        b = b.strip().rstrip("—- ").strip()
        if not b:
            continue
        bullets.append(b)
    bullets = bullets[:3]

    if not bullets:
        # fallback bullets from tags (concrete questions)
        bullets = build_clarify_questions(tags, max_questions=3)

    out_lines = [qline] + ["— " + b for b in bullets[:3]]
    return _to_second_person_ru("\n".join(out_lines).strip())


def llm_rewrite_need_clarify(
    query: str,
    *,
    hint_codes: List[str],
    meta: Dict[str, Any],
    hint_text: str = "",
    hint_examples: Optional[List[str]] = None,
    model: str = OLLAMA_MODEL,
) -> str:
    tags = derive_clarify_tags(query, meta or {}, hint_codes or [])

    # We want the clarification to be short and actionable. The codes drive a fixed
    # set of 2–3 concrete questions; LLM may only polish wording.
    base_questions = build_clarify_questions(tags, max_questions=3)
    if not base_questions:
        base_questions = build_clarify_questions(["ask_details", "ask_product", "ask_channel"], max_questions=3)

    hint_block = ""
    ht = (hint_text or "").strip()
    if ht:
        # best_text comes from your own gold/clarify bank. We still cap it to avoid prompt bloat.
        ht = ht[:1200]
        hint_block = "\n\nПодсказка из clarify_bank (можно использовать как основу, если подходит):\n" + ht

    examples_block = ""
    if hint_examples:
        ex = hint_examples[:3]
        examples_block = "\n\nПохожие запросы (для ориентира):\n" + "\n".join([f"- {x}" for x in ex])

    must_block = "\n".join([f"- {q}" for q in base_questions])

    prompt = f"""Ты — помощник банка. Сформулируй уточняющий вопрос к запросу пользователя.

ЖЁСТКИЕ правила:
- Пиши по-русски, естественно.
- Всегда обращайся к пользователю на «вы», используй «ваш/ваша/ваше».
- Не придумывай факты/числа.
- Не проси персональные данные (ФИО, номер карты/счёта, телефон, паспорт, коды из СМС).
- Не пиши слова/метки вида "ask_product", "ask_channel" и т.п.

Формат ответа строго:
1) ОДНО короткое предложение-вопрос (заканчивается '?')
2) Затем РОВНО {len(base_questions)} буллета, каждый начинается с "— "

ОБЯЗАТЕЛЬНЫЕ уточняющие вопросы (их смысл надо сохранить; можно слегка перефразировать, но НЕ добавляй новых):
{must_block}

Запрос пользователя: {query}

Доп. контекст (можно использовать, если помогает):
{hint_block}{examples_block}

Сгенерируй уточнение в этом формате.
"""
    try:
        raw = ollama_generate(prompt, model=model, temperature=0.1)
        cleaned = _clean_llm_text_need_clarify(raw, tags)
        if not cleaned:
            return fallback_clarify(tags)
        return cleaned
    except Exception:
        return fallback_clarify(tags)


def normalize_clarify_note(note_text: str, tags: List[str]) -> str:
    """Normalize a human-written clarify note into the same short format we use everywhere.

    This is useful for `clarify_mode="gold"` where we want to show the bank's own
    clarification phrasing instead of the generic rule-based message.
    """
    t = (note_text or "").strip()
    if not t:
        return fallback_clarify(tags)

    # drop obviously sensitive lines
    lines_raw = [ln.strip() for ln in t.splitlines() if ln.strip()]
    safe = []
    for ln in lines_raw:
        low = ln.lower()
        if any(re.search(p, low) for p in SENSITIVE_LINE_PATTERNS):
            continue
        safe.append(ln)
    if not safe:
        return fallback_clarify(tags)

    qline = safe[0].rstrip(".").strip()
    if not qline.endswith("?"):
        qline += "?"

    bullets = []
    for ln in safe[1:]:
        ln = re.sub(r"^[-•—]\s*", "", ln).strip()
        if ln:
            bullets.append("— " + ln)

    if not bullets:
        bullets = ["— " + q for q in build_clarify_questions(tags, max_questions=3)]

    out = "\n".join([qline] + bullets[:3]).strip()
    return _to_second_person_ru(out)


def fallback_answer_ok(docs: List[Dict[str, Any]], *, max_docs: int = 4) -> str:
    lines = []
    for d in (docs or [])[:max_docs]:
        title = str(d.get("title") or "").strip()
        url = str(d.get("url") or "").strip()
        lines.append(f"- {title} {url}".rstrip())
    if not lines:
        return "Не удалось сформировать ответ: нет подходящих источников."
    return "LLM недоступна, ниже найденные источники:\n" + "\n".join(lines)


def llm_answer_ok(query: str, docs: List[Dict[str, Any]], *, model: str = OLLAMA_MODEL) -> str:
    context = build_context_from_docs(docs, max_docs=6)
    prompt = f"""Вопрос пользователя: {query}

Источники (выдержки):
{context}

Задача:
1) Ответь по сути, опираясь ТОЛЬКО на источники.
2) Если в источниках нет ответа — честно скажи, что в базе нет точного ответа, и задай 1-2 уточняющих вопроса.
3) В конце коротко перечисли 2-4 источника (title или url), на которые опирался.
"""
    try:
        return ollama_generate(prompt, model=model, temperature=0.2).strip()
    except Exception:
        return fallback_answer_ok(docs)
