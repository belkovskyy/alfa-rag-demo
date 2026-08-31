from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
import joblib


ALLOWED_STATUS = ("ok", "need_clarify", "no_answer")


def decision_policy_rules(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Main (recommended) decision policy: deterministic rules based on meta features.
    """
    if meta.get("retrieval_error"):
        return "no_answer", f"Ошибка retrieval: {meta['retrieval_error']}"

    intent = meta.get("intent", "faq")
    top_score = float(meta.get("top_score", 0.0))
    overlap = float(meta.get("overlap", 0.0))
    n_base = int(meta.get("n_base", 0))
    n_keys = int(meta.get("n_keys", 0))

    min_score = float(meta.get("min_score", 0.80))
    strong_score = float(meta.get("strong_score", 0.86))
    min_docs = int(meta.get("min_docs", 3))
    min_overlap = float(meta.get("min_overlap", 0.15))

    has_product = bool(meta.get("has_product_markers", False))
    has_problem = bool(meta.get("has_problem_markers", False))

    has_process = bool(meta.get("has_process_markers", False))
    has_action = bool(meta.get("has_action_markers", False))
    has_family = bool(meta.get("has_family_markers", False))
    has_personal_id = bool(meta.get("has_personal_id_context", False))

    needs_context = bool(meta.get("needs_context", False))
    underspecified = bool(meta.get("underspecified", False))

    # 1) personal intent -> need_clarify
    if intent == "personal":
        return "need_clarify", "Похоже на персональный кейс. Уточните продукт/канал/детали (без персональных данных)."

    # 2) strong personal/process markers -> need_clarify
    if has_process or has_action or has_family or has_personal_id:
        return "need_clarify", "Похоже на конкретный кейс/процесс. Уточните продукт и детали (без персональных данных)."

    # 3) vague / needs_context
    if intent == "vague" or needs_context:
        if top_score >= strong_score and overlap >= min_overlap and n_base >= min_docs and (n_keys >= 2):
            return "ok", f"Похоже, нашёл по теме (top_score={top_score:.3f}). Если уточните детали — будет точнее."
        return "need_clarify", "Запрос короткий/расплывчатый. Уточните продукт/раздел и что именно нужно."

    # 4) weak retrieval
    if n_base < min_docs or top_score < min_score:
        return "no_answer", f"Недостаточно уверенно (top_score={top_score:.3f}). Попробуйте переформулировать."

    if n_keys == 0:
        return "need_clarify", "Не понял ключевые слова запроса. Уточните предмет: услуга/продукт/раздел."

    # 5) OK gate
    def ok_gate() -> bool:
        if intent != "faq":
            return False
        if n_base < min_docs:
            return False
        if top_score < strong_score:
            return False
        if overlap < min_overlap:
            return False
        if underspecified and not has_product:
            return False
        if n_keys <= 1 and not has_product:
            return False
        if has_problem and not has_product:
            return False
        return True

    if ok_gate():
        return "ok", f"Нашёл релевантные источники (top_score={top_score:.3f}, overlap={overlap:.2f})."

    return "need_clarify", "Похоже, нужно уточнение (продукт/канал/детали). Ниже — общие материалы по теме."


def _meta_to_row(meta: Dict[str, Any]) -> pd.DataFrame:
    cols = [
        "intent","top_score","overlap","n_base","n_keys","word_count",
        "has_question_mark","has_question_word",
        "has_product_markers","has_problem_markers",
        "has_process_markers","has_action_markers","has_family_markers","has_personal_id_context",
        "needs_context","underspecified",
        "is_callcenter","pinned_doc",
    ]
    row = {c: meta.get(c, 0) for c in cols}
    row["intent"] = meta.get("intent", "faq")
    return pd.DataFrame([row])


@dataclass
class DecisionModelGate:
    """
    Optional ML gate on top of rules. Recommended to keep OFF by default.
    """
    model_path: str
    ok_threshold: float = 0.55

    def __post_init__(self):
        self.model = joblib.load(self.model_path)

    def predict_ok_proba(self, meta: Dict[str, Any]) -> float:
        X = _meta_to_row(meta)
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        p = dict(zip(classes, proba))
        return float(p.get("ok", 0.0))

    def apply(self, meta: Dict[str, Any]) -> Tuple[str, str]:
        p_ok = self.predict_ok_proba(meta)
        if p_ok >= self.ok_threshold:
            return "ok", f"ML: ok (p_ok={p_ok:.2f})"
        return "need_clarify", f"ML: need_clarify (p_ok={p_ok:.2f})"
