from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import RERANK_TOPN
from .decision import decision_policy_rules, DecisionModelGate
from .guardrails import guardrails_features
from .retrieval import Retriever, normalize_results
from .rerank import Reranker
from .clarify import ClarifyBank, derive_clarify_tags, pick_clarify_hint_topk
from .llm import llm_answer_ok, llm_rewrite_need_clarify, normalize_clarify_note


@dataclass
class Pipeline:
    retriever: Retriever
    reranker: Optional[Reranker] = None
    decision_model: Optional[DecisionModelGate] = None  # optional ML gate

    def retrieval_search_docs(
        self,
        query: str,
        *,
        k_chunks: int = 80,
        k_docs: int = 10,
        oversample_docs: int = 60,
    ) -> Tuple[pd.DataFrame, Optional[str], pd.DataFrame]:
        """
        Returns:
          ranked_df, err, dense_df
        ranked_df = dense_df with optional rerank re-ordering.
        dense_df  = always dense order (for stable meta/thresholds).
        """
        q = (query or "").strip()
        try:
            dense_df = self.retriever.search_docs_dense(q, k_chunks=k_chunks, k_docs=max(int(oversample_docs), int(k_docs)))
            dense_df = normalize_results(dense_df).sort_values("score", ascending=False).reset_index(drop=True)

            ranked_df = dense_df.copy()

            # Rerank is enabled when a reranker instance is provided.
            # (The CLI scripts may still gate creation of the reranker via USE_RERANK env.)
            if self.reranker is not None:
                ranked_df = self.reranker.rerank(
                    q,
                    ranked_df,
                    topn=min(int(RERANK_TOPN), len(ranked_df)),
                    keep=max(int(k_docs), 6),
                    batch_size=32,
                )
                ranked_df = normalize_results(ranked_df)

            ranked_df = ranked_df.head(int(k_docs)).reset_index(drop=True)
            dense_df = dense_df.head(int(k_docs)).reset_index(drop=True)
            return ranked_df, None, dense_df

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            empty = normalize_results(pd.DataFrame())
            return empty, err, empty

    def safe_search_docs(
        self,
        query: str,
        *,
        k_chunks: int = 80,
        k_docs: int = 10,
        min_score: float = 0.80,
        min_docs: int = 3,
        strong_score: float = 0.86,
        min_overlap: float = 0.15,
    ) -> Dict[str, Any]:
        ranked_df, err, dense_df = self.retrieval_search_docs(
            query,
            k_chunks=k_chunks,
            k_docs=k_docs,
            oversample_docs=max(15, k_docs * 2),
        )

        topk_df, meta = guardrails_features(
            query,
            ranked_df,
            dense_df,
            retrieval_error=err,
            k_docs=k_docs,
        )

        meta.update({
            "min_score": float(min_score),
            "strong_score": float(strong_score),
            "min_docs": int(min_docs),
            "min_overlap": float(min_overlap),
        })

        # Decision: rules first; optional ML gate can override only if you want.
        status, message = decision_policy_rules(meta)
        if self.decision_model is not None:
            # Optional: only use ML on borderline cases
            if status == "need_clarify" and meta.get("intent", "faq") == "faq":
                ml_status, ml_msg = self.decision_model.apply(meta)
                # keep conservative: allow ML to switch need_clarify -> ok only when confident
                if ml_status == "ok":
                    status, message = ml_status, ml_msg

        return {
            "status": status,
            "message": message,
            "results": topk_df,
            "docs": topk_df.to_dict("records"),
            "meta": meta,
        }

    def ask(
        self,
        query: str,
        *,
        k_docs: int = 8,
        k_chunks: int = 80,
        clarify_bank: Optional[ClarifyBank] = None,
        embed_fn=None,
        clarify_min_sim: float = 0.70,
        # If a query is *very* similar to a need_clarify example in gold_labels.csv,
        # we can directly use that human-written clarification text as the main "final".
        # This mirrors the notebook behaviour and makes need_clarify much more specific.
        clarify_gold_use_min_sim: float = 0.86,
        clarify_topk: int = 5,
        clarify_mode: str = "llm",  # "off" | "gold" | "llm" | "gold_then_llm"
        use_llm: bool = True,
        llm_for: str = "both",  # "need_clarify" | "ok" | "both"
        llm_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        out = self.safe_search_docs(query, k_docs=k_docs, k_chunks=k_chunks)
        status = out.get("status")
        msg = out.get("message", "")
        docs = out.get("docs", []) or []
        meta = out.get("meta", {}) or {}

        hint = None
        if status == "need_clarify" and clarify_bank is not None and embed_fn is not None:
            hint = pick_clarify_hint_topk(
                query,
                clarify_bank,
                embed_fn,
                topk=clarify_topk,
                min_sim=clarify_min_sim,
                max_examples=3,
            )

        final_msg = msg

        if status == "need_clarify":
            hint_codes = (hint.get("codes") if hint else []) or []
            best_text = (hint.get("best_text") if hint else "") or ""
            top_sim = float(hint.get("top_sim", 0.0)) if hint else 0.0
            ex = [e.get("matched_query") for e in (hint.get("examples") if hint else []) or [] if e.get("matched_query")]

            # Tags are useful both for LLM prompting and for gold fallback normalization.
            tags = derive_clarify_tags(query, meta or {}, hint_codes or [])

            # In modes that are allowed to use gold ("gold" / "gold_then_llm"),
            # if we found a very close match in the clarify bank, we prefer that
            # human-written clarification as the main output.
            gold_can_be_used = clarify_mode in {"gold", "gold_then_llm"}
            gold_is_strong = bool(best_text) and (top_sim >= float(clarify_gold_use_min_sim))
            if gold_can_be_used and gold_is_strong:
                final_msg = normalize_clarify_note(best_text, tags)
                return {"out": out, "final": final_msg, "hint": hint}

            if clarify_mode == "off":
                final_msg = msg

            elif clarify_mode == "gold":
                # Use the best human-written clarify note (if any) as the final clarification.
                final_msg = normalize_clarify_note(best_text or msg, tags)

            elif clarify_mode in {"llm", "gold_then_llm"}:
                if use_llm and llm_for in {"need_clarify", "both"}:
                    # In gold_then_llm mode we still pass the gold hint text into the prompt.
                    hint_text = best_text if clarify_mode == "gold_then_llm" else ""
                    if llm_model:
                        final_msg = llm_rewrite_need_clarify(
                            query,
                            hint_codes=hint_codes,
                            meta=meta,
                            hint_text=hint_text,
                            hint_examples=ex,
                            model=llm_model,
                        )
                    else:
                        final_msg = llm_rewrite_need_clarify(
                            query,
                            hint_codes=hint_codes,
                            meta=meta,
                            hint_text=hint_text,
                            hint_examples=ex,
                        )
                else:
                    # No LLM: use gold note if available, otherwise rule message.
                    final_msg = normalize_clarify_note(best_text or msg, tags)

            else:
                final_msg = msg

        elif status == "ok":
            if use_llm and llm_for in {"ok", "both"}:
                if llm_model:
                    final_msg = llm_answer_ok(query, docs, model=llm_model)
                else:
                    final_msg = llm_answer_ok(query, docs)
            else:
                final_msg = msg

        return {"out": out, "final": final_msg, "hint": hint}
