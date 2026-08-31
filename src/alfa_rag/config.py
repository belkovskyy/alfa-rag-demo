from __future__ import annotations
import os

# ----- Toggles -----
USE_RERANK: bool = os.getenv("USE_RERANK", "0").strip() in {"1", "true", "yes", "on"}
USE_DECISION_ML: bool = os.getenv("USE_DECISION_ML", "0").strip() in {"1", "true", "yes", "on"}

# ----- Models -----
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
E5_MODEL: str = os.getenv("E5_MODEL", "intfloat/multilingual-e5-small")

# Ollama
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# ----- Retrieval params -----
K_CHUNKS_DEFAULT: int = int(os.getenv("K_CHUNKS", "80"))
K_DOCS_DEFAULT: int = int(os.getenv("K_DOCS", "10"))

RERANK_TOPN: int = int(os.getenv("RERANK_TOPN", "40"))  # how many dense docs to rerank
