"""Central configuration read from the environment (12-factor)."""
from __future__ import annotations
import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Config:
    APP_SECRET = _get("APP_SECRET", "dev-only-change-me")

    PG_DSN = _get("PG_DSN", "postgresql://omni_app:omni_app@localhost:5432/omnishop")
    PG_DSN_ADMIN = _get("PG_DSN_ADMIN", "postgresql://omni:omni@localhost:5432/omnishop")
    REDIS_URL = _get("REDIS_URL", "redis://localhost:6379/0")

    # LLM (ADR-007)
    LLM_PROVIDER = _get("LLM_PROVIDER", "auto")  # auto | stub | anthropic
    ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY", "")
    LLM_MODEL = _get("LLM_MODEL", "claude-opus-5")
    LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "1024"))
    LLM_EFFORT = _get("LLM_EFFORT", "low")

    # Embeddings
    EMBEDDING_PROVIDER = _get("EMBEDDING_PROVIDER", "local")  # local | openai
    OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = int(_get("EMBEDDING_DIM", "384"))

    # OCR (flexible/swappable). tesseract | vlm | disabled
    OCR_PROVIDER = _get("OCR_PROVIDER", "tesseract")
    OCR_MODEL = _get("OCR_MODEL", "")  # for vlm OCR (defaults to the LLM model)

    # Cost estimation ($/1M tokens) — per-tenant COGS in usage metering.
    COST_INPUT_PER_M = float(_get("COST_INPUT_PER_M", "5.0"))
    COST_OUTPUT_PER_M = float(_get("COST_OUTPUT_PER_M", "25.0"))
    COST_EMBEDDING_PER_M = float(_get("COST_EMBEDDING_PER_M", "0.02"))

    # Token estimate fallback when the provider does not report usage.
    @staticmethod
    def est_tokens(text: str) -> int:
        # ~4 chars/token heuristic; good enough for cost estimates when the
        # provider doesn't return real counts.
        return max(1, len(text) // 4)


config = Config()
