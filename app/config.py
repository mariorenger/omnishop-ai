"""Central configuration read from the environment (12-factor)."""
from __future__ import annotations
import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Config:
    APP_SECRET = _get("APP_SECRET", "dev-only-change-me")
    CORS_ORIGINS = [o.strip() for o in _get("CORS_ORIGINS", "*").split(",") if o.strip()]

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

    # Meta (Facebook/Instagram) platform app — the SaaS owns one FB App.
    META_APP_ID = _get("META_APP_ID", "")           # Facebook Login (OAuth)
    META_APP_SECRET = _get("META_APP_SECRET", "")   # webhook signature + token exchange
    META_VERIFY_TOKEN = _get("META_VERIFY_TOKEN", "omnishop-verify")  # webhook handshake
    OAUTH_REDIRECT_BASE = _get("OAUTH_REDIRECT_BASE", "http://localhost:8000")  # public base for callbacks

    # Transactional email
    EMAIL_PROVIDER = _get("EMAIL_PROVIDER", "console")  # console | smtp | resend
    EMAIL_FROM = _get("EMAIL_FROM", "OmniShop AI <no-reply@omnishop.local>")
    RESEND_API_KEY = _get("RESEND_API_KEY", "")
    SMTP_HOST = _get("SMTP_HOST", "")
    SMTP_PORT = int(_get("SMTP_PORT", "587"))
    SMTP_USER = _get("SMTP_USER", "")
    SMTP_PASS = _get("SMTP_PASS", "")

    # Cost estimation ($/1M tokens) — per-tenant COGS in usage metering.
    COST_INPUT_PER_M = float(_get("COST_INPUT_PER_M", "5.0"))
    COST_OUTPUT_PER_M = float(_get("COST_OUTPUT_PER_M", "25.0"))
    COST_EMBEDDING_PER_M = float(_get("COST_EMBEDDING_PER_M", "0.02"))

    # Google Sign-In (OAuth). Client id/secret can also be set in-app (admin).
    GOOGLE_CLIENT_ID = _get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = _get("GOOGLE_CLIENT_SECRET", "")

    # Platform admin bootstrap — the deployer's known admin login. On startup, if
    # this user is absent it is created (and promoted) so the operator can sign in.
    BOOTSTRAP_ADMIN_EMAIL = _get("BOOTSTRAP_ADMIN_EMAIL", "")
    BOOTSTRAP_ADMIN_PASSWORD = _get("BOOTSTRAP_ADMIN_PASSWORD", "")
    # Google emails auto-promoted to platform admin on first sign-in (comma list).
    PLATFORM_ADMIN_EMAILS = [e.strip().lower() for e in _get("PLATFORM_ADMIN_EMAILS", "").split(",") if e.strip()]

    # Token estimate fallback when the provider does not report usage.
    @staticmethod
    def est_tokens(text: str) -> int:
        # ~4 chars/token heuristic; good enough for cost estimates when the
        # provider doesn't return real counts.
        return max(1, len(text) // 4)


config = Config()
