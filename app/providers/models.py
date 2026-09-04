"""List available models from a provider so the UI can offer a dropdown.

Supports Anthropic (SDK), OpenAI-compatible (OpenAI / vLLM / local), and Gemini
(via its OpenAI-compatible /models endpoint). Returns {ok, models, error}.
"""
from __future__ import annotations
from typing import List

from ..config import config
from .llm import GEMINI_OPENAI_BASE

# sensible fallbacks when a provider can't be queried (no key / offline)
_FALLBACK = {
    "anthropic": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    "openai_compatible": ["gpt-4o-mini", "gpt-4o"],
    "stub": ["stub"],
}


def _openai_models(base: str, key: str) -> List[str]:
    import httpx
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    r = httpx.get(f"{base.rstrip('/')}/models", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    ids = [m.get("id", "") for m in data if m.get("id")]
    # gemini returns ids like "models/gemini-1.5-flash"
    return sorted({i.split("/")[-1] for i in ids})


def list_models(cfg: dict) -> dict:
    provider = (cfg.get("provider") or "stub").lower()
    key = cfg.get("api_key") or ""
    try:
        if provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=key or config.ANTHROPIC_API_KEY)
            ids = [m.id for m in client.models.list(limit=50)]
            return {"ok": True, "models": ids or _FALLBACK["anthropic"]}
        if provider == "gemini":
            base = cfg.get("base_url") or GEMINI_OPENAI_BASE
            return {"ok": True, "models": _openai_models(base, key)}
        if provider == "openai_compatible":
            base = cfg.get("base_url") or config.OPENAI_BASE_URL or "https://api.openai.com/v1"
            return {"ok": True, "models": _openai_models(base, key)}
        return {"ok": True, "models": _FALLBACK.get(provider, [])}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "models": _FALLBACK.get(provider, [])}
