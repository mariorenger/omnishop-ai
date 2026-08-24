"""Provider resolution & configuration store.

Resolves which LLM / embedding / OCR provider to use, with precedence:
  platform default  ->  tenant override (if policy allows)  ->  env/stub.

Config lives in `provider_config` (api keys encrypted at rest). Building the
concrete provider objects is delegated to the provider modules (build_llm,
build_embedder, build_ocr) so this module has no heavy imports.
"""
from __future__ import annotations
from typing import Optional

from ..config import config
from ..db import no_tenant
from ..security import decrypt_secret, encrypt_secret

# ---- platform policy -------------------------------------------------------

def get_platform_settings() -> dict:
    with no_tenant() as conn:
        row = conn.execute(
            "SELECT allow_tenant_llm, allow_tenant_ocr FROM platform_settings WHERE id=1"
        ).fetchone()
    return {"allow_tenant_llm": bool(row["allow_tenant_llm"]),
            "allow_tenant_ocr": bool(row["allow_tenant_ocr"])} if row else \
           {"allow_tenant_llm": True, "allow_tenant_ocr": True}


def set_platform_settings(*, allow_tenant_llm: Optional[bool] = None, allow_tenant_ocr: Optional[bool] = None) -> None:
    cur = get_platform_settings()
    al = cur["allow_tenant_llm"] if allow_tenant_llm is None else allow_tenant_llm
    ao = cur["allow_tenant_ocr"] if allow_tenant_ocr is None else allow_tenant_ocr
    with no_tenant() as conn:
        conn.execute("UPDATE platform_settings SET allow_tenant_llm=%s, allow_tenant_ocr=%s WHERE id=1", (al, ao))


# ---- config store ----------------------------------------------------------

def _load(scope: str) -> Optional[dict]:
    with no_tenant() as conn:
        row = conn.execute(
            "SELECT provider, model, base_url, api_key_enc, extra FROM provider_config WHERE scope=%s", (scope,)
        ).fetchone()
    if not row:
        return None
    api_key = ""
    if row["api_key_enc"]:
        try:
            api_key = decrypt_secret(bytes(row["api_key_enc"]))
        except Exception:  # noqa: BLE001
            api_key = ""
    return {"provider": row["provider"], "model": row["model"], "base_url": row["base_url"],
            "api_key": api_key, "extra": row["extra"] or {}}


def public_view(scope: str) -> Optional[dict]:
    c = _load(scope)
    if not c:
        return None
    return {"provider": c["provider"], "model": c["model"], "base_url": c["base_url"],
            "has_key": bool(c["api_key"]), "extra": c["extra"]}


def write_config(scope: str, *, provider: str, model: str = "", base_url: str = "",
                 api_key: Optional[str] = None, extra: Optional[dict] = None) -> None:
    import json
    # keep existing key if api_key is None (not provided); empty string clears it
    existing = _load(scope)
    if api_key is None:
        enc = None
        keep_existing = existing and existing["api_key"]
    else:
        enc = encrypt_secret(api_key) if api_key else None
        keep_existing = False
    with no_tenant() as conn:
        if keep_existing:
            conn.execute(
                """INSERT INTO provider_config (scope, provider, model, base_url, extra, updated_at)
                   VALUES (%s,%s,%s,%s,%s, now())
                   ON CONFLICT (scope) DO UPDATE SET provider=EXCLUDED.provider, model=EXCLUDED.model,
                     base_url=EXCLUDED.base_url, extra=EXCLUDED.extra, updated_at=now()""",
                (scope, provider, model, base_url, json.dumps(extra or {})),
            )
        else:
            conn.execute(
                """INSERT INTO provider_config (scope, provider, model, base_url, api_key_enc, extra, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s, now())
                   ON CONFLICT (scope) DO UPDATE SET provider=EXCLUDED.provider, model=EXCLUDED.model,
                     base_url=EXCLUDED.base_url, api_key_enc=EXCLUDED.api_key_enc, extra=EXCLUDED.extra, updated_at=now()""",
                (scope, provider, model, base_url, enc, json.dumps(extra or {})),
            )


def delete_config(scope: str) -> None:
    with no_tenant() as conn:
        conn.execute("DELETE FROM provider_config WHERE scope=%s", (scope,))


# ---- env fallbacks ---------------------------------------------------------

def _env_llm() -> dict:
    want = config.LLM_PROVIDER
    provider = "anthropic" if (want == "anthropic" or (want == "auto" and config.ANTHROPIC_API_KEY)) else "stub"
    return {"provider": provider, "model": config.LLM_MODEL, "base_url": "",
            "api_key": config.ANTHROPIC_API_KEY,
            "extra": {"max_tokens": config.LLM_MAX_TOKENS, "effort": config.LLM_EFFORT}}


def _env_embedding() -> dict:
    if config.EMBEDDING_PROVIDER == "openai" and config.OPENAI_API_KEY:
        return {"provider": "openai_compatible", "model": config.EMBEDDING_MODEL,
                "base_url": config.OPENAI_BASE_URL, "api_key": config.OPENAI_API_KEY, "extra": {}}
    return {"provider": "local", "model": "local-hash", "base_url": "", "api_key": "", "extra": {}}


def _env_ocr() -> dict:
    return {"provider": config.OCR_PROVIDER, "model": config.OCR_MODEL, "base_url": "", "api_key": "", "extra": {}}


# ---- resolution ------------------------------------------------------------

def resolve_llm_config(org_id: Optional[str]) -> dict:
    if org_id and get_platform_settings()["allow_tenant_llm"]:
        c = _load(f"llm:org:{org_id}")
        if c:
            return c
    return _load("llm:platform") or _env_llm()


def resolve_embedding_config() -> dict:
    return _load("embedding:platform") or _env_embedding()


def resolve_payment_config() -> dict:
    return _load("payment:platform") or {"provider": "manual", "model": "", "base_url": "", "api_key": "", "extra": {}}


def resolve_ocr_config(org_id: Optional[str]) -> dict:
    if org_id and get_platform_settings()["allow_tenant_ocr"]:
        c = _load(f"ocr:org:{org_id}")
        if c:
            return c
    return _load("ocr:platform") or _env_ocr()


# ---- builders (cached by signature) ---------------------------------------

_llm_cache: dict = {}
_emb_cache: dict = {}


def _sig(c: dict) -> tuple:
    return (c["provider"], c.get("model"), c.get("base_url"), bool(c.get("api_key")),
            tuple(sorted((c.get("extra") or {}).items())))


def get_llm(org_id: Optional[str] = None):
    from .llm import build_llm
    cfg = resolve_llm_config(org_id)
    key = _sig(cfg)
    if key not in _llm_cache:
        _llm_cache[key] = build_llm(cfg)
    return _llm_cache[key]


def get_embedder():
    from .embeddings import build_embedder
    cfg = resolve_embedding_config()
    key = _sig(cfg)
    if key not in _emb_cache:
        _emb_cache[key] = build_embedder(cfg)
    return _emb_cache[key]


def get_ocr(org_id: Optional[str] = None):
    from .ocr import build_ocr
    return build_ocr(resolve_ocr_config(org_id), org_id)
