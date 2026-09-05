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


# ---- operational config (admin-editable, env fallback) --------------------
# These used to be pure env vars. They now live in provider_config so the
# platform admin can change them in the UI without a redeploy; env stays only
# as the first-boot fallback.

def resolve_email_config() -> dict:
    """Transactional email settings for the whole platform."""
    c = _load("notify:email")
    if c:
        ex = c.get("extra") or {}
        return {"provider": c["provider"] or "console",
                "from": ex.get("from") or config.EMAIL_FROM,
                "secret": c.get("api_key", ""),   # resend key OR smtp password
                "smtp_host": ex.get("smtp_host", ""),
                "smtp_port": int(ex.get("smtp_port") or config.SMTP_PORT),
                "smtp_user": ex.get("smtp_user", "")}
    return {"provider": config.EMAIL_PROVIDER, "from": config.EMAIL_FROM,
            "secret": config.RESEND_API_KEY if config.EMAIL_PROVIDER == "resend" else config.SMTP_PASS,
            "smtp_host": config.SMTP_HOST, "smtp_port": config.SMTP_PORT, "smtp_user": config.SMTP_USER}


def resolve_meta_app() -> dict:
    """Shared Facebook App credentials (used by OAuth + webhook verify/signature)."""
    c = _load("channel:meta") or {}
    ex = c.get("extra") or {}
    return {"app_id": c.get("model") or config.META_APP_ID,
            "app_secret": c.get("api_key") or config.META_APP_SECRET,
            "verify_token": ex.get("verify_token") or config.META_VERIFY_TOKEN}


import time as _time  # noqa: E402
_runtime_cache: dict = {"val": None, "ts": 0.0}


def _runtime() -> dict:
    now = _time.time()
    if _runtime_cache["val"] is None or now - _runtime_cache["ts"] > 30:
        c = _load("platform:runtime")
        _runtime_cache["val"] = (c or {}).get("extra") or {}
        _runtime_cache["ts"] = now
    return _runtime_cache["val"]


def public_base() -> str:
    """Public HTTPS base URL for OAuth redirects + channel webhooks."""
    return (_runtime().get("public_base") or config.OAUTH_REDIRECT_BASE or "").rstrip("/")


def stale_seconds() -> int:
    v = _runtime().get("stale_seconds")
    try:
        return int(v)
    except (TypeError, ValueError):
        return config.CHANNEL_STALE_SECONDS


def set_runtime(*, public_base: Optional[str] = None, stale_seconds: Optional[int] = None) -> None:
    cur = dict(_runtime())
    if public_base is not None:
        cur["public_base"] = public_base.strip().rstrip("/")
    if stale_seconds is not None:
        cur["stale_seconds"] = int(stale_seconds)
    write_config("platform:runtime", provider="runtime", extra=cur)
    _runtime_cache["val"] = None    # invalidate immediately


# ---- builders (cached by signature) ---------------------------------------

_llm_cache: dict = {}
_emb_cache: dict = {}


def _sig(c: dict) -> tuple:
    return (c["provider"], c.get("model"), c.get("base_url"), bool(c.get("api_key")),
            tuple(sorted((c.get("extra") or {}).items())))


def get_llm(org_id: Optional[str] = None, model: Optional[str] = None):
    """Resolve the LLM for an org. A non-empty `model` overrides the resolved
    model (used for per-bot model selection) while keeping the same provider,
    endpoint and credentials."""
    from .llm import build_llm
    cfg = resolve_llm_config(org_id)
    if model:
        cfg = dict(cfg, model=model)
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
