"""Provider settings: tenants self-serve their LLM/OCR (if platform allows), and
platform admins have full control (defaults, policy, and any tenant's config)."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..errors import forbidden
from ..providers import registry
from ..providers.embeddings import build_embedder
from ..providers.llm import ContextBlock, build_llm
from ..tenancy import CurrentUser, OrgContext, require_platform_admin, require_role

router = APIRouter(prefix="/api", tags=["settings"])

LLM_PROVIDERS = [
    {"id": "anthropic", "label": "Anthropic (Claude)", "base_url": "", "needs_key": True},
    {"id": "openai_compatible", "label": "OpenAI", "base_url": "https://api.openai.com/v1", "needs_key": True},
    {"id": "gemini", "label": "Google Gemini", "base_url": "", "needs_key": True},
    {"id": "openai_compatible", "label": "vLLM / Local (OpenAI-compatible)", "base_url": "http://localhost:8000/v1", "needs_key": False},
    {"id": "stub", "label": "Stub (no key, demo)", "base_url": "", "needs_key": False},
]
OCR_PROVIDERS = [
    {"id": "tesseract", "label": "Tesseract (local)"},
    {"id": "vlm", "label": "VLM (dùng LLM đa phương thức)"},
    {"id": "disabled", "label": "Tắt OCR"},
]


class LLMConfigBody(BaseModel):
    provider: str
    model: str = ""
    base_url: str = ""
    api_key: Optional[str] = None      # None = keep existing; "" = clear
    max_tokens: Optional[int] = None


class OCRConfigBody(BaseModel):
    provider: str
    model: str = ""
    lang: str = ""


def _extra_llm(body: LLMConfigBody) -> dict:
    return {"max_tokens": body.max_tokens} if body.max_tokens else {}


def _test_llm(cfg: dict) -> dict:
    try:
        prov = build_llm(cfg)
        res = prov.answer(question="Xin chào, bạn hoạt động chứ?", context=[
            ContextBlock(source="knowledge", title="Test", body="Đây là kiểm tra kết nối.")
        ], history=[], shop_name="Test")
        return {"ok": True, "model": res.model, "reply": (res.text or "")[:200]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# ---- tenant self-serve (org admin/owner) -----------------------------------

@router.get("/settings/llm")
def get_llm_settings(ctx: OrgContext = Depends(require_role("admin"))):
    policy = registry.get_platform_settings()
    return {
        "can_edit": policy["allow_tenant_llm"],
        "org_config": registry.public_view(f"llm:org:{ctx.org_id}"),
        "effective": {k: v for k, v in registry.resolve_llm_config(ctx.org_id).items() if k != "api_key"},
        "providers": LLM_PROVIDERS,
    }


@router.put("/settings/llm")
def put_llm_settings(body: LLMConfigBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not registry.get_platform_settings()["allow_tenant_llm"]:
        raise forbidden("tenant LLM configuration is disabled by the platform admin")
    registry.write_config(f"llm:org:{ctx.org_id}", provider=body.provider, model=body.model,
                          base_url=body.base_url, api_key=body.api_key, extra=_extra_llm(body))
    audit.record("settings.llm.update", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 detail={"provider": body.provider, "model": body.model})
    return {"ok": True}


@router.delete("/settings/llm")
def delete_llm_settings(ctx: OrgContext = Depends(require_role("admin"))):
    registry.delete_config(f"llm:org:{ctx.org_id}")
    return {"ok": True}


@router.post("/settings/llm/test")
def test_llm_settings(body: LLMConfigBody, ctx: OrgContext = Depends(require_role("admin"))):
    cfg = {"provider": body.provider, "model": body.model, "base_url": body.base_url,
           "api_key": body.api_key or (registry._load(f"llm:org:{ctx.org_id}") or {}).get("api_key", ""),
           "extra": _extra_llm(body)}
    return _test_llm(cfg)


@router.get("/settings/ocr")
def get_ocr_settings(ctx: OrgContext = Depends(require_role("admin"))):
    policy = registry.get_platform_settings()
    return {"can_edit": policy["allow_tenant_ocr"],
            "org_config": registry.public_view(f"ocr:org:{ctx.org_id}"),
            "effective": {k: v for k, v in registry.resolve_ocr_config(ctx.org_id).items() if k != "api_key"},
            "providers": OCR_PROVIDERS}


@router.put("/settings/ocr")
def put_ocr_settings(body: OCRConfigBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not registry.get_platform_settings()["allow_tenant_ocr"]:
        raise forbidden("tenant OCR configuration is disabled by the platform admin")
    extra = {"lang": body.lang} if body.lang else {}
    registry.write_config(f"ocr:org:{ctx.org_id}", provider=body.provider, model=body.model, extra=extra)
    return {"ok": True}


# ---- platform admin (full control) -----------------------------------------

@router.get("/admin/settings")
def admin_get_settings(_: CurrentUser = Depends(require_platform_admin)):
    return {
        "policy": registry.get_platform_settings(),
        "llm": registry.public_view("llm:platform"),
        "embedding": registry.public_view("embedding:platform"),
        "ocr": registry.public_view("ocr:platform"),
        "llm_providers": LLM_PROVIDERS,
        "ocr_providers": OCR_PROVIDERS,
    }


class PolicyBody(BaseModel):
    allow_tenant_llm: Optional[bool] = None
    allow_tenant_ocr: Optional[bool] = None


@router.put("/admin/settings/policy")
def admin_set_policy(body: PolicyBody, admin: CurrentUser = Depends(require_platform_admin)):
    registry.set_platform_settings(allow_tenant_llm=body.allow_tenant_llm, allow_tenant_ocr=body.allow_tenant_ocr)
    audit.record("admin.policy.update", actor_user_id=admin.id, detail=body.model_dump())
    return {"ok": True}


@router.put("/admin/settings/llm")
def admin_set_llm(body: LLMConfigBody, admin: CurrentUser = Depends(require_platform_admin)):
    registry.write_config("llm:platform", provider=body.provider, model=body.model,
                          base_url=body.base_url, api_key=body.api_key, extra=_extra_llm(body))
    audit.record("admin.llm.update", actor_user_id=admin.id, detail={"provider": body.provider})
    return {"ok": True}


@router.put("/admin/settings/embedding")
def admin_set_embedding(body: LLMConfigBody, admin: CurrentUser = Depends(require_platform_admin)):
    registry.write_config("embedding:platform", provider=body.provider, model=body.model,
                          base_url=body.base_url, api_key=body.api_key, extra={})
    audit.record("admin.embedding.update", actor_user_id=admin.id, detail={"provider": body.provider})
    return {"ok": True}


@router.put("/admin/settings/ocr")
def admin_set_ocr(body: OCRConfigBody, admin: CurrentUser = Depends(require_platform_admin)):
    extra = {"lang": body.lang} if body.lang else {}
    registry.write_config("ocr:platform", provider=body.provider, model=body.model, extra=extra)
    return {"ok": True}


@router.put("/admin/settings/org-llm/{org_id}")
def admin_set_org_llm(org_id: str, body: LLMConfigBody, admin: CurrentUser = Depends(require_platform_admin)):
    registry.write_config(f"llm:org:{org_id}", provider=body.provider, model=body.model,
                          base_url=body.base_url, api_key=body.api_key, extra=_extra_llm(body))
    audit.record("admin.org_llm.update", organization_id=org_id, actor_user_id=admin.id,
                 detail={"provider": body.provider})
    return {"ok": True}


@router.post("/admin/settings/llm/test")
def admin_test_llm(body: LLMConfigBody, _: CurrentUser = Depends(require_platform_admin)):
    cfg = {"provider": body.provider, "model": body.model, "base_url": body.base_url,
           "api_key": body.api_key or (registry._load("llm:platform") or {}).get("api_key", ""),
           "extra": _extra_llm(body)}
    return _test_llm(cfg)


@router.post("/admin/settings/embedding/test")
def admin_test_embedding(body: LLMConfigBody, _: CurrentUser = Depends(require_platform_admin)):
    cfg = {"provider": body.provider, "model": body.model, "base_url": body.base_url,
           "api_key": body.api_key or (registry._load("embedding:platform") or {}).get("api_key", ""), "extra": {}}
    try:
        emb = build_embedder(cfg)
        v = emb.embed_one("kiểm tra")
        return {"ok": True, "dim": len(v)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
