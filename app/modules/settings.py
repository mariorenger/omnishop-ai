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
from ..providers.models import list_models
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


@router.post("/settings/llm/models")
def tenant_list_models(body: LLMConfigBody, ctx: OrgContext = Depends(require_role("admin"))):
    key = body.api_key or (registry._load(f"llm:org:{ctx.org_id}") or {}).get("api_key", "")
    return list_models({"provider": body.provider, "base_url": body.base_url, "api_key": key})


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


PAYMENT_PROVIDERS = [
    {"id": "manual", "label": "Thủ công / chuyển khoản (demo)"},
    {"id": "vietqr", "label": "VietQR (QR chuyển khoản)"},
    {"id": "vnpay", "label": "VNPay (thẻ/ATM/QR nội địa)"},
    {"id": "momo", "label": "MoMo (ví điện tử)"},
    {"id": "stripe", "label": "Stripe (thẻ quốc tế)"},
]


class PaymentConfigBody(BaseModel):
    provider: str
    api_key: Optional[str] = None       # None = keep; "" = clear. Holds the SECRET:
                                        # stripe secret / vnpay hash_secret / momo secret_key
    # stripe
    publishable_key: str = ""
    webhook_secret: str = ""
    success_url: str = ""
    cancel_url: str = ""
    currency: str = "USD"
    # vietqr
    bank_bin: str = ""
    account_no: str = ""
    account_name: str = ""
    template: str = "compact2"
    # vnpay
    tmn_code: str = ""
    pay_url: str = ""
    return_url: str = ""
    # momo
    partner_code: str = ""
    access_key: str = ""
    redirect_url: str = ""
    ipn_url: str = ""
    endpoint: str = ""


@router.get("/admin/settings/payment")
def admin_get_payment(_: CurrentUser = Depends(require_platform_admin)):
    return {"config": registry.public_view("payment:platform"), "providers": PAYMENT_PROVIDERS}


@router.put("/admin/settings/payment")
def admin_set_payment(body: PaymentConfigBody, admin: CurrentUser = Depends(require_platform_admin)):
    extra = {
        "publishable_key": body.publishable_key, "webhook_secret": body.webhook_secret,
        "success_url": body.success_url, "cancel_url": body.cancel_url, "currency": body.currency,
        "bank_bin": body.bank_bin, "account_no": body.account_no,
        "account_name": body.account_name, "template": body.template or "compact2",
        "tmn_code": body.tmn_code, "pay_url": body.pay_url, "return_url": body.return_url,
        "partner_code": body.partner_code, "access_key": body.access_key,
        "redirect_url": body.redirect_url, "ipn_url": body.ipn_url, "endpoint": body.endpoint,
    }
    registry.write_config("payment:platform", provider=body.provider, api_key=body.api_key, extra=extra)
    audit.record("admin.payment.update", actor_user_id=admin.id, detail={"provider": body.provider})
    return {"ok": True}


class MetaAppBody(BaseModel):
    app_id: str = ""
    app_secret: Optional[str] = None
    verify_token: str = "omnishop-verify"


@router.get("/admin/settings/meta")
def admin_get_meta(_: CurrentUser = Depends(require_platform_admin)):
    v = registry.public_view("channel:meta")
    return {"app_id": (v or {}).get("model", ""), "has_secret": bool(v and v.get("has_key")),
            "verify_token": ((v or {}).get("extra") or {}).get("verify_token", "omnishop-verify")}


@router.put("/admin/settings/meta")
def admin_set_meta(body: MetaAppBody, admin: CurrentUser = Depends(require_platform_admin)):
    registry.write_config("channel:meta", provider="meta", model=body.app_id, api_key=body.app_secret,
                          extra={"verify_token": body.verify_token})
    audit.record("admin.meta.update", actor_user_id=admin.id, detail={"app_id": body.app_id})
    return {"ok": True}


@router.post("/admin/settings/llm/models")
def admin_list_models(body: LLMConfigBody, _: CurrentUser = Depends(require_platform_admin)):
    key = body.api_key or (registry._load("llm:platform") or {}).get("api_key", "")
    return list_models({"provider": body.provider, "base_url": body.base_url, "api_key": key})


@router.post("/admin/settings/embedding/models")
def admin_list_embedding_models(body: LLMConfigBody, _: CurrentUser = Depends(require_platform_admin)):
    key = body.api_key or (registry._load("embedding:platform") or {}).get("api_key", "")
    return list_models({"provider": body.provider, "base_url": body.base_url, "api_key": key})


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


# ============================ admin: plans & cost rates ======================
# Everything an operator prices from is editable here — no redeploy, no code.

_PLAN_ENT_KEYS = ("llm_mode", "billing_mode", "ai_tokens_month", "overage_per_1k",
                  "payg_per_1k", "ai_messages_month", "shops", "channels", "storage_mb",
                  "human_handoff", "channels_allowed")


class PlanBody(BaseModel):
    name: Optional[str] = None
    price_month: Optional[float] = None
    entitlements: Optional[dict] = None      # merged into existing entitlements


@router.get("/admin/plans")
def admin_list_plans(_: CurrentUser = Depends(require_platform_admin)):
    from ..db import no_tenant
    with no_tenant() as conn:
        rows = conn.execute(
            "SELECT code, name, price_month, entitlements FROM plan ORDER BY price_month"
        ).fetchall()
    return [{"code": r["code"], "name": r["name"], "price_month": float(r["price_month"]),
             "entitlements": r["entitlements"]} for r in rows]


@router.put("/admin/plans/{code}")
def admin_update_plan(code: str, body: PlanBody, admin: CurrentUser = Depends(require_platform_admin)):
    import json as _json
    from ..db import no_tenant
    from ..errors import not_found
    with no_tenant() as conn:
        cur = conn.execute("SELECT name, price_month, entitlements FROM plan WHERE code=%s", (code,)).fetchone()
        if not cur:
            raise not_found("plan not found")
        ent = dict(cur["entitlements"] or {})
        for k, v in (body.entitlements or {}).items():
            if k in _PLAN_ENT_KEYS:
                ent[k] = v
        name = body.name if body.name is not None else cur["name"]
        price = body.price_month if body.price_month is not None else float(cur["price_month"])
        conn.execute("UPDATE plan SET name=%s, price_month=%s, entitlements=%s WHERE code=%s",
                     (name, price, _json.dumps(ent), code))
    audit.record("admin.plan.update", actor_user_id=admin.id, target=code,
                 detail={"price_month": price})
    return {"code": code, "name": name, "price_month": price, "entitlements": ent}


class CostBody(BaseModel):
    cost_input_per_m: Optional[float] = None
    cost_output_per_m: Optional[float] = None
    cost_embedding_per_m: Optional[float] = None


@router.get("/admin/settings/cost")
def admin_get_cost(_: CurrentUser = Depends(require_platform_admin)):
    from . import usage
    return usage.cost_rates()


@router.put("/admin/settings/cost")
def admin_set_cost(body: CostBody, admin: CurrentUser = Depends(require_platform_admin)):
    from ..db import no_tenant
    from . import usage
    with no_tenant() as conn:
        conn.execute(
            """UPDATE platform_settings
               SET cost_input_per_m=coalesce(%s, cost_input_per_m),
                   cost_output_per_m=coalesce(%s, cost_output_per_m),
                   cost_embedding_per_m=coalesce(%s, cost_embedding_per_m)
               WHERE id=1""",
            (body.cost_input_per_m, body.cost_output_per_m, body.cost_embedding_per_m),
        )
    usage._rates_cache["val"] = None    # invalidate cache immediately
    audit.record("admin.cost.update", actor_user_id=admin.id)
    return usage.cost_rates()


# ============================ admin: Google Sign-In ==========================

class GoogleBody(BaseModel):
    client_id: str = ""
    client_secret: Optional[str] = None     # None = keep; "" = clear


@router.get("/admin/settings/google")
def admin_get_google(_: CurrentUser = Depends(require_platform_admin)):
    v = registry.public_view("auth:google") or {}
    return {"client_id": v.get("model", ""), "has_secret": v.get("has_key", False)}


@router.put("/admin/settings/google")
def admin_set_google(body: GoogleBody, admin: CurrentUser = Depends(require_platform_admin)):
    registry.write_config("auth:google", provider="google", model=body.client_id,
                          api_key=body.client_secret, extra={})
    audit.record("admin.google.update", actor_user_id=admin.id, detail={"client_id": body.client_id})
    return {"ok": True}
