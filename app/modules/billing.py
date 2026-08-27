"""Plans, subscription, and the EntitlementService (ADR-004).

Entitlements map plan -> capabilities/quotas; the backend enforces them (never
the frontend). Payments are out of scope for the MVP (subscription.provider =
'manual'); a BillingProvider/PaymentProvider adapter slots in later.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from .. import audit
from ..db import no_tenant, tenant_tx
from ..errors import bad_request, not_found
from ..providers.payment import get_payment
from ..tenancy import OrgContext, get_org_context, require_role

router = APIRouter(prefix="/api", tags=["billing"])

DEFAULT_PLAN = "free"


def resolve_entitlements(org_id: str) -> dict:
    with no_tenant() as conn:
        row = conn.execute(
            """SELECT p.code, p.name, p.price_month, p.entitlements
               FROM subscription s JOIN plan p ON p.code = s.plan_code
               WHERE s.organization_id = %s""",
            (org_id,),
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT code, name, price_month, entitlements FROM plan WHERE code=%s",
                (DEFAULT_PLAN,),
            ).fetchone()
    ent = dict(row["entitlements"])
    ent["_plan"] = row["code"]
    ent["_plan_name"] = row["name"]
    ent["_price_month"] = float(row["price_month"])
    return ent


def ai_messages_used(org_id: str) -> int:
    with tenant_tx(org_id) as conn:
        row = conn.execute(
            """SELECT count(*) AS n FROM usage_event
               WHERE kind='ai_message' AND created_at >= date_trunc('month', now())"""
        ).fetchone()
    return int(row["n"])


def check_ai_quota(org_id: str) -> dict:
    ent = resolve_entitlements(org_id)
    limit = int(ent.get("ai_messages_month", 0))
    used = ai_messages_used(org_id)
    return {"allowed": used < limit, "used": used, "limit": limit, "plan": ent["_plan"]}


def channel_allowed(org_id: str, kind: str) -> bool:
    ent = resolve_entitlements(org_id)
    allowed = ent.get("channels_allowed", ["website"])
    return kind in allowed


class ChangePlan(BaseModel):
    plan_code: str


@router.get("/plans")
def list_plans():
    with no_tenant() as conn:
        rows = conn.execute(
            "SELECT code, name, price_month, entitlements FROM plan ORDER BY price_month"
        ).fetchall()
    return [dict(r, price_month=float(r["price_month"])) for r in rows]


@router.get("/subscription")
def get_subscription(ctx: OrgContext = Depends(get_org_context)):
    ent = resolve_entitlements(ctx.org_id)
    quota = check_ai_quota(ctx.org_id)
    return {"entitlements": ent, "quota": quota}


@router.post("/subscription")
def change_plan(body: ChangePlan, ctx: OrgContext = Depends(require_role("owner"))):
    # Direct plan change (used for free/downgrade). Paid upgrades go through checkout.
    with no_tenant() as conn:
        plan = conn.execute("SELECT code, price_month FROM plan WHERE code=%s", (body.plan_code,)).fetchone()
        if not plan:
            raise bad_request("unknown plan")
        conn.execute(
            """INSERT INTO subscription (organization_id, plan_code)
               VALUES (%s,%s)
               ON CONFLICT (organization_id)
               DO UPDATE SET plan_code = EXCLUDED.plan_code, status='active'""",
            (ctx.org_id, body.plan_code),
        )
    return {"ok": True, "plan": body.plan_code}


class Checkout(BaseModel):
    plan_code: str


@router.post("/billing/checkout")
def checkout(body: Checkout, ctx: OrgContext = Depends(require_role("owner"))):
    with no_tenant() as conn:
        plan = conn.execute("SELECT code, price_month FROM plan WHERE code=%s", (body.plan_code,)).fetchone()
    if not plan:
        raise bad_request("unknown plan")
    prov = get_payment()
    with tenant_tx(ctx.org_id) as conn:
        inv = conn.execute(
            """INSERT INTO invoice (organization_id, plan_code, amount, currency, provider)
               VALUES (%s,%s,%s,'USD',%s) RETURNING id""",
            (ctx.org_id, plan["code"], plan["price_month"], prov.name),
        ).fetchone()
        invoice_id = str(inv["id"])
    co = prov.create_checkout(invoice_id=invoice_id, amount=float(plan["price_month"]),
                              currency="USD", plan_code=plan["code"])
    if co.get("external_ref"):
        with tenant_tx(ctx.org_id) as conn:
            conn.execute("UPDATE invoice SET external_ref=%s WHERE id=%s", (co["external_ref"], invoice_id))
    audit.record("billing.checkout", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=invoice_id, detail={"plan": plan["code"], "provider": prov.name})
    return {"invoice_id": invoice_id, "amount": float(plan["price_month"]), "plan": plan["code"], **co}


def _activate_invoice(invoice_id: str, provider: str = "stripe") -> bool:
    """Mark an invoice paid and activate its subscription (webhook / callback path,
    no org context — uses the superuser connection). Returns True if activated."""
    from ..db import admin_tx
    with admin_tx() as conn:
        inv = conn.execute(
            "SELECT organization_id, plan_code, amount, currency, status FROM invoice WHERE id=%s", (invoice_id,)
        ).fetchone()
        if not inv or inv["status"] == "paid":
            return False
        conn.execute("UPDATE invoice SET status='paid', paid_at=now() WHERE id=%s", (invoice_id,))
        conn.execute(
            """INSERT INTO payment (organization_id, invoice_id, amount, currency, provider, status)
               VALUES (%s,%s,%s,%s,%s,'succeeded')""",
            (inv["organization_id"], invoice_id, inv["amount"], inv["currency"], provider),
        )
        conn.execute(
            """INSERT INTO subscription (organization_id, plan_code, provider) VALUES (%s,%s,%s)
               ON CONFLICT (organization_id)
               DO UPDATE SET plan_code=EXCLUDED.plan_code, status='active', provider=EXCLUDED.provider""",
            (inv["organization_id"], inv["plan_code"], provider),
        )
    return True


@router.post("/billing/webhook/stripe")
async def stripe_webhook(request: Request):
    import hashlib
    import hmac
    import json as _json
    from ..providers.payment import stripe_webhook_secret
    body = await request.body()
    secret = stripe_webhook_secret()
    if secret:
        header = request.headers.get("stripe-signature", "")
        parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
        signed = f"{parts.get('t','')}.{body.decode()}"
        expected = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, parts.get("v1", "")):
            return {"ok": False, "error": "bad signature"}
    event = _json.loads(body or b"{}")
    if event.get("type") == "checkout.session.completed":
        obj = event.get("data", {}).get("object", {})
        invoice_id = (obj.get("metadata") or {}).get("invoice_id") or obj.get("client_reference_id")
        if invoice_id:
            _activate_invoice(invoice_id)
    return {"ok": True}


def _web_base() -> str:
    from ..config import config as _cfg
    origins = getattr(_cfg, "CORS_ORIGINS", []) or []
    for o in origins:
        if o and o != "*":
            return o.rstrip("/")
    return "http://localhost:3000"


@router.get("/billing/return/vnpay")
def vnpay_return(request: Request):
    """Customer is redirected here by VNPay after paying. Verify + activate, then
    bounce back to the web app with a status flag."""
    from fastapi.responses import RedirectResponse
    from ..providers.payment import get_payment
    params = dict(request.query_params)
    prov = get_payment()
    ok = getattr(prov, "verify", lambda p: False)(params) if prov.name == "vnpay" else False
    if ok:
        _activate_invoice(params.get("vnp_TxnRef", ""), provider="vnpay")
    return RedirectResponse(url=f"{_web_base()}/?paid={'1' if ok else '0'}", status_code=302)


@router.post("/billing/ipn/vnpay")
async def vnpay_ipn(request: Request):
    """Server-to-server confirmation (source of truth for VNPay)."""
    from ..providers.payment import get_payment
    params = dict(request.query_params)
    prov = get_payment()
    if prov.name != "vnpay":
        return {"RspCode": "99", "Message": "gateway not active"}
    if not prov.verify(params):
        return {"RspCode": "97", "Message": "Invalid signature"}
    _activate_invoice(params.get("vnp_TxnRef", ""), provider="vnpay")
    return {"RspCode": "00", "Message": "Confirm Success"}


@router.post("/billing/ipn/momo")
async def momo_ipn(request: Request):
    """MoMo IPN (source of truth). Verify signature, then activate."""
    import json as _json
    from ..providers.payment import get_payment
    body = await request.body()
    params = _json.loads(body or b"{}")
    prov = get_payment()
    if prov.name != "momo":
        return {"ok": False, "error": "gateway not active"}
    if not prov.verify(params):
        return {"ok": False, "error": "bad signature"}
    _activate_invoice(str(params.get("orderId", "")), provider="momo")
    return {"ok": True}


@router.post("/billing/checkout/{invoice_id}/confirm")
def confirm_payment(invoice_id: str, ctx: OrgContext = Depends(require_role("owner"))):
    with tenant_tx(ctx.org_id) as conn:
        inv = conn.execute(
            "SELECT id, plan_code, amount, currency, status FROM invoice WHERE id=%s", (invoice_id,)
        ).fetchone()
        if not inv:
            raise not_found("invoice not found")
        if inv["status"] != "paid":
            conn.execute("UPDATE invoice SET status='paid', paid_at=now() WHERE id=%s", (invoice_id,))
            conn.execute(
                """INSERT INTO payment (organization_id, invoice_id, amount, currency, provider, status)
                   VALUES (%s,%s,%s,%s,'manual','succeeded')""",
                (ctx.org_id, invoice_id, inv["amount"], inv["currency"]),
            )
    with no_tenant() as conn:
        conn.execute(
            """INSERT INTO subscription (organization_id, plan_code, provider) VALUES (%s,%s,'manual')
               ON CONFLICT (organization_id)
               DO UPDATE SET plan_code=EXCLUDED.plan_code, status='active', provider='manual'""",
            (ctx.org_id, inv["plan_code"]),
        )
    audit.record("billing.paid", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=invoice_id, detail={"plan": inv["plan_code"]})
    from ..providers.email import send_safe
    send_safe(ctx.user.email, "Xác nhận thanh toán OmniShop AI",
              f"<p>Cảm ơn bạn! Gói <b>{inv['plan_code']}</b> đã được kích hoạt. "
              f"Số tiền: {inv['amount']} {inv['currency']}.</p>")
    return {"ok": True, "plan": inv["plan_code"]}


@router.get("/billing/invoices")
def list_invoices(ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        rows = conn.execute(
            """SELECT id, plan_code, amount, currency, status, provider, created_at, paid_at
               FROM invoice ORDER BY created_at DESC LIMIT 50"""
        ).fetchall()
    return [{"id": str(r["id"]), "plan": r["plan_code"], "amount": float(r["amount"]),
             "currency": r["currency"], "status": r["status"], "provider": r["provider"],
             "created_at": r["created_at"].isoformat(),
             "paid_at": r["paid_at"].isoformat() if r["paid_at"] else None} for r in rows]
