"""Plans, subscription, and the EntitlementService (ADR-004).

Entitlements map plan -> capabilities/quotas; the backend enforces them (never
the frontend). Payments are out of scope for the MVP (subscription.provider =
'manual'); a BillingProvider/PaymentProvider adapter slots in later.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..db import no_tenant, tenant_tx
from ..errors import bad_request
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
def change_plan(body: ChangePlan, ctx: OrgContext = Depends(require_role("admin"))):
    with no_tenant() as conn:
        exists = conn.execute("SELECT 1 FROM plan WHERE code=%s", (body.plan_code,)).fetchone()
        if not exists:
            raise bad_request("unknown plan")
        conn.execute(
            """INSERT INTO subscription (organization_id, plan_code)
               VALUES (%s,%s)
               ON CONFLICT (organization_id)
               DO UPDATE SET plan_code = EXCLUDED.plan_code, status='active'""",
            (ctx.org_id, body.plan_code),
        )
    return {"ok": True, "plan": body.plan_code}
