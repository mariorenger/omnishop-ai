"""Platform admin control plane (cross-tenant). Read-only overview of tenants,
usage and cost for operability (architecture §; risk register). Uses the
superuser pool (RLS bypassed) and is gated to platform admins only."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import audit
from ..db import admin_tx
from ..errors import bad_request, not_found
from ..tenancy import CurrentUser, require_platform_admin, require_platform_reader

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/tenants")
def tenants(_: CurrentUser = Depends(require_platform_reader)):
    with admin_tx() as conn:
        rows = conn.execute(
            """
            SELECT o.id, o.name, o.status,
                   coalesce(s.plan_code,'free') AS plan,
                   (SELECT count(*) FROM shop sh WHERE sh.organization_id=o.id) AS shops,
                   (SELECT count(*) FROM usage_event u
                      WHERE u.organization_id=o.id AND u.kind='ai_message'
                        AND u.created_at >= date_trunc('month', now())) AS ai_messages,
                   (SELECT coalesce(sum(estimated_cost),0) FROM usage_event u
                      WHERE u.organization_id=o.id
                        AND u.created_at >= date_trunc('month', now())) AS cost_month
            FROM organization o
            LEFT JOIN subscription s ON s.organization_id=o.id
            ORDER BY o.created_at DESC
            """
        ).fetchall()
    return [{"id": str(r["id"]), "name": r["name"], "status": r["status"], "plan": r["plan"],
             "shops": int(r["shops"]), "ai_messages": int(r["ai_messages"]),
             "cost_month": float(r["cost_month"])} for r in rows]


@router.get("/overview")
def overview(_: CurrentUser = Depends(require_platform_reader)):
    with admin_tx() as conn:
        row = conn.execute(
            """SELECT
                 (SELECT count(*) FROM organization) AS tenants,
                 (SELECT count(*) FROM shop) AS shops,
                 (SELECT count(*) FROM conversation) AS conversations,
                 (SELECT count(*) FROM usage_event WHERE kind='ai_message'
                    AND created_at >= date_trunc('month', now())) AS ai_messages_month,
                 (SELECT coalesce(sum(estimated_cost),0) FROM usage_event
                    WHERE created_at >= date_trunc('month', now())) AS cost_month"""
        ).fetchone()
    return {"tenants": int(row["tenants"]), "shops": int(row["shops"]),
            "conversations": int(row["conversations"]),
            "ai_messages_month": int(row["ai_messages_month"]),
            "cost_month": float(row["cost_month"])}


def _csv_response(header, rows, filename):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/reports/tenants.csv")
def export_tenants_csv(_: CurrentUser = Depends(require_platform_reader)):
    """Per-tenant status report (this month) — managers can export for reporting."""
    with admin_tx() as conn:
        rows = conn.execute(
            """SELECT o.name, o.status, coalesce(s.plan_code,'free') AS plan,
                      (SELECT count(*) FROM shop sh WHERE sh.organization_id=o.id) AS shops,
                      (SELECT count(*) FROM usage_event u WHERE u.organization_id=o.id AND u.kind='ai_message'
                         AND u.created_at >= date_trunc('month', now())) AS ai_messages,
                      (SELECT coalesce(sum(input_tokens+output_tokens),0) FROM usage_event u
                         WHERE u.organization_id=o.id AND u.created_at >= date_trunc('month', now())) AS tokens,
                      (SELECT coalesce(sum(estimated_cost),0) FROM usage_event u
                         WHERE u.organization_id=o.id AND u.created_at >= date_trunc('month', now())) AS cost
               FROM organization o LEFT JOIN subscription s ON s.organization_id=o.id
               ORDER BY cost DESC"""
        ).fetchall()
    data = [[r["name"], r["status"], r["plan"], int(r["shops"]), int(r["ai_messages"]),
             int(r["tokens"]), round(float(r["cost"]), 6)] for r in rows]
    return _csv_response(["tenant", "status", "plan", "shops", "ai_messages", "tokens", "cost_usd"],
                         data, "omnishop-tenants.csv")


@router.get("/reports/usage.csv")
def export_usage_csv(_: CurrentUser = Depends(require_platform_reader)):
    """Daily platform usage for the last 30 days."""
    with admin_tx() as conn:
        rows = conn.execute(
            """SELECT date_trunc('day', created_at)::date AS day,
                      count(*) FILTER (WHERE kind='ai_message') AS ai_messages,
                      coalesce(sum(input_tokens+output_tokens),0) AS tokens,
                      coalesce(sum(estimated_cost),0) AS cost
               FROM usage_event WHERE created_at >= now() - interval '30 days'
               GROUP BY 1 ORDER BY 1"""
        ).fetchall()
    data = [[str(r["day"]), int(r["ai_messages"]), int(r["tokens"]), round(float(r["cost"]), 6)] for r in rows]
    return _csv_response(["day", "ai_messages", "tokens", "cost_usd"], data, "omnishop-usage.csv")


@router.get("/finance")
def finance(_: CurrentUser = Depends(require_platform_reader)):
    """Platform P&L: revenue collected (paid invoices) vs AI cost (COGS), profit
    and margin; input/output tokens + cost per model; and revenue-vs-cost per
    tenant — so the operator can see whether the business is profitable."""
    with admin_tx() as conn:
        # Revenue excludes provider='admin_manual' — those are plans an admin
        # granted (comped), so they must not count as profit.
        rev = conn.execute(
            """SELECT
                 coalesce(sum(amount) FILTER (WHERE status='paid' AND provider<>'admin_manual' AND coalesce(paid_at,created_at) >= date_trunc('month', now())),0) AS rev_month,
                 coalesce(sum(amount) FILTER (WHERE status='paid' AND provider<>'admin_manual'),0) AS rev_all,
                 coalesce(sum(amount) FILTER (WHERE status in ('pending','submitted')),0) AS pending,
                 count(*) FILTER (WHERE status='paid' AND provider<>'admin_manual' AND coalesce(paid_at,created_at) >= date_trunc('month', now())) AS paid_month,
                 coalesce(sum(amount) FILTER (WHERE provider='admin_manual' AND coalesce(paid_at,created_at) >= date_trunc('month', now())),0) AS comped_month
               FROM invoice"""
        ).fetchone()
        cost = conn.execute(
            """SELECT
                 coalesce(sum(estimated_cost) FILTER (WHERE created_at >= date_trunc('month', now())),0) AS cost_month,
                 coalesce(sum(estimated_cost),0) AS cost_all
               FROM usage_event"""
        ).fetchone()
        by_model = conn.execute(
            """SELECT coalesce(nullif(llm_model,''),'—') AS model,
                      count(*) AS messages,
                      coalesce(sum(input_tokens),0) AS input_tokens,
                      coalesce(sum(output_tokens),0) AS output_tokens,
                      coalesce(sum(estimated_cost),0) AS cost
               FROM usage_event
               WHERE kind='ai_message' AND created_at >= date_trunc('month', now())
               GROUP BY 1 ORDER BY cost DESC"""
        ).fetchall()
        by_tenant = conn.execute(
            """SELECT o.id, o.name, coalesce(s.plan_code,'free') AS plan,
                      (SELECT coalesce(sum(amount),0) FROM invoice i
                         WHERE i.organization_id=o.id AND i.status='paid' AND i.provider<>'admin_manual'
                           AND coalesce(i.paid_at,i.created_at) >= date_trunc('month', now())) AS revenue,
                      (SELECT coalesce(sum(estimated_cost),0) FROM usage_event u
                         WHERE u.organization_id=o.id AND u.created_at >= date_trunc('month', now())) AS cost,
                      (SELECT coalesce(sum(input_tokens),0) FROM usage_event u
                         WHERE u.organization_id=o.id AND u.created_at >= date_trunc('month', now())) AS input_tokens,
                      (SELECT coalesce(sum(output_tokens),0) FROM usage_event u
                         WHERE u.organization_id=o.id AND u.created_at >= date_trunc('month', now())) AS output_tokens
               FROM organization o LEFT JOIN subscription s ON s.organization_id=o.id
               ORDER BY revenue DESC, cost DESC"""
        ).fetchall()
    rev_month = float(rev["rev_month"]); cost_month = float(cost["cost_month"])
    return {
        "revenue_month": round(rev_month, 2), "revenue_all": round(float(rev["rev_all"]), 2),
        "pending": round(float(rev["pending"]), 2), "paid_invoices_month": int(rev["paid_month"]),
        "comped_month": round(float(rev["comped_month"]), 2),
        "cost_month": round(cost_month, 4), "cost_all": round(float(cost["cost_all"]), 4),
        "profit_month": round(rev_month - cost_month, 2),
        "margin_month": round((rev_month - cost_month) / rev_month * 100, 1) if rev_month > 0 else None,
        "by_model": [{"model": r["model"], "messages": int(r["messages"]),
                      "input_tokens": int(r["input_tokens"]), "output_tokens": int(r["output_tokens"]),
                      "cost": round(float(r["cost"]), 4)} for r in by_model],
        "by_tenant": [{"id": str(r["id"]), "name": r["name"], "plan": r["plan"],
                       "revenue": round(float(r["revenue"]), 2), "cost": round(float(r["cost"]), 4),
                       "input_tokens": int(r["input_tokens"]), "output_tokens": int(r["output_tokens"]),
                       "profit": round(float(r["revenue"]) - float(r["cost"]), 2)} for r in by_tenant],
    }


@router.get("/reports/finance.csv")
def export_finance_csv(_: CurrentUser = Depends(require_platform_reader)):
    """Per-model token & cost breakdown (this month) for finance reporting."""
    with admin_tx() as conn:
        rows = conn.execute(
            """SELECT coalesce(nullif(llm_model,''),'—') AS model, count(*) AS messages,
                      coalesce(sum(input_tokens),0) AS input_tokens,
                      coalesce(sum(output_tokens),0) AS output_tokens,
                      coalesce(sum(estimated_cost),0) AS cost
               FROM usage_event WHERE kind='ai_message' AND created_at >= date_trunc('month', now())
               GROUP BY 1 ORDER BY cost DESC"""
        ).fetchall()
    data = [[r["model"], int(r["messages"]), int(r["input_tokens"]), int(r["output_tokens"]),
             round(float(r["cost"]), 6)] for r in rows]
    return _csv_response(["model", "ai_messages", "input_tokens", "output_tokens", "cost_usd"],
                         data, "omnishop-finance-by-model.csv")


@router.get("/audit")
def audit_log(limit: int = 100, _: CurrentUser = Depends(require_platform_reader)):
    """Recent privileged actions (who did what, when) — admins & managers can view."""
    with admin_tx() as conn:
        rows = conn.execute(
            """SELECT a.action, a.target, a.detail, a.created_at, u.email AS actor
               FROM audit_log a LEFT JOIN app_user u ON u.id = a.actor_user_id
               ORDER BY a.created_at DESC LIMIT %s""",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [{"action": r["action"], "target": r["target"], "detail": r["detail"],
             "actor": r["actor"], "created_at": r["created_at"].isoformat()} for r in rows]


# ============================ admin: tenant plan management ==================
# Managing which plan each tenant is on, confirming manual/QR payments, and
# granting a plan directly (admin_manual — comped, excluded from revenue).

@router.get("/billing/tenants")
def billing_tenants(limit: int = 25, offset: int = 0, q: str = "",
                    _: CurrentUser = Depends(require_platform_admin)):
    """Paginated tenant list for plan management: current plan + this month's
    revenue + how many invoices are awaiting confirmation."""
    limit = max(1, min(limit, 100)); offset = max(0, offset)
    like = f"%{q.strip().lower()}%" if q.strip() else None
    with admin_tx() as conn:
        where = "WHERE lower(o.name) LIKE %s" if like else ""
        params_total = (like,) if like else ()
        total = conn.execute(f"SELECT count(*) AS n FROM organization o {where}", params_total).fetchone()["n"]
        params = (params_total + (limit, offset)) if like else (limit, offset)
        rows = conn.execute(
            f"""SELECT o.id, o.name, o.status, coalesce(s.plan_code,'free') AS plan, s.provider AS plan_provider,
                       (SELECT count(*) FROM invoice i WHERE i.organization_id=o.id AND i.status in ('pending','submitted')) AS pending,
                       (SELECT coalesce(sum(amount),0) FROM invoice i WHERE i.organization_id=o.id
                          AND i.status='paid' AND i.provider<>'admin_manual'
                          AND coalesce(i.paid_at,i.created_at) >= date_trunc('month', now())) AS revenue_month
                FROM organization o LEFT JOIN subscription s ON s.organization_id=o.id
                {where} ORDER BY o.created_at DESC LIMIT %s OFFSET %s""",
            params,
        ).fetchall()
    items = [{"id": str(r["id"]), "name": r["name"], "status": r["status"], "plan": r["plan"],
              "plan_provider": r["plan_provider"], "pending": int(r["pending"]),
              "revenue_month": round(float(r["revenue_month"]), 2)} for r in rows]
    return {"items": items, "total": int(total), "limit": limit, "offset": offset,
            "has_more": offset + len(items) < int(total)}


class SetPlanBody(BaseModel):
    plan_code: str


@router.put("/tenants/{org_id}/plan")
def admin_set_tenant_plan(org_id: str, body: SetPlanBody, admin: CurrentUser = Depends(require_platform_admin)):
    """Grant a plan to a tenant directly (no payment). Recorded as an
    'admin_manual' invoice so it shows in history but is EXCLUDED from revenue."""
    with admin_tx() as conn:
        plan = conn.execute("SELECT code, price_month FROM plan WHERE code=%s", (body.plan_code,)).fetchone()
        if not plan:
            raise not_found("plan not found")
        if not conn.execute("SELECT 1 FROM organization WHERE id=%s", (org_id,)).fetchone():
            raise not_found("tenant not found")
        conn.execute(
            """INSERT INTO invoice (organization_id, plan_code, amount, currency, status, provider, paid_at)
               VALUES (%s,%s,%s,'USD','paid','admin_manual', now())""",
            (org_id, plan["code"], plan["price_month"]),
        )
        conn.execute(
            """INSERT INTO subscription (organization_id, plan_code, provider) VALUES (%s,%s,'admin_manual')
               ON CONFLICT (organization_id)
               DO UPDATE SET plan_code=EXCLUDED.plan_code, status='active', provider='admin_manual'""",
            (org_id, plan["code"]),
        )
    audit.record("admin.tenant.plan", organization_id=org_id, actor_user_id=admin.id,
                 target=org_id, detail={"plan": plan["code"], "type": "admin_manual"})
    return {"ok": True, "plan": plan["code"], "type": "admin_manual"}


@router.get("/billing/pending")
def billing_pending(limit: int = 25, offset: int = 0, _: CurrentUser = Depends(require_platform_admin)):
    """Invoices awaiting confirmation (tenant reported a bank/QR transfer)."""
    limit = max(1, min(limit, 100)); offset = max(0, offset)
    with admin_tx() as conn:
        total = conn.execute(
            "SELECT count(*) AS n FROM invoice WHERE status in ('pending','submitted')").fetchone()["n"]
        rows = conn.execute(
            """SELECT i.id, i.plan_code, i.amount, i.currency, i.status, i.provider, i.created_at,
                      o.name AS tenant
               FROM invoice i JOIN organization o ON o.id = i.organization_id
               WHERE i.status in ('pending','submitted')
               ORDER BY i.created_at DESC LIMIT %s OFFSET %s""",
            (limit, offset),
        ).fetchall()
    items = [{"id": str(r["id"]), "plan": r["plan_code"], "amount": float(r["amount"]),
              "currency": r["currency"], "status": r["status"], "provider": r["provider"],
              "tenant": r["tenant"], "created_at": r["created_at"].isoformat()} for r in rows]
    return {"items": items, "total": int(total), "limit": limit, "offset": offset,
            "has_more": offset + len(items) < int(total)}


@router.post("/invoices/{invoice_id}/confirm")
def admin_confirm_invoice(invoice_id: str, admin: CurrentUser = Depends(require_platform_admin)):
    """Admin confirms a real bank/QR payment after checking it — activates the
    plan and counts as revenue (keeps the invoice's own provider, e.g. vietqr)."""
    from .billing import _activate_invoice
    with admin_tx() as conn:
        inv = conn.execute("SELECT organization_id, provider, status FROM invoice WHERE id=%s", (invoice_id,)).fetchone()
        if not inv:
            raise not_found("invoice not found")
        if inv["status"] == "paid":
            return {"ok": True, "status": "paid"}
    activated = _activate_invoice(invoice_id, provider=inv["provider"] or "manual")
    audit.record("admin.invoice.confirm", organization_id=str(inv["organization_id"]),
                 actor_user_id=admin.id, target=invoice_id)
    return {"ok": bool(activated), "status": "paid"}
