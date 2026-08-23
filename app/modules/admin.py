"""Platform admin control plane (cross-tenant). Read-only overview of tenants,
usage and cost for operability (architecture §; risk register). Uses the
superuser pool (RLS bypassed) and is gated to platform admins only."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..db import admin_tx
from ..tenancy import CurrentUser, require_platform_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/tenants")
def tenants(_: CurrentUser = Depends(require_platform_admin)):
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
def overview(_: CurrentUser = Depends(require_platform_admin)):
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
