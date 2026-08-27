"""Platform admin control plane (cross-tenant). Read-only overview of tenants,
usage and cost for operability (architecture §; risk register). Uses the
superuser pool (RLS bypassed) and is gated to platform admins only."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..db import admin_tx
from ..tenancy import CurrentUser, require_platform_reader

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
