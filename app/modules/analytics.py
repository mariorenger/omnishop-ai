"""Analytics for the tenant dashboard (bot monitoring) and the platform admin."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..db import admin_tx, tenant_tx
from ..tenancy import CurrentUser, OrgContext, get_org_context, require_platform_admin

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics/overview")
def overview(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        totals = conn.execute(
            """
            WITH msg AS (
              SELECT m.role, m.created_at, m.meta
              FROM message m JOIN conversation c ON c.id = m.conversation_id
              WHERE c.shop_id = %s
            )
            SELECT
              (SELECT count(*) FROM conversation WHERE shop_id=%s) AS conversations,
              (SELECT count(*) FROM conversation WHERE shop_id=%s AND status IN ('needs_human','human')) AS handoff,
              count(*) FILTER (WHERE role='ai') AS ai_messages,
              count(*) FILTER (WHERE role='agent') AS human_replies,
              count(*) FILTER (WHERE role='customer') AS customer_messages
            FROM msg
            """,
            (shop_id, shop_id, shop_id),
        ).fetchone()
        cost = conn.execute(
            """SELECT coalesce(sum(estimated_cost),0) AS cost, coalesce(avg(latency_ms),0) AS lat
               FROM usage_event WHERE shop_id=%s AND kind='ai_message'
                 AND created_at >= date_trunc('month', now())""",
            (shop_id,),
        ).fetchone()
        series = conn.execute(
            """
            SELECT to_char(d.day,'DD/MM') AS day,
                   coalesce(count(*) FILTER (WHERE m.role='ai'),0) AS ai,
                   coalesce(count(*) FILTER (WHERE m.role='agent'),0) AS human,
                   coalesce(count(*) FILTER (WHERE m.role='customer'),0) AS customer
            FROM generate_series(current_date - interval '13 day', current_date, interval '1 day') d(day)
            LEFT JOIN conversation c ON c.shop_id=%s
            LEFT JOIN message m ON m.conversation_id=c.id AND date_trunc('day', m.created_at)=d.day
            GROUP BY d.day ORDER BY d.day
            """,
            (shop_id,),
        ).fetchall()
        intents = conn.execute(
            """SELECT coalesce(m.meta->>'intent','khác') AS intent, count(*) AS n
               FROM message m JOIN conversation c ON c.id=m.conversation_id
               WHERE c.shop_id=%s AND m.role='ai' AND m.created_at >= now() - interval '30 day'
               GROUP BY 1 ORDER BY 2 DESC LIMIT 6""",
            (shop_id,),
        ).fetchall()
    ai = int(totals["ai_messages"]); human = int(totals["human_replies"])
    ai_rate = round(ai / (ai + human) * 100) if (ai + human) else 0
    return {
        "totals": {
            "conversations": int(totals["conversations"]),
            "handoff": int(totals["handoff"]),
            "ai_messages": ai,
            "human_replies": human,
            "customer_messages": int(totals["customer_messages"]),
            "ai_rate": ai_rate,
            "avg_latency_ms": int(cost["lat"]),
            "cost_month": float(cost["cost"]),
        },
        "series": [{"day": r["day"], "ai": int(r["ai"]), "human": int(r["human"]), "customer": int(r["customer"])} for r in series],
        "intents": [{"intent": r["intent"], "count": int(r["n"])} for r in intents],
    }


@router.get("/admin/analytics")
def admin_analytics(_: CurrentUser = Depends(require_platform_admin)):
    with admin_tx() as conn:
        series = conn.execute(
            """
            SELECT to_char(d.day,'DD/MM') AS day,
                   coalesce(count(u.*) FILTER (WHERE u.kind='ai_message'),0) AS ai_messages,
                   coalesce(sum(u.estimated_cost),0) AS cost
            FROM generate_series(current_date - interval '13 day', current_date, interval '1 day') d(day)
            LEFT JOIN usage_event u ON date_trunc('day', u.created_at)=d.day
            GROUP BY d.day ORDER BY d.day
            """
        ).fetchall()
    return {"series": [{"day": r["day"], "ai_messages": int(r["ai_messages"]), "cost": float(r["cost"])} for r in series]}
