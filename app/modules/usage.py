"""Usage metering & cost accounting (ADR-007, cost model). Every AI call and
embedding batch writes a usage_event so we can compute per-tenant COGS."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends

from ..config import config
from ..db import tenant_tx
from ..tenancy import OrgContext, get_org_context

router = APIRouter(prefix="/api/usage", tags=["usage"])


def estimate_cost(input_tokens: int, output_tokens: int, embedding_tokens: int = 0) -> float:
    return round(
        input_tokens / 1_000_000 * config.COST_INPUT_PER_M
        + output_tokens / 1_000_000 * config.COST_OUTPUT_PER_M
        + embedding_tokens / 1_000_000 * config.COST_EMBEDDING_PER_M,
        6,
    )


def record_ai_message(
    org_id: str,
    *,
    shop_id: Optional[str],
    channel_id: Optional[str],
    conversation_id: Optional[str],
    model: str,
    input_tokens: int,
    output_tokens: int,
    retrieval_count: int,
    latency_ms: int,
) -> float:
    cost = estimate_cost(input_tokens, output_tokens)
    with tenant_tx(org_id) as conn:
        conn.execute(
            """INSERT INTO usage_event
               (organization_id, shop_id, channel_id, conversation_id, kind, llm_model,
                input_tokens, output_tokens, retrieval_count, latency_ms, estimated_cost)
               VALUES (%s,%s,%s,%s,'ai_message',%s,%s,%s,%s,%s,%s)""",
            (org_id, shop_id, channel_id, conversation_id, model,
             input_tokens, output_tokens, retrieval_count, latency_ms, cost),
        )
    return cost


def record_embedding(org_id: str, embedding_tokens: int) -> float:
    cost = estimate_cost(0, 0, embedding_tokens)
    with tenant_tx(org_id) as conn:
        conn.execute(
            """INSERT INTO usage_event (organization_id, kind, embedding_tokens, estimated_cost)
               VALUES (%s,'embedding',%s,%s)""",
            (org_id, embedding_tokens, cost),
        )
    return cost


def summary(org_id: str) -> dict:
    with tenant_tx(org_id) as conn:
        row = conn.execute(
            """SELECT
                 count(*) FILTER (WHERE kind='ai_message') AS ai_messages,
                 coalesce(sum(input_tokens),0) AS input_tokens,
                 coalesce(sum(output_tokens),0) AS output_tokens,
                 coalesce(sum(embedding_tokens),0) AS embedding_tokens,
                 coalesce(sum(estimated_cost),0) AS total_cost
               FROM usage_event
               WHERE created_at >= date_trunc('month', now())"""
        ).fetchone()
    return {k: (float(v) if k == "total_cost" else int(v)) for k, v in row.items()}


@router.get("/summary")
def get_summary(ctx: OrgContext = Depends(get_org_context)):
    return summary(ctx.org_id)
