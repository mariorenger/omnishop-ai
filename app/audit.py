"""Append-only audit log for privileged actions (ADR-008, risk R-06/R-17)."""
from __future__ import annotations
import json
from typing import Optional

from .db import no_tenant


def record(
    action: str,
    *,
    organization_id: Optional[str] = None,
    actor_user_id: Optional[str] = None,
    target: Optional[str] = None,
    detail: Optional[dict] = None,
    correlation_id: Optional[str] = None,
) -> None:
    # audit_log has no RLS; safe to write via the app pool without org context.
    with no_tenant() as conn:
        conn.execute(
            """INSERT INTO audit_log (organization_id, actor_user_id, action, target, detail, correlation_id)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (organization_id, actor_user_id, action, target, json.dumps(detail or {}), correlation_id),
        )
