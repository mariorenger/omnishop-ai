"""QueueProvider — Valkey-backed job queue (ADR-006 MVP; BullMQ/Temporal later).

enqueue() pushes a JSON job onto a Valkey list and mirrors it into the `job`
table for observability/retry. The worker (app/worker.py) BRPOPs and processes.
"""
from __future__ import annotations
import json
from typing import Optional

import redis

from ..config import config
from ..db import no_tenant

QUEUE_KEY = "omnishop:jobs"

_client: Optional[redis.Redis] = None


def client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
    return _client


def enqueue(kind: str, payload: dict, organization_id: Optional[str] = None) -> str:
    with no_tenant() as conn:
        row = conn.execute(
            "INSERT INTO job (organization_id, kind, payload) VALUES (%s,%s,%s) RETURNING id",
            (organization_id, kind, json.dumps(payload)),
        ).fetchone()
        job_id = str(row["id"])
    msg = {"job_id": job_id, "kind": kind, "payload": payload, "organization_id": organization_id}
    client().lpush(QUEUE_KEY, json.dumps(msg))
    return job_id


def pop(timeout: int = 5) -> Optional[dict]:
    res = client().brpop(QUEUE_KEY, timeout=timeout)
    if not res:
        return None
    _, raw = res
    return json.loads(raw)


def mark(job_id: str, status: str, error: Optional[str] = None) -> None:
    with no_tenant() as conn:
        conn.execute(
            "UPDATE job SET status=%s, error=%s, updated_at=now() WHERE id=%s",
            (status, error, job_id),
        )
