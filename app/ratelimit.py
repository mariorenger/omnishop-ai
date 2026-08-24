"""Simple Valkey-backed rate limiting (fixed window). Fail-open if cache is down
so a cache outage never takes the app down."""
from __future__ import annotations

from fastapi import HTTPException

from .providers.queue import client


def check(key: str, limit: int, window_s: int) -> None:
    """Raise 429 if `key` exceeds `limit` requests within `window_s`."""
    try:
        c = client()
        rk = f"rl:{key}"
        n = c.incr(rk)
        if n == 1:
            c.expire(rk, window_s)
        if int(n) > limit:
            raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu, vui lòng thử lại sau.")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — cache unavailable: fail open
        return
