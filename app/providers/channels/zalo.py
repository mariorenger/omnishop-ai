"""Zalo Official Account channel via the Zalo OA Open API (v3).

Real send/verify. Going live needs a verified Official Account + an app the OA
has granted, and OA access tokens must be refreshed periodically (out of scope of
this module — the tenant stores a valid access token). Routing key: oa_id.

- token check: GET /v2.0/oa/getoa
- outbound: POST /v3.0/oa/message/cs  (customer-service message)
- inbound webhook: {app_id, sender:{id}, recipient:{id=oa_id}, message:{text},
  event_name:"user_send_text"}
"""
from __future__ import annotations
from typing import Tuple

import httpx

OPENAPI = "https://openapi.zalo.me"


def verify_token(access_token: str) -> Tuple[bool, str]:
    try:
        r = httpx.get(f"{OPENAPI}/v2.0/oa/getoa", headers={"access_token": access_token}, timeout=20)
        data = r.json()
        if data.get("error") not in (0, None):
            return False, data.get("message", "invalid token")
        return True, (data.get("data") or {}).get("name", "connected")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def send_message(access_token: str, user_id: str, text: str) -> Tuple[bool, str]:
    try:
        r = httpx.post(
            f"{OPENAPI}/v3.0/oa/message/cs",
            headers={"access_token": access_token, "Content-Type": "application/json"},
            json={"recipient": {"user_id": user_id}, "message": {"text": text}},
            timeout=30,
        )
        data = r.json()
        ok = data.get("error") in (0, None)
        return bool(ok), ("sent" if ok else data.get("message", r.text))
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def normalize_event(payload: dict):
    """Return (oa_id, sender_id, text) from a Zalo OA webhook, or (None, None, None)."""
    if payload.get("event_name") not in ("user_send_text", None):
        # only handle inbound text for now
        if payload.get("event_name") != "user_send_text":
            return None, None, None
    oa_id = str((payload.get("recipient") or {}).get("id", "") or payload.get("oa_id", ""))
    sender = str((payload.get("sender") or {}).get("id", ""))
    text = (payload.get("message") or {}).get("text")
    if oa_id and sender and text:
        return oa_id, sender, text
    return None, None, None
