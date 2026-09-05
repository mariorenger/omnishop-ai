"""Meta (Facebook Messenger + Instagram) channel via the Graph API.

Real implementation: token verification, outbound send, and webhook signature
check. It works once the tenant supplies a valid Page/IG access token issued by
the platform's Facebook App (which requires Meta App Review for messaging
permissions). The SaaS owns one Facebook App; tenants connect their Pages to it.

- App-level (platform): APP_SECRET (webhook signature), VERIFY_TOKEN (handshake).
- Channel-level (per tenant): page_id + page_access_token.
"""
from __future__ import annotations
import hashlib
import hmac
from typing import Tuple

import httpx

GRAPH = "https://graph.facebook.com/v21.0"


def verify_page_token(page_access_token: str, page_id: str = "") -> Tuple[bool, str]:
    """Lightweight check that the token is valid; returns (ok, name_or_error)."""
    try:
        r = httpx.get(f"{GRAPH}/me", params={"fields": "id,name", "access_token": page_access_token}, timeout=20)
        if r.status_code != 200:
            return False, r.json().get("error", {}).get("message", f"HTTP {r.status_code}")
        data = r.json()
        return True, data.get("name", data.get("id", "connected"))
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def subscribe_page(page_access_token: str, page_id: str,
                   fields: str = "messages,messaging_postbacks,messaging_optins,message_reactions") -> Tuple[bool, str]:
    """Subscribe the Page to the platform app so Messenger/IG events are delivered
    to our webhook. Without this a valid token still receives nothing."""
    try:
        r = httpx.post(f"{GRAPH}/{page_id}/subscribed_apps",
                       params={"subscribed_fields": fields, "access_token": page_access_token}, timeout=20)
        if r.status_code == 200 and r.json().get("success"):
            return True, "page subscribed"
        return False, r.json().get("error", {}).get("message", r.text)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def send_message(page_access_token: str, recipient_id: str, text: str) -> Tuple[bool, str]:
    try:
        r = httpx.post(
            f"{GRAPH}/me/messages",
            params={"access_token": page_access_token},
            json={"recipient": {"id": recipient_id}, "messaging_type": "RESPONSE", "message": {"text": text}},
            timeout=30,
        )
        if r.status_code != 200:
            return False, r.text
        return True, "sent"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def verify_signature(app_secret: str, body: bytes, header: str) -> bool:
    """Validate X-Hub-Signature-256 = 'sha256=<hmac>'."""
    if not app_secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1])


def normalize_entries(payload: dict):
    """Yield (page_id, sender_id, text, ts) from a Messenger/IG webhook body.
    ts is the event's unix time in SECONDS (Meta sends milliseconds)."""
    for entry in payload.get("entry", []):
        page_id = str(entry.get("id", ""))
        for ev in entry.get("messaging", []):
            sender = str(ev.get("sender", {}).get("id", ""))
            msg = ev.get("message", {})
            text = msg.get("text")
            ts_ms = ev.get("timestamp")
            ts = (float(ts_ms) / 1000.0) if ts_ms else None
            if sender and text:
                yield page_id, sender, text, ts
