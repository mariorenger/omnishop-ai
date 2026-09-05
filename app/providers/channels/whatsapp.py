"""WhatsApp Cloud API channel (Meta Graph). Shares the platform Facebook App and
the same webhook endpoint as Messenger/Instagram (object = 'whatsapp_business_
account'). Going live needs a WhatsApp Business Account + a permanent access
token. Routing key: phone_number_id.

- outbound: POST /{phone_number_id}/messages
- inbound: entry[].changes[].value.messages[].{from, text.body};
  value.metadata.phone_number_id identifies the channel.
"""
from __future__ import annotations
from typing import Tuple

import httpx

GRAPH = "https://graph.facebook.com/v21.0"


def verify_token(access_token: str, phone_number_id: str) -> Tuple[bool, str]:
    try:
        r = httpx.get(f"{GRAPH}/{phone_number_id}",
                      params={"fields": "display_phone_number,verified_name", "access_token": access_token},
                      timeout=20)
        if r.status_code != 200:
            return False, r.json().get("error", {}).get("message", f"HTTP {r.status_code}")
        d = r.json()
        return True, d.get("verified_name", d.get("display_phone_number", "connected"))
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def send_message(access_token: str, phone_number_id: str, to: str, text: str) -> Tuple[bool, str]:
    try:
        r = httpx.post(
            f"{GRAPH}/{phone_number_id}/messages",
            params={"access_token": access_token},
            json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}},
            timeout=30,
        )
        return (r.status_code == 200), ("sent" if r.status_code == 200 else r.text)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def normalize_entries(payload: dict):
    """Yield (phone_number_id, sender, text, name, ts) from a WhatsApp Cloud
    webhook body. `name` is the sender's profile name when present, else ''.
    ts is the message's unix time in seconds (used to skip stale backlog)."""
    for entry in payload.get("entry", []):
        for ch in entry.get("changes", []):
            value = ch.get("value", {})
            pnid = str((value.get("metadata") or {}).get("phone_number_id", ""))
            names = {str(c.get("wa_id", "")): ((c.get("profile") or {}).get("name") or "")
                     for c in value.get("contacts", [])}
            for m in value.get("messages", []):
                sender = str(m.get("from", ""))
                text = (m.get("text") or {}).get("body")
                ts = m.get("timestamp")
                if pnid and sender and text:
                    yield pnid, sender, text, names.get(sender, ""), ts
