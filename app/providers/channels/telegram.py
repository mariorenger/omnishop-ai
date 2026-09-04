"""Telegram channel via the Bot API. Fully live with just a bot token from
@BotFather — no partner approval needed.

- token verification: getMe
- outbound: sendMessage
- inbound: Telegram POSTs updates to a per-channel webhook URL we register
  (setWebhook). We route by the channel's public_key embedded in that URL.
"""
from __future__ import annotations
from typing import Tuple

import httpx

API = "https://api.telegram.org"


def verify_token(token: str) -> Tuple[bool, str]:
    try:
        r = httpx.get(f"{API}/bot{token}/getMe", timeout=20)
        if r.status_code != 200 or not r.json().get("ok"):
            return False, r.json().get("description", f"HTTP {r.status_code}")
        u = r.json()["result"]
        return True, "@" + u.get("username", u.get("first_name", "bot"))
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def set_webhook(token: str, url: str) -> Tuple[bool, str]:
    try:
        r = httpx.post(f"{API}/bot{token}/setWebhook", json={"url": url}, timeout=20)
        ok = r.status_code == 200 and r.json().get("ok")
        return bool(ok), r.json().get("description", "ok")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def webhook_info(token: str) -> dict:
    """getWebhookInfo — what URL Telegram will deliver to, and any last error."""
    try:
        r = httpx.get(f"{API}/bot{token}/getWebhookInfo", timeout=20)
        d = r.json().get("result", {}) or {}
        return {"url": d.get("url", ""), "pending": int(d.get("pending_update_count", 0)),
                "last_error": d.get("last_error_message", "")}
    except Exception as e:  # noqa: BLE001
        return {"url": "", "pending": 0, "last_error": str(e)}


def send_message(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    try:
        r = httpx.post(f"{API}/bot{token}/sendMessage",
                       json={"chat_id": chat_id, "text": text}, timeout=30)
        return (r.status_code == 200), ("sent" if r.status_code == 200 else r.text)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def normalize_update(update: dict):
    """Return (chat_id, text) from a Telegram update, or (None, None)."""
    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = msg.get("text")
    if chat_id is not None and text:
        return str(chat_id), text
    return None, None


def sender_name(update: dict) -> str:
    """Best-effort display name of the sender (first/last name or @username)."""
    frm = ((update.get("message") or update.get("edited_message") or {}).get("from")) or {}
    name = " ".join(x for x in (frm.get("first_name"), frm.get("last_name")) if x).strip()
    return name or (("@" + frm["username"]) if frm.get("username") else "")
