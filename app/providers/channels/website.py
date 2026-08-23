"""Website chat widget channel. The reply is returned synchronously in the HTTP
response, so `send` is a no-op here; the abstraction still holds for channels
(Meta/Shopee) that push replies out-of-band.
"""
from __future__ import annotations
from .base import ChannelEvent, OutboundMessage


class WebsiteWidgetProvider:
    kind = "website"

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        # Public widget uses the channel public_key in the URL as the shared secret.
        return True

    def normalize(self, body: dict) -> ChannelEvent:
        return ChannelEvent(
            customer_ref=str(body.get("session_id") or "anon"),
            text=str(body.get("text") or ""),
            raw=body,
        )

    def send(self, channel_config: dict, customer_ref: str, msg: OutboundMessage) -> None:
        return None  # reply delivered in the HTTP response
