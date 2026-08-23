"""In-memory fake channel for tests (Milestone 3). Captures sent messages."""
from __future__ import annotations
from typing import List

from .base import ChannelEvent, OutboundMessage


class FakeProvider:
    kind = "fake"

    def __init__(self):
        self.sent: List[tuple[str, str]] = []

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        return True

    def normalize(self, body: dict) -> ChannelEvent:
        return ChannelEvent(customer_ref=str(body.get("customer_ref", "t")), text=str(body.get("text", "")))

    def send(self, channel_config: dict, customer_ref: str, msg: OutboundMessage) -> None:
        self.sent.append((customer_ref, msg.text))
