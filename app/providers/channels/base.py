"""ChannelProvider abstraction (architecture §4). Platform-specific logic lives
behind this interface; the conversation/RAG/LLM core only sees canonical events.

MVP ships WebsiteWidgetProvider (works today) + FakeProvider (tests). Meta,
TikTok Shop and Shopee are added as new providers without touching the core.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ChannelEvent:
    """Canonical inbound event, normalized from any platform payload."""
    customer_ref: str
    text: str
    raw: dict = field(default_factory=dict)


@dataclass
class OutboundMessage:
    text: str


class ChannelProvider(Protocol):
    kind: str
    def verify_webhook(self, headers: dict, body: bytes) -> bool: ...
    def normalize(self, body: dict) -> ChannelEvent: ...
    def send(self, channel_config: dict, customer_ref: str, msg: OutboundMessage) -> None: ...
