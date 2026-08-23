"""PaymentProvider (ADR-004). MVP ships a manual/mock provider so the full
checkout flow works end-to-end without a gateway account. Stripe / VNPay / MoMo
plug in behind the same interface (redirect + webhook confirmation).
"""
from __future__ import annotations
from typing import Protocol

from ..config import config


class PaymentProvider(Protocol):
    name: str
    def create_checkout(self, *, invoice_id: str, amount: float, currency: str, plan_code: str) -> dict: ...


class ManualPaymentProvider:
    """No external gateway: the app confirms the invoice directly (demo/manual
    bank transfer). Returns no redirect; the UI shows a confirm step."""
    name = "manual"

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": None,
                "instructions": "Xác nhận thanh toán để kích hoạt gói (demo / chuyển khoản thủ công)."}


def get_payment() -> PaymentProvider:
    # Only 'manual' is implemented in the MVP; real gateways are the documented seam.
    _ = config  # reserved for PAYMENT_PROVIDER selection later
    return ManualPaymentProvider()
