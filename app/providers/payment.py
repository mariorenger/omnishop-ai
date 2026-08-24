"""PaymentProvider (ADR-004). The gateway is configured by the platform admin
(provider + keys, stored encrypted). The tenant checkout flow uses whatever the
admin configured — no gateway is hard-coded.

- manual : no gateway; the app confirms invoices directly (demo / bank transfer).
- stripe : real Stripe Checkout Session via the HTTP API (needs a secret key);
           confirmation via the Stripe webhook (checkout.session.completed).
- vnpay  : scaffold (VN gateway) — signature-based redirect; enable when keys +
           merchant account are provided.
"""
from __future__ import annotations
from typing import Optional, Tuple

from .registry import resolve_payment_config


class ManualPaymentProvider:
    name = "manual"

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": None,
                "instructions": "Xác nhận thanh toán để kích hoạt gói (chuyển khoản thủ công / demo)."}


class StripePaymentProvider:
    name = "stripe"

    def __init__(self, secret_key: str, extra: dict):
        self.secret_key = secret_key
        self.extra = extra or {}

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        import httpx
        success = self.extra.get("success_url") or "http://localhost:3000/?paid=1"
        cancel = self.extra.get("cancel_url") or "http://localhost:3000/?canceled=1"
        cur = (currency or "usd").lower()
        data = {
            "mode": "payment",
            "success_url": success,
            "cancel_url": cancel,
            "client_reference_id": invoice_id,
            "metadata[invoice_id]": invoice_id,
            "line_items[0][quantity]": "1",
            "line_items[0][price_data][currency]": cur,
            "line_items[0][price_data][unit_amount]": str(int(round(float(amount) * 100))),
            "line_items[0][price_data][product_data][name]": f"OmniShop AI — {plan_code}",
        }
        r = httpx.post("https://api.stripe.com/v1/checkout/sessions", data=data,
                       auth=(self.secret_key, ""), timeout=30)
        if r.status_code >= 300:
            return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": None,
                    "error": r.json().get("error", {}).get("message", r.text)}
        s = r.json()
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": s.get("url"),
                "external_ref": s.get("id")}


class ScaffoldPaymentProvider:
    """A configured-but-not-yet-live gateway (e.g. VNPay/MoMo)."""
    def __init__(self, name: str):
        self.name = name

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": None,
                "instructions": f"Cổng {self.name} đã cấu hình nhưng chưa bật. Dùng xác nhận thủ công tạm thời."}


def get_payment():
    cfg = resolve_payment_config()
    provider = (cfg.get("provider") or "manual").lower()
    if provider == "stripe" and cfg.get("api_key"):
        return StripePaymentProvider(cfg["api_key"], cfg.get("extra") or {})
    if provider in ("vnpay", "momo"):
        return ScaffoldPaymentProvider(provider)
    return ManualPaymentProvider()


def stripe_webhook_secret() -> str:
    cfg = resolve_payment_config()
    return (cfg.get("extra") or {}).get("webhook_secret", "")
