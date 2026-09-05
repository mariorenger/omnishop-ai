"""PaymentProvider (ADR-004). The gateway is configured by the platform admin
(provider + keys, stored encrypted). The tenant checkout flow uses whatever the
admin configured — no gateway is hard-coded.

Providers:
- manual : no gateway; the app confirms invoices directly (demo / bank transfer).
- stripe : real Stripe Checkout Session via the HTTP API; confirmed by webhook.
- vietqr : real VietQR quick-link — renders a bank-transfer QR from the merchant's
           bank account (no gateway contract needed). Confirmed manually or by a
           bank-reconciliation webhook.
- vnpay  : real VNPay redirect (HMAC-SHA512 signed); confirmed by return + IPN.
- momo   : real MoMo AIO v2 create-payment (HMAC-SHA256 signed); confirmed by IPN.

Each provider returns a dict from create_checkout with at most one of:
  redirect_url   -> hosted gateway page the customer is sent to (vnpay/momo/stripe)
  qr_image_url   -> a QR image to display in-app (vietqr)
  instructions   -> manual text (manual / not-yet-live)
VN gateways charge in VND; the numeric `amount` is treated as VND for them.
"""
from __future__ import annotations
import hashlib
import hmac
import time
import urllib.parse
from typing import Optional

from .registry import resolve_payment_config


def invoice_code(invoice_id) -> str:
    """Short transfer memo embedded in the bank content so an incoming transfer
    can be matched back to its invoice (used by VietQR + the SePay webhook)."""
    return "OMNI" + str(invoice_id).replace("-", "")[:8].upper()


class ManualPaymentProvider:
    name = "manual"

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        note = invoice_code(invoice_id)
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": None,
                "transfer_note": note,
                "instructions": f"Chuyển khoản theo hướng dẫn của cửa hàng, nội dung ghi: {note}. "
                                "Gói kích hoạt sau khi thanh toán được xác nhận."}


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


class VietQRPaymentProvider:
    """VietQR quick-link. Renders a real bank-transfer QR from the merchant bank
    account — works immediately, no gateway contract. Reconciliation is manual
    (owner confirms) or via a bank webhook (out of scope here)."""
    name = "vietqr"

    def __init__(self, extra: dict):
        self.extra = extra or {}

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        bank = str(self.extra.get("bank_bin") or self.extra.get("bank") or "").strip()
        account = str(self.extra.get("account_no") or "").strip()
        name = str(self.extra.get("account_name") or "").strip()
        template = str(self.extra.get("template") or "compact2").strip()
        if not (bank and account):
            return {"provider": self.name, "invoice_id": invoice_id,
                    "error": "VietQR chưa cấu hình số tài khoản / ngân hàng."}
        amt = int(round(float(amount)))
        info = invoice_code(invoice_id)
        qs = urllib.parse.urlencode({"amount": amt, "addInfo": info, "accountName": name})
        qr_image_url = f"https://img.vietqr.io/image/{bank}-{account}-{template}.png?{qs}"
        return {"provider": self.name, "invoice_id": invoice_id, "qr_image_url": qr_image_url,
                "amount_vnd": amt, "transfer_note": info,
                "instructions": "Quét mã VietQR bằng app ngân hàng để chuyển khoản, "
                                "nội dung giữ nguyên. Gói kích hoạt sau khi xác nhận."}


class VNPayPaymentProvider:
    """Real VNPay redirect (pay.vnpay.vn / sandbox). Params sorted + HMAC-SHA512."""
    name = "vnpay"

    def __init__(self, hash_secret: str, extra: dict):
        self.hash_secret = hash_secret
        self.extra = extra or {}

    def _sign(self, params: dict) -> str:
        data = "&".join(f"{k}={urllib.parse.quote_plus(str(params[k]))}" for k in sorted(params))
        return hmac.new(self.hash_secret.encode(), data.encode(), hashlib.sha512).hexdigest()

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        tmn = str(self.extra.get("tmn_code") or "").strip()
        if not (tmn and self.hash_secret):
            return {"provider": self.name, "invoice_id": invoice_id,
                    "error": "VNPay chưa cấu hình TMN code / hash secret."}
        pay_url = self.extra.get("pay_url") or "https://sandbox.vnpayment.vn/paymentv2/vpcpay.html"
        return_url = self.extra.get("return_url") or "http://localhost:8000/api/billing/return/vnpay"
        params = {
            "vnp_Version": "2.1.0", "vnp_Command": "pay", "vnp_TmnCode": tmn,
            "vnp_Amount": int(round(float(amount) * 100)), "vnp_CurrCode": "VND",
            "vnp_TxnRef": str(invoice_id), "vnp_OrderInfo": f"OmniShop {plan_code}",
            "vnp_OrderType": "other", "vnp_Locale": "vn", "vnp_ReturnUrl": return_url,
            "vnp_IpAddr": "127.0.0.1", "vnp_CreateDate": time.strftime("%Y%m%d%H%M%S"),
        }
        secure = self._sign(params)
        query = "&".join(f"{k}={urllib.parse.quote_plus(str(params[k]))}" for k in sorted(params))
        redirect = f"{pay_url}?{query}&vnp_SecureHash={secure}"
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": redirect}

    def verify(self, params: dict) -> bool:
        got = params.get("vnp_SecureHash", "")
        check = {k: v for k, v in params.items() if k not in ("vnp_SecureHash", "vnp_SecureHashType")}
        expected = self._sign(check)
        return hmac.compare_digest(expected, got) and params.get("vnp_ResponseCode") == "00"


class MoMoPaymentProvider:
    """Real MoMo AIO v2 create-payment. rawSignature is alphabetical; HMAC-SHA256."""
    name = "momo"

    def __init__(self, secret_key: str, extra: dict):
        self.secret_key = secret_key
        self.extra = extra or {}

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        import httpx
        partner = str(self.extra.get("partner_code") or "").strip()
        access = str(self.extra.get("access_key") or "").strip()
        if not (partner and access and self.secret_key):
            return {"provider": self.name, "invoice_id": invoice_id,
                    "error": "MoMo chưa cấu hình partner code / access key / secret."}
        endpoint = self.extra.get("endpoint") or "https://test-payment.momo.vn/v2/gateway/api/create"
        redirect_url = self.extra.get("redirect_url") or "http://localhost:3000/?paid=1"
        ipn_url = self.extra.get("ipn_url") or "http://localhost:8000/api/billing/ipn/momo"
        amt = str(int(round(float(amount))))
        request_id = f"{invoice_id}-{int(time.time())}"
        order_id = str(invoice_id)
        order_info = f"OmniShop {plan_code}"
        raw = (f"accessKey={access}&amount={amt}&extraData=&ipnUrl={ipn_url}"
               f"&orderId={order_id}&orderInfo={order_info}&partnerCode={partner}"
               f"&redirectUrl={redirect_url}&requestId={request_id}&requestType=captureWallet")
        signature = hmac.new(self.secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()
        payload = {
            "partnerCode": partner, "accessKey": access, "requestId": request_id,
            "amount": amt, "orderId": order_id, "orderInfo": order_info,
            "redirectUrl": redirect_url, "ipnUrl": ipn_url, "extraData": "",
            "requestType": "captureWallet", "signature": signature, "lang": "vi",
        }
        try:
            r = httpx.post(endpoint, json=payload, timeout=30)
            data = r.json()
        except Exception as e:  # noqa: BLE001
            return {"provider": self.name, "invoice_id": invoice_id, "error": str(e)}
        if data.get("resultCode") not in (0, "0") or not data.get("payUrl"):
            return {"provider": self.name, "invoice_id": invoice_id,
                    "error": data.get("message", "MoMo create payment failed")}
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": data["payUrl"],
                "external_ref": data.get("requestId")}

    def verify(self, params: dict) -> bool:
        access = str(self.extra.get("access_key") or "").strip()
        got = params.get("signature", "")
        raw = (f"accessKey={access}&amount={params.get('amount','')}"
               f"&extraData={params.get('extraData','')}&message={params.get('message','')}"
               f"&orderId={params.get('orderId','')}&orderInfo={params.get('orderInfo','')}"
               f"&orderType={params.get('orderType','')}&partnerCode={params.get('partnerCode','')}"
               f"&payType={params.get('payType','')}&requestId={params.get('requestId','')}"
               f"&responseTime={params.get('responseTime','')}&resultCode={params.get('resultCode','')}"
               f"&transId={params.get('transId','')}")
        expected = hmac.new(self.secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, got) and str(params.get("resultCode")) == "0"


class ScaffoldPaymentProvider:
    """A configured-but-not-yet-live gateway placeholder."""
    def __init__(self, name: str):
        self.name = name

    def create_checkout(self, *, invoice_id, amount, currency, plan_code) -> dict:
        return {"provider": self.name, "invoice_id": invoice_id, "redirect_url": None,
                "instructions": f"Cổng {self.name} đã cấu hình nhưng chưa bật. Dùng xác nhận thủ công tạm thời."}


def get_payment():
    cfg = resolve_payment_config()
    provider = (cfg.get("provider") or "manual").lower()
    extra = cfg.get("extra") or {}
    if provider == "stripe" and cfg.get("api_key"):
        return StripePaymentProvider(cfg["api_key"], extra)
    if provider == "vietqr":
        return VietQRPaymentProvider(extra)
    if provider == "vnpay":
        return VNPayPaymentProvider(cfg.get("api_key") or "", extra)
    if provider == "momo":
        return MoMoPaymentProvider(cfg.get("api_key") or "", extra)
    return ManualPaymentProvider()


def stripe_webhook_secret() -> str:
    cfg = resolve_payment_config()
    return (cfg.get("extra") or {}).get("webhook_secret", "")
