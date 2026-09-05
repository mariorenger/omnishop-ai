"""Payment gateway crypto — the parts we can verify without a live merchant.
Signatures must accept a genuine callback and reject a tampered one."""
from app.providers.payment import (MoMoPaymentProvider, VietQRPaymentProvider,
                                    VNPayPaymentProvider)


def test_vietqr_quicklink_url_converts_usd_to_vnd():
    # plan price is USD; VietQR charges VND -> amount is converted by usd_vnd
    p = VietQRPaymentProvider({"bank_bin": "970436", "account_no": "0123456789",
                               "account_name": "OMNISHOP JSC", "usd_vnd": 25000})
    co = p.create_checkout(invoice_id="abcd1234", amount=2, currency="USD", plan_code="growth")
    url = co["qr_image_url"]
    assert url.startswith("https://img.vietqr.io/image/970436-0123456789-")
    assert "amount=50000" in url                # 2 USD * 25000 = 50.000 đ
    assert co["amount_vnd"] == 50000


def test_vietqr_missing_account_errors():
    p = VietQRPaymentProvider({})
    co = p.create_checkout(invoice_id="x", amount=1000, currency="VND", plan_code="free")
    assert "error" in co


def test_vnpay_redirect_and_signature_roundtrip():
    p = VNPayPaymentProvider("SECRETHASH", {"tmn_code": "OMNI0001"})
    co = p.create_checkout(invoice_id="inv-1", amount=199000, currency="VND", plan_code="growth")
    assert "vpcpay.html" in co["redirect_url"]      # correct VNPay endpoint
    assert "vnp_SecureHash=" in co["redirect_url"]
    # a genuine return (signed over the full param set, incl. response code) verifies
    ret = {"vnp_TxnRef": "inv-1", "vnp_Amount": "19900000", "vnp_ResponseCode": "00"}
    ret["vnp_SecureHash"] = p._sign(ret)
    assert p.verify(ret) is True
    # tampering the amount breaks the signature
    bad = dict(ret, vnp_Amount="1")
    assert p.verify(bad) is False


def test_momo_ipn_signature_roundtrip():
    import hashlib
    import hmac
    p = MoMoPaymentProvider("MOMOSECRET", {"access_key": "AK"})
    fields = {"partnerCode": "P", "orderId": "o", "requestId": "r", "amount": "199000",
              "orderInfo": "x", "orderType": "momo", "transId": "9", "resultCode": "0",
              "message": "ok", "payType": "qr", "responseTime": "1", "extraData": ""}
    raw = (f"accessKey=AK&amount={fields['amount']}&extraData={fields['extraData']}"
           f"&message={fields['message']}&orderId={fields['orderId']}&orderInfo={fields['orderInfo']}"
           f"&orderType={fields['orderType']}&partnerCode={fields['partnerCode']}"
           f"&payType={fields['payType']}&requestId={fields['requestId']}"
           f"&responseTime={fields['responseTime']}&resultCode={fields['resultCode']}"
           f"&transId={fields['transId']}")
    fields["signature"] = hmac.new(b"MOMOSECRET", raw.encode(), hashlib.sha256).hexdigest()
    assert p.verify(fields) is True
    assert p.verify(dict(fields, amount="1")) is False
