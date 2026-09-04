"""Channel webhook parsing + signature verification (no network)."""
import hashlib
import hmac

from app.providers.channels import meta, telegram, whatsapp, zalo


def test_telegram_normalize_update():
    chat_id, text = telegram.normalize_update(
        {"message": {"chat": {"id": 12345}, "text": "xin chào"}})
    assert chat_id == "12345" and text == "xin chào"
    # non-text update -> ignored
    assert telegram.normalize_update({"message": {"chat": {"id": 1}}}) == (None, None)


def test_whatsapp_normalize_entries():
    payload = {"object": "whatsapp_business_account", "entry": [
        {"changes": [{"value": {"metadata": {"phone_number_id": "PN1"},
                                 "contacts": [{"wa_id": "849xx", "profile": {"name": "Chị Lan"}}],
                                 "messages": [{"from": "849xx", "text": {"body": "còn hàng không"}}]}}]}]}
    got = list(whatsapp.normalize_entries(payload))
    assert got == [("PN1", "849xx", "còn hàng không", "Chị Lan")]


def test_zalo_normalize_event():
    ev = {"event_name": "user_send_text", "recipient": {"id": "OA1"},
          "sender": {"id": "U1"}, "message": {"text": "giá bao nhiêu"}}
    assert zalo.normalize_event(ev) == ("OA1", "U1", "giá bao nhiêu")
    # a non-text event is ignored
    assert zalo.normalize_event({"event_name": "follow"}) == (None, None, None)


def test_meta_signature_verify():
    secret = "appsecret"
    body = b'{"hello":"world"}'
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert meta.verify_signature(secret, body, sig) is True
    assert meta.verify_signature(secret, body, "sha256=deadbeef") is False
    assert meta.verify_signature(secret, b"tampered", sig) is False


def test_meta_normalize_entries():
    payload = {"entry": [{"id": "PAGE1", "messaging": [
        {"sender": {"id": "PSID"}, "message": {"text": "hi"}}]}]}
    assert list(meta.normalize_entries(payload)) == [("PAGE1", "PSID", "hi")]
