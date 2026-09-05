"""SePay webhook: auto-confirm a bank transfer by matching the OMNI code in the
transfer content to a pending invoice, then activating the plan.

Config + log reads go through the provider store / admin pool directly so the
test doesn't depend on the login rate limiter (kept deterministic under load)."""
import random

from conftest import requires_db

_RID = random.randint(10_000_000, 99_000_000)


def _set_key(key: str):
    from app.providers import registry
    registry.write_config("payment:sepay", provider="sepay", api_key=key, extra={"account_no": "0123"})


@requires_db
def test_sepay_webhook_auto_activates(client, tenant):
    from app.db import admin_tx
    from app.providers.payment import invoice_code
    h = tenant["headers"]
    _set_key("sepaysecret")

    co = client.post("/api/billing/checkout", json={"plan_code": "growth"}, headers=h).json()
    inv = co["invoice_id"]
    code = invoice_code(inv)

    # wrong key is rejected
    bad = client.post("/webhook/sepay-webhook",
                      json={"id": 1, "transferType": "in", "transferAmount": 100, "content": code},
                      headers={"Authorization": "Apikey wrong"})
    assert bad.status_code == 401

    # correct key + matching content -> matched + plan activated
    sid = _RID; ref = f"FT{_RID}"
    r = client.post("/webhook/sepay-webhook",
                    json={"id": sid, "gateway": "Vietcombank", "accountNumber": "0123",
                          "transferType": "in", "transferAmount": 129,
                          "content": f"CT chuyen tien {code} cam on", "referenceCode": ref},
                    headers={"Authorization": "Apikey sepaysecret"})
    assert r.status_code == 200 and r.json()["matched"] is True
    assert client.get("/api/subscription", headers=h).json()["entitlements"]["_plan"] == "growth"

    # replay of the same SePay id is a no-op (idempotent)
    dup = client.post("/webhook/sepay-webhook",
                      json={"id": sid, "transferType": "in", "transferAmount": 129, "content": code},
                      headers={"Authorization": "Apikey sepaysecret"})
    assert dup.json().get("duplicate") is True

    # the transaction is logged (admin sees it via /api/admin/billing/sepay)
    with admin_tx() as conn:
        row = conn.execute("SELECT status, matched_invoice FROM sepay_transaction WHERE reference=%s", (ref,)).fetchone()
    assert row and row["status"] == "activated" and str(row["matched_invoice"]) == inv


@requires_db
def test_sepay_unmatched_transfer_is_logged_not_applied(client):
    _set_key("sepaysecret2")
    r = client.post("/webhook/sepay-webhook",
                    json={"id": _RID + 1, "transferType": "in", "transferAmount": 50000,
                          "content": "khong co ma gi", "referenceCode": f"FT{_RID + 1}"},
                    headers={"Authorization": "Apikey sepaysecret2"})
    assert r.status_code == 200 and r.json()["matched"] is False


@requires_db
def test_sepay_not_configured_rejects(client):
    from app.providers import registry
    registry.delete_config("payment:sepay")
    r = client.post("/webhook/sepay-webhook",
                    json={"id": 555003, "transferType": "in", "transferAmount": 1000, "content": "x"},
                    headers={"Authorization": "Apikey anything"})
    assert r.status_code == 400
