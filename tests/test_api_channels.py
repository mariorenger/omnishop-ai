"""Channel connect + verify. Guards the verify_channel path (regression: it read
r["config"] without selecting it -> KeyError)."""
from conftest import requires_db


@requires_db
def test_connect_and_verify_telegram(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    # telegram needs a plan that allows it — upgrade this org's subscription
    from app.db import admin_tx
    with admin_tx() as conn:
        conn.execute("UPDATE subscription SET plan_code='growth' WHERE organization_id=%s", (tenant["org_id"],))
    # connect a telegram channel with a (fake) token — stored + status returned
    r = client.post("/api/channels", json={"shop_id": shop, "kind": "telegram", "name": "TG",
                                            "credentials": {"bot_token": "123:FAKE"}}, headers=h)
    assert r.status_code == 200, r.text
    ch_id = r.json()["id"]
    assert r.json()["status"] in ("connected", "degraded")

    # verify must not crash (was KeyError 'config') and returns a status + note
    v = client.post(f"/api/channels/{ch_id}/verify", headers=h)
    assert v.status_code == 200, v.text
    assert v.json()["status"] in ("connected", "degraded")


@requires_db
def test_channel_list_reports_kind(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    client.post("/api/channels", json={"shop_id": shop, "kind": "website", "name": "Web"}, headers=h)
    rows = client.get(f"/api/channels?shop_id={shop}", headers=h).json()
    assert any(c["kind"] == "website" for c in rows)
