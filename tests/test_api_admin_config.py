"""Admin-owned config: plan pricing and cost rates are editable from the API,
and tenants cannot touch them."""
import pytest

from conftest import requires_db


def _admin(client):
    r = client.post("/api/auth/login", json={"email": "admin@omnishop.local", "password": "admin12345"})
    if r.status_code != 200:
        pytest.skip("seeded platform admin not present")
    return {"Authorization": f"Bearer {r.json()['token']}"}


@requires_db
def test_tenant_cannot_edit_plans_or_cost(client, tenant):
    h = tenant["headers"]
    assert client.get("/api/admin/plans", headers=h).status_code in (401, 403)
    assert client.put("/api/admin/plans/growth", json={"price_month": 1}, headers=h).status_code in (401, 403)
    assert client.get("/api/admin/settings/cost", headers=h).status_code in (401, 403)


@requires_db
def test_admin_edits_plan_price_and_tokens(client):
    h = _admin(client)
    plans = {p["code"]: p for p in client.get("/api/admin/plans", headers=h).json()}
    orig = plans["growth"]
    try:
        client.put("/api/admin/plans/growth", headers=h,
                   json={"price_month": 129, "entitlements": {"ai_tokens_month": 6000000}})
        after = {p["code"]: p for p in client.get("/api/admin/plans", headers=h).json()}["growth"]
        assert after["price_month"] == 129
        assert after["entitlements"]["ai_tokens_month"] == 6000000
        # public /api/plans reflects the change too
        pub = {p["code"]: p for p in client.get("/api/plans").json()}["growth"]
        assert pub["price_month"] == 129
    finally:
        client.put("/api/admin/plans/growth", headers=h,
                   json={"price_month": orig["price_month"], "entitlements": orig["entitlements"]})


@requires_db
def test_admin_edits_cost_rates(client):
    h = _admin(client)
    orig = client.get("/api/admin/settings/cost", headers=h).json()
    try:
        d = client.put("/api/admin/settings/cost", headers=h,
                       json={"cost_input_per_m": 7.5, "cost_output_per_m": 30}).json()
        assert d["input"] == 7.5 and d["output"] == 30
    finally:
        client.put("/api/admin/settings/cost", headers=h,
                   json={"cost_input_per_m": orig["input"], "cost_output_per_m": orig["output"],
                         "cost_embedding_per_m": orig["embedding"]})
