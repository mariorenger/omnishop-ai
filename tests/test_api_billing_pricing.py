"""Flexible pricing + token quota semantics."""
from conftest import requires_db


@requires_db
def test_plans_expose_modes(client):
    plans = {p["code"]: p for p in client.get("/api/plans").json()}
    assert plans["starter"]["entitlements"]["llm_mode"] == "byok"
    assert plans["growth"]["entitlements"]["llm_mode"] == "managed"
    assert plans["growth"]["entitlements"]["ai_tokens_month"] >= 1_000_000
    assert plans["payg"]["entitlements"]["billing_mode"] == "payg"
    assert plans["payg"]["entitlements"]["payg_per_1k"] > 0


@requires_db
def test_quota_is_mode_aware(client, tenant):
    h = tenant["headers"]

    # managed subscription -> token allowance reported, allowed while under cap
    client.post("/api/subscription", json={"plan_code": "growth"}, headers=h)
    q = client.get("/api/subscription", headers=h).json()["quota"]
    assert q["llm_mode"] == "managed"
    assert q["tokens_included"] >= 1_000_000
    assert q["allowed"] is True

    # pay-as-you-go -> never blocked
    client.post("/api/subscription", json={"plan_code": "payg"}, headers=h)
    q2 = client.get("/api/subscription", headers=h).json()["quota"]
    assert q2["billing_mode"] == "payg" and q2["allowed"] is True

    # byok -> message fair-use cap surfaced
    client.post("/api/subscription", json={"plan_code": "starter"}, headers=h)
    q3 = client.get("/api/subscription", headers=h).json()["quota"]
    assert q3["llm_mode"] == "byok" and q3["messages_limit"] > 0


@requires_db
def test_usage_by_customer_endpoint(client, tenant):
    r = client.get("/api/usage/by-customer", headers=tenant["headers"])
    assert r.status_code == 200 and isinstance(r.json(), list)
