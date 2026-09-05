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


def _force_plan(org_id, code):
    """Set a plan directly (bypassing the payment guard) — this test checks quota
    semantics, not the payment flow."""
    from app.db import admin_tx
    with admin_tx() as conn:
        conn.execute(
            """INSERT INTO subscription (organization_id, plan_code) VALUES (%s,%s)
               ON CONFLICT (organization_id) DO UPDATE SET plan_code=EXCLUDED.plan_code, status='active'""",
            (org_id, code))


@requires_db
def test_quota_is_mode_aware(client, tenant):
    h, org = tenant["headers"], tenant["org_id"]

    # managed subscription -> token allowance reported, allowed while under cap
    _force_plan(org, "growth")
    q = client.get("/api/subscription", headers=h).json()["quota"]
    assert q["llm_mode"] == "managed"
    assert q["tokens_included"] >= 1_000_000
    assert q["allowed"] is True

    # pay-as-you-go -> never blocked
    _force_plan(org, "payg")
    q2 = client.get("/api/subscription", headers=h).json()["quota"]
    assert q2["billing_mode"] == "payg" and q2["allowed"] is True

    # byok -> message fair-use cap surfaced
    _force_plan(org, "starter")
    q3 = client.get("/api/subscription", headers=h).json()["quota"]
    assert q3["llm_mode"] == "byok" and q3["messages_limit"] > 0


@requires_db
def test_tenant_cannot_self_activate_paid_plan(client, tenant):
    """A paid plan must go through checkout/payment — POST /subscription only
    accepts free plans."""
    h = tenant["headers"]
    r = client.post("/api/subscription", json={"plan_code": "growth"}, headers=h)
    assert r.status_code >= 400
    # free is fine
    assert client.post("/api/subscription", json={"plan_code": "free"}, headers=h).status_code == 200


@requires_db
def test_renewal_reminder_surfaces(client, tenant):
    """A paid plan nearing its period end reports an 'expiring' renewal so the UI
    can remind the tenant; a free plan never expires."""
    from app.db import admin_tx
    h, org = tenant["headers"], tenant["org_id"]
    _force_plan(org, "growth")
    with admin_tx() as conn:
        conn.execute("UPDATE subscription SET current_period_end = now() + interval '3 days' WHERE organization_id=%s", (org,))
    rn = client.get("/api/subscription", headers=h).json()["renewal"]
    assert rn["expires"] is True and rn["expiring"] is True and rn["days_left"] <= 7

    _force_plan(org, "free")
    assert client.get("/api/subscription", headers=h).json()["renewal"]["expires"] is False


@requires_db
def test_report_transfer_does_not_activate(client, tenant):
    """QR/bank transfer: reporting a transfer marks the invoice 'submitted' and
    does NOT activate the plan (an admin must confirm)."""
    h, org = tenant["headers"], tenant["org_id"]
    _force_plan(org, "free")
    co = client.post("/api/billing/checkout", json={"plan_code": "growth"}, headers=h).json()
    inv = co["invoice_id"]
    r = client.post(f"/api/billing/checkout/{inv}/submitted", headers=h).json()
    assert r["status"] == "submitted"
    # plan is still free — not activated by the tenant's self-report
    assert client.get("/api/subscription", headers=h).json()["entitlements"]["_plan"] == "free"


@requires_db
def test_tenant_cancels_own_request(client, tenant):
    """The tenant can cancel their own unpaid payment request."""
    h, org = tenant["headers"], tenant["org_id"]
    _force_plan(org, "free")
    co = client.post("/api/billing/checkout", json={"plan_code": "growth"}, headers=h).json()
    inv = co["invoice_id"]
    assert client.post(f"/api/billing/checkout/{inv}/cancel", headers=h).json()["status"] == "void"
    row = next(i for i in client.get("/api/billing/invoices", headers=h).json()["items"] if i["id"] == inv)
    assert row["status"] == "void"


@requires_db
def test_usage_by_customer_endpoint(client, tenant):
    r = client.get("/api/usage/by-customer", headers=tenant["headers"])
    assert r.status_code == 200 and isinstance(r.json(), list)
