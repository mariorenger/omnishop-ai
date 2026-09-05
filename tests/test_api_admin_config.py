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
def test_admin_finance_pnl(client, tenant):
    # tenants cannot see platform finance
    assert client.get("/api/admin/finance", headers=tenant["headers"]).status_code in (401, 403)
    h = _admin(client)
    f = client.get("/api/admin/finance", headers=h).json()
    for k in ("revenue_month", "cost_month", "profit_month", "by_model", "by_tenant"):
        assert k in f
    assert isinstance(f["by_model"], list) and isinstance(f["by_tenant"], list)
    # profit = revenue - cost
    assert abs(f["profit_month"] - (f["revenue_month"] - f["cost_month"])) < 0.01
    # per-model CSV export works
    r = client.get("/api/admin/reports/finance.csv", headers=h)
    assert r.status_code == 200 and "model" in r.text.splitlines()[0]


@requires_db
def test_cannot_remove_last_platform_admin(client):
    h = _admin(client)
    # make the seeded admin the LAST admin (demote any others to manager first),
    # so the guard is exercised deterministically even on a reused DB
    for s in client.get("/api/auth/staff", headers=h).json():
        if s["platform_role"] == "admin" and s["email"].lower() != "admin@omnishop.local":
            client.put("/api/auth/staff", json={"email": s["email"], "platform_role": "manager"}, headers=h)
    # demoting the only admin (self) must be blocked so nobody gets locked out
    r = client.put("/api/auth/staff", json={"email": "admin@omnishop.local", "platform_role": "none"}, headers=h)
    assert r.status_code >= 400
    assert client.get("/api/admin/overview", headers=h).status_code == 200


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


@requires_db
def test_admin_runtime_settings_move_to_db(client):
    """Public domain + stale window are editable in the admin API and take effect
    immediately (env is only the fallback)."""
    from app.providers import registry
    h = _admin(client)
    d = client.put("/api/admin/settings/runtime", headers=h,
                   json={"public_base": "https://demo.omnishop.vn/", "stale_seconds": 123}).json()
    assert d["public_base"] == "https://demo.omnishop.vn" and d["stale_seconds"] == 123
    assert registry.public_base() == "https://demo.omnishop.vn" and registry.stale_seconds() == 123
    # tenants (unauthenticated here) cannot read it
    assert client.get("/api/admin/settings/runtime").status_code in (401, 403)
    registry.delete_config("platform:runtime")
    registry._runtime_cache["val"] = None


@requires_db
def test_admin_grants_plan_comped_not_revenue(client, tenant):
    """Admin granting a plan is admin_manual: it activates the plan but does NOT
    add to revenue (it's comped)."""
    h = _admin(client)
    org = tenant["org_id"]
    before = client.get("/api/admin/finance", headers=h).json()["revenue_month"]
    r = client.put(f"/api/admin/tenants/{org}/plan", headers=h, json={"plan_code": "growth"}).json()
    assert r.get("type") == "admin_manual"
    assert client.get("/api/subscription", headers=tenant["headers"]).json()["entitlements"]["_plan"] == "growth"
    after = client.get("/api/admin/finance", headers=h).json()
    assert after["revenue_month"] == before          # comped grant is not revenue
    assert after["comped_month"] >= 0
    # the tenant shows in the paginated management list
    tl = client.get(f"/api/admin/billing/tenants?q=Test", headers=h).json()
    assert "items" in tl and tl["total"] >= 1


@requires_db
def test_admin_confirms_reported_transfer(client, tenant):
    """A tenant reports a bank transfer (submitted); an admin confirms it, which
    activates the plan."""
    h_admin = _admin(client)
    h = tenant["headers"]
    co = client.post("/api/billing/checkout", json={"plan_code": "growth"}, headers=h).json()
    inv = co["invoice_id"]
    client.post(f"/api/billing/checkout/{inv}/submitted", headers=h)
    pend = client.get("/api/admin/billing/pending", headers=h_admin).json()["items"]
    assert any(p["id"] == inv for p in pend)
    assert client.post(f"/api/admin/invoices/{inv}/confirm", headers=h_admin).json()["ok"] is True
    assert client.get("/api/subscription", headers=h).json()["entitlements"]["_plan"] == "growth"


@requires_db
def test_admin_rejects_pending_request(client, tenant):
    """Admin can reject a pending/submitted payment request — it voids without
    activating the plan."""
    h_admin = _admin(client)
    h = tenant["headers"]
    co = client.post("/api/billing/checkout", json={"plan_code": "growth"}, headers=h).json()
    inv = co["invoice_id"]
    client.post(f"/api/billing/checkout/{inv}/submitted", headers=h)
    assert client.post(f"/api/admin/invoices/{inv}/reject", headers=h_admin).json()["status"] == "void"
    # plan not activated; request no longer pending
    assert client.get("/api/subscription", headers=h).json()["entitlements"]["_plan"] != "growth"
    pend = client.get("/api/admin/billing/pending", headers=h_admin).json()["items"]
    assert all(p["id"] != inv for p in pend)


@requires_db
def test_admin_email_settings_and_secret_kept(client):
    """Email provider is set from the UI; the secret is write-only (kept on
    re-save when omitted)."""
    from app.providers import registry
    h = _admin(client)
    client.put("/api/admin/settings/email", headers=h,
               json={"provider": "resend", "from_addr": "OmniShop <no-reply@demo.vn>", "secret": "re_testkey"})
    g = client.get("/api/admin/settings/email", headers=h).json()
    assert g["provider"] == "resend" and g["has_secret"] is True and g["from"].startswith("OmniShop")
    # re-saving without the secret keeps the stored one
    client.put("/api/admin/settings/email", headers=h,
               json={"provider": "resend", "from_addr": "OmniShop <no-reply@demo.vn>"})
    assert registry.resolve_email_config()["secret"] == "re_testkey"
    # console test send always succeeds (no external call)
    r = client.post("/api/admin/settings/email/test", headers=h,
                    json={"provider": "console", "to": "someone@example.com"}).json()
    assert r["ok"] is True
    registry.delete_config("notify:email")
