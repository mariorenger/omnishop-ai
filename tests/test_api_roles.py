"""Platform roles: admin (full) vs manager (read-only reports), plus admin
bootstrap and the Google Sign-In config gate."""
import uuid

import pytest

from conftest import requires_db


def _admin_headers(client):
    r = client.post("/api/auth/login", json={"email": "admin@omnishop.local", "password": "admin12345"})
    if r.status_code != 200:
        pytest.skip("seeded platform admin not present")
    return {"Authorization": f"Bearer {r.json()['token']}"}


@requires_db
def test_tenant_user_has_no_control_plane(client, tenant):
    h = tenant["headers"]
    assert client.get("/api/admin/overview", headers=h).status_code == 403
    assert client.get("/api/admin/tenants", headers=h).status_code == 403
    me = client.get("/api/auth/me", headers=h).json()["user"]
    assert me["platform_role"] in (None, "")


@requires_db
def test_admin_reads_and_exports(client):
    h = _admin_headers(client)
    assert client.get("/api/admin/overview", headers=h).status_code == 200
    csv = client.get("/api/admin/reports/tenants.csv", headers=h)
    assert csv.status_code == 200 and "tenant" in csv.text.splitlines()[0]
    assert client.get("/api/admin/reports/usage.csv", headers=h).status_code == 200


@requires_db
def test_manager_can_read_report_but_not_edit(client):
    admin_h = _admin_headers(client)
    # create a fresh user, then promote to manager
    email = f"mgr_{uuid.uuid4().hex[:8]}@test.local"
    d = client.post("/api/auth/signup", json={"email": email, "password": "pw12345678", "org_name": "Mgr Co"}).json()
    mh = {"Authorization": f"Bearer {d['token']}"}
    client.put("/api/auth/staff", json={"email": email, "platform_role": "manager"}, headers=admin_h)

    # manager: reads + export allowed
    assert client.get("/api/admin/overview", headers=mh).status_code == 200
    assert client.get("/api/admin/tenants", headers=mh).status_code == 200
    assert client.get("/api/admin/reports/tenants.csv", headers=mh).status_code == 200
    assert client.get("/api/auth/me", headers=mh).json()["user"]["platform_role"] == "manager"

    # manager: every write is forbidden
    assert client.put("/api/admin/plans/growth", json={"price_month": 1}, headers=mh).status_code == 403
    assert client.put("/api/admin/settings/cost", json={"cost_input_per_m": 1}, headers=mh).status_code == 403
    assert client.put("/api/admin/branding", json={"app_name": "X"}, headers=mh).status_code == 403
    assert client.put("/api/auth/staff", json={"email": email, "platform_role": "admin"}, headers=mh).status_code == 403


@requires_db
def test_bootstrap_admin_creates_login(client):
    from app.config import config
    from app.modules import auth
    email = f"boot_{uuid.uuid4().hex[:8]}@test.local"
    old = config.BOOTSTRAP_ADMIN_EMAIL, config.BOOTSTRAP_ADMIN_PASSWORD
    config.BOOTSTRAP_ADMIN_EMAIL, config.BOOTSTRAP_ADMIN_PASSWORD = email, "bootpw12345"
    try:
        auth.bootstrap_admin()
        login = client.post("/api/auth/login", json={"email": email, "password": "bootpw12345"})
        assert login.status_code == 200
        token = login.json()["token"]
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["user"]
        assert me["platform_role"] == "admin" and me["is_platform_admin"] is True
    finally:
        config.BOOTSTRAP_ADMIN_EMAIL, config.BOOTSTRAP_ADMIN_PASSWORD = old


@requires_db
def test_google_config_gate(client):
    r = client.get("/api/auth/google/config")
    assert r.status_code == 200 and "enabled" in r.json()
