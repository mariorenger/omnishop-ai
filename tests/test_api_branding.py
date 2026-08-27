"""Config-driven branding: public read always works; admin write is gated."""
from conftest import requires_db


@requires_db
def test_public_branding_has_defaults(client):
    b = client.get("/api/branding").json()
    assert b["app_name"] and "accent_color" in b


@requires_db
def test_tenant_cannot_write_branding(client, tenant):
    # a normal tenant owner is not a platform admin
    r = client.put("/api/admin/branding", json={"app_name": "Hack"}, headers=tenant["headers"])
    assert r.status_code in (401, 403)


@requires_db
def test_admin_can_set_app_name(client):
    # uses the seeded platform admin if present; skips otherwise
    import pytest
    login = client.post("/api/auth/login", json={"email": "admin@omnishop.local", "password": "admin12345"})
    if login.status_code != 200:
        pytest.skip("seeded platform admin not present")
    h = {"Authorization": f"Bearer {login.json()['token']}"}
    orig = client.get("/api/admin/branding", headers=h).json()["app_name"]
    try:
        client.put("/api/admin/branding", json={"app_name": "QA Brand"}, headers=h)
        assert client.get("/api/branding").json()["app_name"] == "QA Brand"
    finally:
        client.put("/api/admin/branding", json={"app_name": orig}, headers=h)
