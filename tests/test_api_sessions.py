"""Session revocation (logout-all) and the admin audit log."""
import uuid

import pytest

from conftest import requires_db


@requires_db
def test_logout_all_revokes_existing_token(client):
    email = f"s_{uuid.uuid4().hex[:8]}@test.local"
    d = client.post("/api/auth/signup", json={"email": email, "password": "pw12345678"}).json()
    h = {"Authorization": f"Bearer {d['token']}"}
    assert client.get("/api/auth/me", headers=h).status_code == 200
    # revoke -> the same token must stop working
    assert client.post("/api/auth/logout-all", headers=h).json()["ok"] is True
    assert client.get("/api/auth/me", headers=h).status_code == 401
    # a fresh login issues a token that works again
    d2 = client.post("/api/auth/login", json={"email": email, "password": "pw12345678"}).json()
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {d2['token']}"}).status_code == 200


@requires_db
def test_audit_log_records_login(client):
    admin = client.post("/api/auth/login", json={"email": "admin@omnishop.local", "password": "admin12345"})
    if admin.status_code != 200:
        pytest.skip("seeded platform admin not present")
    h = {"Authorization": f"Bearer {admin.json()['token']}"}
    rows = client.get("/api/admin/audit", headers=h).json()
    assert isinstance(rows, list) and any(r["action"] == "auth.login" for r in rows)
