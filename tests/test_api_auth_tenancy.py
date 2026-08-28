"""Auth + tenant isolation (RLS): one tenant must never read another's rows."""
import uuid

from conftest import requires_db


@requires_db
def test_signup_login_me(client):
    email = f"u_{uuid.uuid4().hex[:8]}@test.local"
    r = client.post("/api/auth/signup", json={"email": email, "password": "pw12345678", "org_name": "Acme"})
    assert r.status_code == 200
    token = r.json()["token"]
    # wrong password rejected
    assert client.post("/api/auth/login", json={"email": email, "password": "nope"}).status_code >= 400
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["user"]["email"] == email


@requires_db
def test_cross_tenant_isolation(client, tenant):
    # tenant A creates a product
    a = tenant
    r = client.post("/api/products", json={"shop_id": a["shop_id"], "name": "Bí mật A", "price": 100},
                    headers=a["headers"])
    assert r.status_code == 200, r.text

    # tenant B is a different org
    email = f"b_{uuid.uuid4().hex[:8]}@test.local"
    d = client.post("/api/auth/signup", json={"email": email, "password": "pw12345678", "org_name": "B Co"}).json()
    hb = {"Authorization": f"Bearer {d['token']}", "X-Org-Id": d["orgs"][0]["id"]}

    # B cannot list products under A's shop (RLS: shop not visible)
    resp = client.get(f"/api/products?shop_id={a['shop_id']}", headers=hb)
    assert resp.status_code >= 400 or resp.json() == []


@requires_db
def test_create_additional_workspace(client, tenant):
    # a logged-in user can create another workspace (org) in-app and own it
    before = len(client.get("/api/auth/me", headers=tenant["headers"]).json()["orgs"])
    r = client.post("/api/orgs", json={"name": "Workspace 2"}, headers=tenant["headers"])
    assert r.status_code == 200 and r.json()["role"] == "owner"
    orgs = client.get("/api/auth/me", headers=tenant["headers"]).json()["orgs"]
    assert len(orgs) == before + 1
    # the new org works as a tenant context (create a shop in it)
    h2 = {"Authorization": tenant["headers"]["Authorization"], "X-Org-Id": r.json()["id"]}
    assert client.post("/api/shops", json={"name": "Shop 2"}, headers=h2).status_code == 200
