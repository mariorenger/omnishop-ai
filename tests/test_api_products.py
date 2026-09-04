"""Products: create, edit and delete (catalog the AI uses for price/stock)."""
from conftest import requires_db


@requires_db
def test_product_create_update_delete(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    r = client.post("/api/products", json={"shop_id": shop, "name": "Áo thun", "price": 100000,
                                           "variants": [{"name": "M", "stock": 5}]}, headers=h)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    # edit: rename + change price + variants
    u = client.put(f"/api/products/{pid}", json={"shop_id": shop, "name": "Áo thun cotton", "price": 120000,
                                                 "variants": [{"name": "L", "stock": 3}]}, headers=h)
    assert u.status_code == 200, u.text
    got = next(p for p in client.get(f"/api/products?shop_id={shop}", headers=h).json() if p["id"] == pid)
    assert got["name"] == "Áo thun cotton" and got["price"] == 120000
    assert [v["name"] for v in got["variants"]] == ["L"]

    # delete
    assert client.delete(f"/api/products/{pid}", headers=h).json()["ok"] is True
    assert all(p["id"] != pid for p in client.get(f"/api/products?shop_id={shop}", headers=h).json())
