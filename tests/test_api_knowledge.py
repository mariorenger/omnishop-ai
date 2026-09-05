"""Knowledge: async upload -> worker ingest -> status ready + extracted text,
plus reprocess, delete and knowledge-base rename."""
from conftest import drain_jobs, requires_db


@requires_db
def test_upload_ingest_view_delete(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    body = "Chính sách đổi trả\n\nKhách được đổi trả trong 7 ngày nếu còn tem mác."
    r = client.post("/api/knowledge/upload",
                    data={"shop_id": shop, "title": "Đổi trả"},
                    files={"file": ("doipolicy.txt", body.encode(), "text/plain")},
                    headers=h)
    assert r.status_code == 200, r.text
    doc_id = r.json()["id"]
    assert r.json()["status"] == "queued"          # returns instantly, async

    drain_jobs()                                    # worker extracts + embeds

    docs = client.get(f"/api/knowledge/documents?shop_id={shop}", headers=h).json()["items"]
    row = next(d for d in docs if d["id"] == doc_id)
    assert row["status"] == "ready" and row["char_count"] > 0 and row["chunks"] >= 1

    detail = client.get(f"/api/knowledge/documents/{doc_id}", headers=h).json()
    assert "đổi trả" in detail["text"].lower()

    # reprocess re-queues
    assert client.post(f"/api/knowledge/documents/{doc_id}/reprocess", headers=h).json()["status"] == "queued"
    drain_jobs()

    # delete
    assert client.delete(f"/api/knowledge/documents/{doc_id}", headers=h).json()["ok"] is True
    docs2 = client.get(f"/api/knowledge/documents?shop_id={shop}", headers=h).json()["items"]
    assert all(d["id"] != doc_id for d in docs2)


@requires_db
def test_document_deactivate_excludes_from_retrieval(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    r = client.post("/api/knowledge/documents", json={"shop_id": shop, "title": "Giờ mở cửa",
                    "text": "Cửa hàng mở cửa từ 8 giờ sáng đến 10 giờ tối mỗi ngày."}, headers=h)
    doc_id = r.json()["id"]
    drain_jobs()

    q = {"shop_id": shop, "query": "cửa hàng mở cửa mấy giờ", "top_k": 5}
    titles = [c["title"] for c in client.post("/api/rag/query", json=q, headers=h).json()["chunks"]]
    assert any("Giờ mở cửa" in t for t in titles), "active doc should be retrievable"

    # deactivate -> excluded from retrieval but still listed
    assert client.put(f"/api/knowledge/documents/{doc_id}/active", json={"active": False}, headers=h).json()["active"] is False
    row = next(d for d in client.get(f"/api/knowledge/documents?shop_id={shop}", headers=h).json()["items"] if d["id"] == doc_id)
    assert row["active"] is False
    titles2 = [c["title"] for c in client.post("/api/rag/query", json=q, headers=h).json()["chunks"]]
    assert all("Giờ mở cửa" not in t for t in titles2), "inactive doc must not be retrieved"

    # reactivate -> retrievable again
    client.put(f"/api/knowledge/documents/{doc_id}/active", json={"active": True}, headers=h)
    titles3 = [c["title"] for c in client.post("/api/rag/query", json=q, headers=h).json()["chunks"]]
    assert any("Giờ mở cửa" in t for t in titles3)


@requires_db
def test_edit_extracted_text_reindexes(client, tenant):
    """Tenants can correct extracted/OCR text; saving re-chunks + re-embeds and
    the corrected content becomes retrievable."""
    h, shop = tenant["headers"], tenant["shop_id"]
    r = client.post("/api/knowledge/documents", json={"shop_id": shop, "title": "Bảo hành",
                    "text": "Sản phaam bảo hanh 12 thang."}, headers=h)   # deliberately garbled
    doc_id = r.json()["id"]
    drain_jobs()

    # edit title + fix the text
    e = client.put(f"/api/knowledge/documents/{doc_id}",
                   json={"title": "Chính sách bảo hành", "text": "Sản phẩm được bảo hành 12 tháng kể từ ngày mua."},
                   headers=h)
    assert e.status_code == 200 and e.json()["reindexed"] is True
    drain_jobs()

    detail = client.get(f"/api/knowledge/documents/{doc_id}", headers=h).json()
    assert detail["title"] == "Chính sách bảo hành"
    assert "bảo hành 12 tháng" in detail["text"]
    q = {"shop_id": shop, "query": "sản phẩm bảo hành bao lâu", "top_k": 5}
    titles = [c["title"] for c in client.post("/api/rag/query", json=q, headers=h).json()["chunks"]]
    assert any("bảo hành" in t.lower() for t in titles)


@requires_db
def test_knowledge_limits_and_per_file_cap(client, tenant):
    """Per-plan knowledge limits are exposed and the per-file size cap is enforced."""
    h, shop = tenant["headers"], tenant["shop_id"]
    lim = client.get(f"/api/knowledge/limits?shop_id={shop}", headers=h).json()
    assert "docs_limit" in lim and lim["max_file_mb"] >= 1
    big = b"x" * ((lim["max_file_mb"] + 1) * 1024 * 1024)   # one MB over the cap
    r = client.post("/api/knowledge/upload", data={"shop_id": shop, "title": "big"},
                    files={"file": ("big.txt", big, "text/plain")}, headers=h)
    assert r.status_code >= 400


@requires_db
def test_knowledge_docs_count_limit(client, tenant):
    """When the document count reaches the plan limit, adding more is blocked."""
    from app.db import admin_tx
    import json as _json
    h, shop, org = tenant["headers"], tenant["shop_id"], tenant["org_id"]
    # force this org's plan to allow just 1 document (temporary custom plan row)
    with admin_tx() as conn:
        conn.execute("INSERT INTO plan (code, name, price_month, entitlements) VALUES ('t_lim1','Lim',0,%s) ON CONFLICT (code) DO UPDATE SET entitlements=EXCLUDED.entitlements",
                     (_json.dumps({"knowledge_docs": 1, "max_file_mb": 25, "llm_mode": "byok", "billing_mode": "subscription"}),))
        conn.execute("INSERT INTO subscription (organization_id, plan_code) VALUES (%s,'t_lim1') ON CONFLICT (organization_id) DO UPDATE SET plan_code='t_lim1'", (org,))
    try:
        r1 = client.post("/api/knowledge/documents", json={"shop_id": shop, "title": "A", "text": "noi dung A"}, headers=h)
        assert r1.status_code == 200, r1.text
        r2 = client.post("/api/knowledge/documents", json={"shop_id": shop, "title": "B", "text": "noi dung B"}, headers=h)
        assert r2.status_code >= 400   # over the 1-doc limit
    finally:
        with admin_tx() as conn:
            conn.execute("UPDATE subscription SET plan_code='free' WHERE organization_id=%s", (org,))


@requires_db
def test_knowledge_base_rename(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    client.get(f"/api/knowledge/kb?shop_id={shop}", headers=h)   # ensure exists
    r = client.put("/api/knowledge/kb", json={"shop_id": shop, "name": "Kho FAQ"}, headers=h)
    assert r.status_code == 200 and r.json()["name"] == "Kho FAQ"
    assert client.get(f"/api/knowledge/kb?shop_id={shop}", headers=h).json()["name"] == "Kho FAQ"
