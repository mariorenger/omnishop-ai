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
def test_knowledge_base_rename(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    client.get(f"/api/knowledge/kb?shop_id={shop}", headers=h)   # ensure exists
    r = client.put("/api/knowledge/kb", json={"shop_id": shop, "name": "Kho FAQ"}, headers=h)
    assert r.status_code == 200 and r.json()["name"] == "Kho FAQ"
    assert client.get(f"/api/knowledge/kb?shop_id={shop}", headers=h).json()["name"] == "Kho FAQ"
