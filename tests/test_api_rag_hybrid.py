"""Hybrid retrieval end-to-end: an exact code matches on the keyword ranker even
when the embedding is weak; natural-language queries retrieve semantically."""
from conftest import drain_jobs, requires_db


@requires_db
def test_hybrid_keyword_and_semantic(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    # seed a product with a distinctive SKU and a knowledge doc
    client.post("/api/products", json={"shop_id": shop, "name": "Giày sneaker trắng",
                                       "description": "Giày thể thao êm chân", "sku": "SKU-GIAY-01",
                                       "price": 850000}, headers=h)
    client.post("/api/knowledge/documents", json={"shop_id": shop, "title": "Bảo hành",
                "text": "Chính sách bảo hành: đổi trả trong 7 ngày, bảo hành 12 tháng."}, headers=h)
    drain_jobs()   # embed product + document

    # 1) exact code -> keyword ranker nails it
    r = client.post("/api/rag/query", json={"shop_id": shop, "query": "SKU-GIAY-01", "top_k": 5}, headers=h).json()
    prods = r["products"]
    assert prods, "expected the SKU to retrieve its product"
    top = prods[0]
    assert top["kwscore"] >= 0.5

    # 2) natural-language question -> semantic retrieval of the knowledge doc
    r2 = client.post("/api/rag/query", json={"shop_id": shop, "query": "cho tôi hỏi về bảo hành sản phẩm",
                                             "top_k": 5}, headers=h).json()
    titles = [c["title"] for c in r2["chunks"]]
    assert any("Bảo hành" in t for t in titles)
