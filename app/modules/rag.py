"""RAG query/preview API — lets tenants test and tune retrieval with explicit
parameters (top_k, score threshold, source filter), and optionally get an answer.
Basic on purpose; the parameters are here so it can be improved without API churn."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..db import tenant_tx
from ..providers.llm import ContextBlock
from ..providers.registry import get_embedder, get_llm
from ..providers.vectorstore import search_chunks, search_products, variants_for
from ..tenancy import OrgContext, get_org_context

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RagQuery(BaseModel):
    shop_id: str
    query: str
    top_k: int = 4
    min_score: float = 0.05
    include_products: bool = True
    include_knowledge: bool = True
    answer: bool = False


@router.post("/query")
def rag_query(body: RagQuery, ctx: OrgContext = Depends(get_org_context)):
    top_k = max(1, min(body.top_k, 20))
    embedder = get_embedder()
    qvec = embedder.embed_one(body.query)
    products, chunks, context = [], [], []
    with tenant_tx(ctx.org_id) as conn:
        if body.include_products:
            for p in search_products(conn, body.shop_id, qvec, k=top_k):
                if float(p["score"]) >= body.min_score:
                    vs = [{"name": v["name"], "stock": int(v["stock"]),
                           "price": float(v["price"]) if v["price"] is not None else None}
                          for v in variants_for(conn, p["id"])]
                    products.append({"id": str(p["id"]), "name": p["name"], "score": round(float(p["score"]), 4),
                                     "price": float(p["price"]) if p["price"] is not None else None,
                                     "currency": p["currency"], "variants": vs})
                    context.append(ContextBlock(source="product", title=p["name"], body=p["description"] or ""))
        if body.include_knowledge:
            for c in search_chunks(conn, body.shop_id, qvec, k=top_k):
                if float(c["score"]) >= body.min_score:
                    chunks.append({"title": c["title"], "score": round(float(c["score"]), 4),
                                   "content": c["content"][:400]})
                    context.append(ContextBlock(source="knowledge", title=c["title"], body=c["content"]))
    out = {"products": products, "chunks": chunks, "context_blocks": len(context)}
    if body.answer:
        shop_name = "cửa hàng"
        with tenant_tx(ctx.org_id) as conn:
            r = conn.execute("SELECT name FROM shop WHERE id=%s", (body.shop_id,)).fetchone()
            if r:
                shop_name = r["name"]
        res = get_llm(ctx.org_id).answer(question=body.query, context=context, history=[], shop_name=shop_name)
        out["answer"] = res.text
        out["model"] = res.model
    return out
