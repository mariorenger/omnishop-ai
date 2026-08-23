"""Knowledge base: upload text, chunk, enqueue embedding (async). Parsing of
PDF/DOCX is deferred to OSS parsers (build-vs-buy) — MVP accepts pasted text."""
from __future__ import annotations
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..db import tenant_tx
from ..errors import bad_request
from ..providers.queue import enqueue
from ..tenancy import OrgContext, get_org_context, require_role

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class DocBody(BaseModel):
    shop_id: str
    title: str
    text: str


def chunk_text(text: str, target: int = 800) -> List[str]:
    parts = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 <= target:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p if len(p) <= target else p[:target]
            # very long paragraph: hard-split
            while len(buf) > target:
                chunks.append(buf[:target])
                buf = buf[target:]
    if buf:
        chunks.append(buf)
    return chunks or [text[:target]]


def _get_or_create_kb(conn, org_id: str, shop_id: str) -> str:
    if not conn.execute("SELECT 1 FROM shop WHERE id=%s", (shop_id,)).fetchone():
        raise bad_request("shop not found in this organization")
    row = conn.execute("SELECT id FROM knowledge_base WHERE shop_id=%s LIMIT 1", (shop_id,)).fetchone()
    if row:
        return str(row["id"])
    row = conn.execute(
        "INSERT INTO knowledge_base (organization_id, shop_id, name) VALUES (%s,%s,'Default') RETURNING id",
        (org_id, shop_id),
    ).fetchone()
    return str(row["id"])


@router.get("/documents")
def list_documents(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        rows = conn.execute(
            """SELECT d.id, d.title, d.status, d.created_at,
                      (SELECT count(*) FROM chunk c WHERE c.document_id = d.id) AS chunks
               FROM document d JOIN knowledge_base kb ON kb.id = d.knowledge_base_id
               WHERE kb.shop_id = %s ORDER BY d.created_at DESC""",
            (shop_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "title": r["title"], "status": r["status"],
             "chunks": int(r["chunks"])} for r in rows]


@router.post("/documents")
def create_document(body: DocBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not body.text.strip():
        raise bad_request("text is empty")
    chunks = chunk_text(body.text)
    with tenant_tx(ctx.org_id) as conn:
        kb_id = _get_or_create_kb(conn, ctx.org_id, body.shop_id)
        doc = conn.execute(
            """INSERT INTO document (organization_id, knowledge_base_id, title, status)
               VALUES (%s,%s,%s,'pending') RETURNING id""",
            (ctx.org_id, kb_id, body.title),
        ).fetchone()
        doc_id = str(doc["id"])
        for i, ch in enumerate(chunks):
            conn.execute(
                """INSERT INTO chunk (organization_id, knowledge_base_id, document_id, ordinal, content)
                   VALUES (%s,%s,%s,%s,%s)""",
                (ctx.org_id, kb_id, doc_id, i, ch),
            )
    enqueue("embed_document", {"document_id": doc_id}, organization_id=ctx.org_id)
    audit.record("knowledge.upload", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=doc_id, detail={"title": body.title, "chunks": len(chunks)})
    return {"id": doc_id, "title": body.title, "chunks": len(chunks), "status": "pending"}
