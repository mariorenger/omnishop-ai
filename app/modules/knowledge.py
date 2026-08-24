"""Knowledge base: add text OR upload files (many types, with flexible OCR),
chunk, and enqueue embedding (async)."""
from __future__ import annotations
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from .. import audit
from ..db import tenant_tx
from ..errors import bad_request
from ..ingest.parse import SUPPORTED, extract_text
from ..providers.queue import enqueue
from ..providers.registry import get_ocr
from ..tenancy import OrgContext, get_org_context, require_role

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class DocBody(BaseModel):
    shop_id: str
    title: str
    text: str
    bot_id: Optional[str] = None


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


def store_document(org_id: str, shop_id: str, title: str, source: Optional[str], text: str, bot_id: Optional[str] = None) -> Tuple[str, int]:
    chunks = chunk_text(text)
    with tenant_tx(org_id) as conn:
        kb_id = _get_or_create_kb(conn, org_id, shop_id)
        if bot_id and not conn.execute("SELECT 1 FROM bot WHERE id=%s AND shop_id=%s", (bot_id, shop_id)).fetchone():
            bot_id = None
        doc = conn.execute(
            """INSERT INTO document (organization_id, knowledge_base_id, title, source, status, bot_id)
               VALUES (%s,%s,%s,%s,'pending',%s) RETURNING id""",
            (org_id, kb_id, title, source, bot_id),
        ).fetchone()
        doc_id = str(doc["id"])
        for i, ch in enumerate(chunks):
            conn.execute(
                """INSERT INTO chunk (organization_id, knowledge_base_id, document_id, ordinal, content, bot_id)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (org_id, kb_id, doc_id, i, ch, bot_id),
            )
    enqueue("embed_document", {"document_id": doc_id}, organization_id=org_id)
    return doc_id, len(chunks)


@router.get("/documents")
def list_documents(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        rows = conn.execute(
            """SELECT d.id, d.title, d.status, d.source, d.created_at,
                      (SELECT count(*) FROM chunk c WHERE c.document_id = d.id) AS chunks
               FROM document d JOIN knowledge_base kb ON kb.id = d.knowledge_base_id
               WHERE kb.shop_id = %s ORDER BY d.created_at DESC""",
            (shop_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "title": r["title"], "status": r["status"], "source": r["source"],
             "chunks": int(r["chunks"])} for r in rows]


@router.post("/documents")
def create_document(body: DocBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not body.text.strip():
        raise bad_request("text is empty")
    doc_id, n = store_document(ctx.org_id, body.shop_id, body.title, "text", body.text, body.bot_id)
    audit.record("knowledge.upload", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=doc_id, detail={"title": body.title, "chunks": n})
    return {"id": doc_id, "title": body.title, "chunks": n, "status": "pending"}


@router.get("/supported")
def supported_types():
    return {"extensions": sorted(SUPPORTED)}


@router.post("/upload")
async def upload_file(
    shop_id: str = Form(...),
    title: str = Form(""),
    bot_id: str = Form(""),
    file: UploadFile = File(...),
    ctx: OrgContext = Depends(require_role("admin")),
):
    data = await file.read()
    if not data:
        raise bad_request("empty file")
    if len(data) > 25 * 1024 * 1024:
        raise bad_request("file too large (max 25MB)")
    ocr = get_ocr(ctx.org_id)
    try:
        text = extract_text(file.filename or "upload", data, ocr=ocr)
    except ValueError as e:
        raise bad_request(str(e))
    if not text.strip():
        raise bad_request("no text could be extracted (for scanned files, configure OCR in Settings)")
    doc_id, n = store_document(ctx.org_id, shop_id, title or (file.filename or "Tài liệu"),
                               file.filename, text, bot_id or None)
    audit.record("knowledge.upload_file", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=doc_id, detail={"filename": file.filename, "chunks": n})
    return {"id": doc_id, "title": title or file.filename, "chunks": n, "status": "pending",
            "extracted_chars": len(text)}
