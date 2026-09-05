"""Knowledge base: add text OR upload files (many types, flexible OCR).

Uploads are ingested ASYNC: the file is stored and a job is queued, so the API
stays responsive even when many tenants upload large/scanned files at once. The
worker extracts text → chunks → embeds, moving the document through
queued → processing → ready | error. Tenants can view the extracted text, the
status/error, char & chunk counts, delete, or re-process a document.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from .. import audit
from ..db import no_tenant, tenant_tx
from ..errors import bad_request, not_found
from ..ingest.parse import SUPPORTED, extract_text
from ..providers.queue import enqueue
from ..tenancy import OrgContext, get_org_context, require_role

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

MAX_UPLOAD = 25 * 1024 * 1024   # hard ceiling; the per-plan limit may be lower


def knowledge_limits(org_id: str, shop_id: str) -> dict:
    """Per-plan knowledge limits + current usage (0 limit = unlimited)."""
    from .billing import resolve_entitlements
    ent = resolve_entitlements(org_id)
    docs_limit = int(ent.get("knowledge_docs", 0) or 0)
    max_file_mb = int(ent.get("max_file_mb", 25) or 25)
    with tenant_tx(org_id) as conn:
        used = conn.execute(
            """SELECT count(*) AS n FROM document d JOIN knowledge_base kb ON kb.id=d.knowledge_base_id
               WHERE kb.shop_id=%s""", (shop_id,)).fetchone()["n"]
    return {"docs_used": int(used), "docs_limit": docs_limit, "max_file_mb": max_file_mb}


def _assert_can_add(org_id: str, shop_id: str) -> None:
    lim = knowledge_limits(org_id, shop_id)
    if lim["docs_limit"] and lim["docs_used"] >= lim["docs_limit"]:
        raise bad_request(f"Đã đạt giới hạn {lim['docs_limit']} tài liệu của gói. Vui lòng nâng cấp để thêm.")


@router.get("/limits")
def get_limits(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    return knowledge_limits(ctx.org_id, shop_id)


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
        "INSERT INTO knowledge_base (organization_id, shop_id, name) VALUES (%s,%s,'Kho kiến thức') RETURNING id",
        (org_id, shop_id),
    ).fetchone()
    return str(row["id"])


def _write_chunks(conn, org_id, kb_id, doc_id, chunks, bot_id):
    conn.execute("DELETE FROM chunk WHERE document_id=%s", (doc_id,))
    for i, ch in enumerate(chunks):
        conn.execute(
            """INSERT INTO chunk (organization_id, knowledge_base_id, document_id, ordinal, content, bot_id)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (org_id, kb_id, doc_id, i, ch, bot_id),
        )


def store_document(org_id: str, shop_id: str, title: str, source: Optional[str], text: str,
                   bot_id: Optional[str] = None) -> Tuple[str, int]:
    """Sync path for pasted text: chunk now, embed async."""
    chunks = chunk_text(text)
    with tenant_tx(org_id) as conn:
        kb_id = _get_or_create_kb(conn, org_id, shop_id)
        if bot_id and not conn.execute("SELECT 1 FROM bot WHERE id=%s AND shop_id=%s", (bot_id, shop_id)).fetchone():
            bot_id = None
        doc = conn.execute(
            """INSERT INTO document (organization_id, knowledge_base_id, title, source, status, bot_id, char_count, mime)
               VALUES (%s,%s,%s,%s,'pending',%s,%s,'text/plain') RETURNING id""",
            (org_id, kb_id, title, source, bot_id, len(text)),
        ).fetchone()
        doc_id = str(doc["id"])
        _write_chunks(conn, org_id, kb_id, doc_id, chunks, bot_id)
    enqueue("embed_document", {"document_id": doc_id}, organization_id=org_id)
    return doc_id, len(chunks)


@router.get("/documents")
def list_documents(shop_id: str, limit: int = 50, offset: int = 0,
                   ctx: OrgContext = Depends(get_org_context)):
    limit = max(1, min(limit, 200)); offset = max(0, offset)
    with tenant_tx(ctx.org_id) as conn:
        total = conn.execute(
            """SELECT count(*) AS n FROM document d JOIN knowledge_base kb ON kb.id = d.knowledge_base_id
               WHERE kb.shop_id = %s""", (shop_id,)).fetchone()["n"]
        rows = conn.execute(
            """SELECT d.id, d.title, d.status, d.source, d.error, d.char_count, d.mime, d.active, d.created_at,
                      (SELECT count(*) FROM chunk c WHERE c.document_id = d.id) AS chunks
               FROM document d JOIN knowledge_base kb ON kb.id = d.knowledge_base_id
               WHERE kb.shop_id = %s ORDER BY d.created_at DESC LIMIT %s OFFSET %s""",
            (shop_id, limit, offset),
        ).fetchall()
    items = [{"id": str(r["id"]), "title": r["title"], "status": r["status"], "source": r["source"],
              "error": r["error"], "char_count": r["char_count"], "mime": r["mime"], "active": bool(r["active"]),
              "chunks": int(r["chunks"]), "created_at": r["created_at"].isoformat()} for r in rows]
    return {"items": items, "total": int(total), "limit": limit, "offset": offset,
            "has_more": offset + len(items) < int(total)}


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute(
            "SELECT id, title, status, source, error, char_count, mime, active, created_at FROM document WHERE id=%s",
            (doc_id,),
        ).fetchone()
        if not r:
            raise not_found("document not found")
        parts = conn.execute(
            "SELECT content FROM chunk WHERE document_id=%s ORDER BY ordinal", (doc_id,)
        ).fetchall()
    text = "\n\n".join(p["content"] for p in parts)
    return {"id": str(r["id"]), "title": r["title"], "status": r["status"], "source": r["source"],
            "error": r["error"], "char_count": r["char_count"], "mime": r["mime"], "active": bool(r["active"]),
            "chunks": len(parts), "created_at": r["created_at"].isoformat(),
            "text": text[:200_000]}


class EditDocBody(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None    # when set, the document is re-chunked and re-embedded


@router.put("/documents/{doc_id}")
def edit_document(doc_id: str, body: EditDocBody, ctx: OrgContext = Depends(require_role("admin"))):
    """Correct the extracted text or rename a document. Editing the text re-chunks
    and re-embeds it (OCR/parse errors are common, so tenants can fix them). Note:
    running "Xử lý lại" on a file-backed document re-extracts from the original
    file and overwrites manual edits."""
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute("SELECT knowledge_base_id, bot_id FROM document WHERE id=%s", (doc_id,)).fetchone()
        if not r:
            raise not_found("document not found")
        if body.title is not None and body.title.strip():
            conn.execute("UPDATE document SET title=%s WHERE id=%s", (body.title.strip(), doc_id))
        rechunk = body.text is not None
        if rechunk:
            if not body.text.strip():
                raise bad_request("text is empty")
            chunks = chunk_text(body.text)
            _write_chunks(conn, ctx.org_id, str(r["knowledge_base_id"]), doc_id, chunks, r["bot_id"])
            conn.execute("UPDATE document SET char_count=%s, status='pending', error=NULL WHERE id=%s",
                         (len(body.text), doc_id))
    if rechunk:
        enqueue("embed_document", {"document_id": doc_id}, organization_id=ctx.org_id)
    audit.record("knowledge.edit", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=doc_id, detail={"reindexed": rechunk})
    return {"ok": True, "reindexed": rechunk, "status": "pending" if rechunk else None}


class ActiveBody(BaseModel):
    active: bool


@router.put("/documents/{doc_id}/active")
def set_document_active(doc_id: str, body: ActiveBody, ctx: OrgContext = Depends(require_role("admin"))):
    """Enable/disable a document for retrieval without deleting it."""
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM document WHERE id=%s", (doc_id,)).fetchone():
            raise not_found("document not found")
        conn.execute("UPDATE document SET active=%s WHERE id=%s", (body.active, doc_id))
    audit.record("knowledge.active", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=doc_id, detail={"active": body.active})
    return {"ok": True, "active": body.active}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, ctx: OrgContext = Depends(require_role("admin"))):
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute("SELECT file_asset_id FROM document WHERE id=%s", (doc_id,)).fetchone()
        if not r:
            raise not_found("document not found")
        conn.execute("DELETE FROM chunk WHERE document_id=%s", (doc_id,))
        conn.execute("DELETE FROM document WHERE id=%s", (doc_id,))
    if r["file_asset_id"]:
        with no_tenant() as conn:
            conn.execute("DELETE FROM file_asset WHERE id=%s", (r["file_asset_id"],))
    audit.record("knowledge.delete", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=doc_id)
    return {"ok": True}


@router.post("/documents/{doc_id}/reprocess")
def reprocess_document(doc_id: str, ctx: OrgContext = Depends(require_role("admin"))):
    """Re-run ingestion (re-extract if a file is stored, else just re-embed)."""
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute("SELECT file_asset_id, source FROM document WHERE id=%s", (doc_id,)).fetchone()
        if not r:
            raise not_found("document not found")
        conn.execute("UPDATE document SET status='queued', error=NULL WHERE id=%s", (doc_id,))
    if r["file_asset_id"]:
        enqueue("ingest_document", {"document_id": doc_id, "file_asset_id": str(r["file_asset_id"]),
                                    "filename": r["source"] or "upload"}, organization_id=ctx.org_id)
    else:
        enqueue("embed_document", {"document_id": doc_id}, organization_id=ctx.org_id)
    return {"ok": True, "status": "queued"}


@router.post("/documents")
def create_document(body: DocBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not body.text.strip():
        raise bad_request("text is empty")
    _assert_can_add(ctx.org_id, body.shop_id)
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
    """Store the raw file and queue extraction+embedding (async). Returns instantly
    with status 'queued' so concurrent uploads never block the API."""
    data = await file.read()
    if not data:
        raise bad_request("empty file")
    _assert_can_add(ctx.org_id, shop_id)
    max_mb = knowledge_limits(ctx.org_id, shop_id)["max_file_mb"]
    cap = min(MAX_UPLOAD, max_mb * 1024 * 1024)
    if len(data) > cap:
        raise bad_request(f"Tệp vượt giới hạn {max_mb}MB của gói. Vui lòng nén nhỏ hơn hoặc nâng cấp gói.")
    filename = file.filename or "upload"
    mime = file.content_type or "application/octet-stream"
    with tenant_tx(ctx.org_id) as conn:
        kb_id = _get_or_create_kb(conn, ctx.org_id, shop_id)
        bid = bot_id or None
        if bid and not conn.execute("SELECT 1 FROM bot WHERE id=%s AND shop_id=%s", (bid, shop_id)).fetchone():
            bid = None
    with no_tenant() as conn:
        asset = conn.execute(
            "INSERT INTO file_asset (organization_id, mime, bytes) VALUES (%s,%s,%s) RETURNING id",
            (ctx.org_id, mime, data),
        ).fetchone()
    with tenant_tx(ctx.org_id) as conn:
        doc = conn.execute(
            """INSERT INTO document (organization_id, knowledge_base_id, title, source, status, bot_id,
                                     mime, bytes, file_asset_id)
               VALUES (%s,%s,%s,%s,'queued',%s,%s,%s,%s) RETURNING id""",
            (ctx.org_id, kb_id, title or filename, filename, bid, mime, len(data), asset["id"]),
        ).fetchone()
        doc_id = str(doc["id"])
    enqueue("ingest_document", {"document_id": doc_id, "file_asset_id": str(asset["id"]), "filename": filename},
            organization_id=ctx.org_id)
    audit.record("knowledge.upload_file", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=doc_id, detail={"filename": filename, "bytes": len(data)})
    return {"id": doc_id, "title": title or filename, "status": "queued"}


# --------------------------- knowledge base (rename) ------------------------

class KBBody(BaseModel):
    shop_id: str
    name: str


@router.get("/kb")
def get_kb(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        kb_id = _get_or_create_kb(conn, ctx.org_id, shop_id)
        r = conn.execute("SELECT name FROM knowledge_base WHERE id=%s", (kb_id,)).fetchone()
    return {"id": kb_id, "name": r["name"]}


@router.put("/kb")
def rename_kb(body: KBBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not body.name.strip():
        raise bad_request("tên không được để trống")
    with tenant_tx(ctx.org_id) as conn:
        kb_id = _get_or_create_kb(conn, ctx.org_id, body.shop_id)
        conn.execute("UPDATE knowledge_base SET name=%s WHERE id=%s", (body.name.strip(), kb_id))
    return {"ok": True, "name": body.name.strip()}
