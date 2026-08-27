"""Background worker: consumes the Valkey queue and computes embeddings async
(QueueProvider, ADR-006). Runs as a separate process (see docker-compose)."""
from __future__ import annotations
import json
import time

from .config import config
from .db import no_tenant, tenant_tx, wait_ready
from .ingest.parse import extract_text
from .modules import usage
from .modules.knowledge import chunk_text
from .providers.embeddings import to_pgvector
from .providers.registry import get_embedder, get_ocr
from .providers.queue import mark, pop


def _embed_doc_chunks(org_id: str, document_id: str) -> None:
    """Embed any not-yet-embedded chunks of a document."""
    embedder = get_embedder()
    with tenant_tx(org_id) as conn:
        chunks = conn.execute(
            "SELECT id, content FROM chunk WHERE document_id=%s AND embedding IS NULL", (document_id,)
        ).fetchall()
    if chunks:
        vecs = embedder.embed([c["content"] for c in chunks])
        tokens = sum(config.est_tokens(c["content"]) for c in chunks)
        with tenant_tx(org_id) as conn:
            for c, v in zip(chunks, vecs):
                conn.execute("UPDATE chunk SET embedding=%s::vector WHERE id=%s", (to_pgvector(v), c["id"]))
        usage.record_embedding(org_id, tokens)


def embed_document(org_id: str, document_id: str) -> None:
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE document SET status='processing' WHERE id=%s", (document_id,))
    _embed_doc_chunks(org_id, document_id)
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE document SET status='ready', error=NULL WHERE id=%s", (document_id,))


def ingest_document(org_id: str, document_id: str, file_asset_id: str, filename: str) -> None:
    """Full async pipeline for an uploaded file: extract -> chunk -> embed.
    Heavy work (PDF/OCR) lives here, off the request path, so concurrent uploads
    from many tenants never block the API."""
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE document SET status='processing', error=NULL WHERE id=%s", (document_id,))
        row = conn.execute(
            "SELECT knowledge_base_id, bot_id FROM document WHERE id=%s", (document_id,)
        ).fetchone()
    if not row:
        return
    with no_tenant() as conn:
        asset = conn.execute("SELECT bytes FROM file_asset WHERE id=%s", (file_asset_id,)).fetchone()
    if not asset:
        _fail(org_id, document_id, "Không tìm thấy tệp đã tải lên.")
        return
    try:
        text = extract_text(filename, bytes(asset["bytes"]), ocr=get_ocr(org_id))
    except Exception as e:  # noqa: BLE001
        _fail(org_id, document_id, f"Lỗi trích xuất: {e}")
        return
    if not text.strip():
        _fail(org_id, document_id,
              "Không trích xuất được văn bản (tệp scan cần bật OCR trong Cài đặt).")
        return
    chunks = chunk_text(text)
    with tenant_tx(org_id) as conn:
        conn.execute("DELETE FROM chunk WHERE document_id=%s", (document_id,))
        for i, ch in enumerate(chunks):
            conn.execute(
                """INSERT INTO chunk (organization_id, knowledge_base_id, document_id, ordinal, content, bot_id)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (org_id, row["knowledge_base_id"], document_id, i, ch, row["bot_id"]),
            )
        conn.execute("UPDATE document SET char_count=%s WHERE id=%s", (len(text), document_id))
    _embed_doc_chunks(org_id, document_id)
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE document SET status='ready', error=NULL WHERE id=%s", (document_id,))


def _fail(org_id: str, document_id: str, msg: str) -> None:
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE document SET status='error', error=%s WHERE id=%s", (msg[:500], document_id))


def embed_product(org_id: str, product_id: str) -> None:
    embedder = get_embedder()
    with tenant_tx(org_id) as conn:
        p = conn.execute(
            "SELECT name, description, sku, attributes FROM product WHERE id=%s", (product_id,)
        ).fetchone()
    if not p:
        return
    text = " ".join(filter(None, [
        p["name"], p["description"] or "", p["sku"] or "",
        " ".join(f"{k}:{v}" for k, v in (p["attributes"] or {}).items()),
    ]))
    vec = embedder.embed_one(text)
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE product SET embedding=%s::vector WHERE id=%s", (to_pgvector(vec), product_id))
    usage.record_embedding(org_id, config.est_tokens(text))


def process(job: dict) -> None:
    kind, payload, org_id, job_id = job["kind"], job["payload"], job.get("organization_id"), job["job_id"]
    mark(job_id, "running")
    try:
        if kind == "ingest_document":
            ingest_document(org_id, payload["document_id"], payload["file_asset_id"], payload.get("filename", "upload"))
        elif kind == "embed_document":
            embed_document(org_id, payload["document_id"])
        elif kind == "embed_product":
            embed_product(org_id, payload["product_id"])
        else:
            raise ValueError(f"unknown job kind: {kind}")
        mark(job_id, "done")
    except Exception as e:  # noqa: BLE001
        mark(job_id, "error", str(e))
        if kind in ("embed_document", "ingest_document") and org_id:
            try:
                _fail(org_id, payload.get("document_id"), str(e))
            except Exception:  # noqa: BLE001
                pass
        print(f"[worker] job {job_id} ({kind}) failed: {e}", flush=True)


def main() -> None:
    print("[worker] starting; waiting for database…", flush=True)
    wait_ready(60)
    print("[worker] ready; consuming jobs", flush=True)
    while True:
        try:
            job = pop(timeout=5)
            if job:
                print(f"[worker] processing {job['kind']} ({job['job_id']})", flush=True)
                process(job)
        except Exception as e:  # noqa: BLE001 — keep the loop alive
            print(f"[worker] loop error: {e}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
