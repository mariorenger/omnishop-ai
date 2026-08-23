"""Background worker: consumes the Valkey queue and computes embeddings async
(QueueProvider, ADR-006). Runs as a separate process (see docker-compose)."""
from __future__ import annotations
import json
import time

from .config import config
from .db import tenant_tx, wait_ready
from .modules import usage
from .providers.embeddings import to_pgvector
from .providers.registry import get_embedder
from .providers.queue import mark, pop


def embed_document(org_id: str, document_id: str) -> None:
    embedder = get_embedder()
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE document SET status='processing' WHERE id=%s", (document_id,))
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
    with tenant_tx(org_id) as conn:
        conn.execute("UPDATE document SET status='ready' WHERE id=%s", (document_id,))


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
        if kind == "embed_document":
            embed_document(org_id, payload["document_id"])
        elif kind == "embed_product":
            embed_product(org_id, payload["product_id"])
        else:
            raise ValueError(f"unknown job kind: {kind}")
        mark(job_id, "done")
    except Exception as e:  # noqa: BLE001
        mark(job_id, "error", str(e))
        if kind == "embed_document" and org_id:
            try:
                with tenant_tx(org_id) as conn:
                    conn.execute("UPDATE document SET status='error' WHERE id=%s", (payload.get("document_id"),))
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
