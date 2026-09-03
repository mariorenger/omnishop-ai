"""Conversation orchestrator (architecture §7). Deterministic control around the
LLM: classify intent, retrieve tenant-scoped context (products + knowledge),
call the LLM within quota, apply a handoff policy, persist, and meter usage.

The LLM never queries the DB — it only sees retrieved context (ADR-007/R-06).
"""
from __future__ import annotations
import json
import time
from typing import List

from ..db import tenant_tx
from ..providers.llm import ContextBlock
from ..providers.registry import get_embedder, get_llm
from ..providers.vectorstore import search_chunks, search_products, variants_for
from . import usage
from .billing import check_ai_quota

_HISTORY = 16  # per-customer conversation memory window (real runs remember context)
_SCORE_MIN = 0.05     # min semantic cosine to trust a vector-only hit
_KW_MIN = 0.35        # min lexical word-similarity to trust a keyword-only hit


def _relevant(row) -> bool:
    """A hybrid hit is kept if it is semantically close OR a strong keyword match."""
    return float(row.get("vscore", row.get("score", 0))) >= _SCORE_MIN or \
        (bool(row.get("kw")) and float(row.get("kwscore", 0)) >= _KW_MIN)

_PRODUCT_HINTS = ("giá", "size", "cỡ", "màu", "còn hàng", "tồn", "stock", "price",
                  "mua", "sản phẩm", "bao nhiêu", "variant", "mẫu")
_ORDER_HINTS = ("đơn", "mã đơn", "giao", "ship", "vận chuyển", "tracking", "order", "khi nào nhận")


def classify(text: str) -> str:
    t = text.lower()
    if any(h in t for h in _ORDER_HINTS):
        return "order"
    if any(h in t for h in _PRODUCT_HINTS):
        return "product"
    return "knowledge"


def _get_or_create_conversation(conn, org_id, shop_id, channel_id, customer_ref) -> str:
    row = conn.execute(
        "SELECT id FROM conversation WHERE channel_id=%s AND customer_ref=%s ORDER BY created_at DESC LIMIT 1",
        (channel_id, customer_ref),
    ).fetchone()
    if row:
        return str(row["id"])
    row = conn.execute(
        """INSERT INTO conversation (organization_id, shop_id, channel_id, customer_ref)
           VALUES (%s,%s,%s,%s) RETURNING id""",
        (org_id, shop_id, channel_id, customer_ref),
    ).fetchone()
    return str(row["id"])


def _fmt_price(v, currency="") -> str:
    if v is None:
        return ""
    n = f"{float(v):g}"
    return (n + " " + currency).strip() if currency else n


def _product_block(conn, p) -> ContextBlock:
    variants = variants_for(conn, p["id"])
    parts = []
    for v in variants:
        price_txt = _fmt_price(v["price"])
        extra = (", " + price_txt) if price_txt else ""
        parts.append(f"{v['name']} (còn {v['stock']}{extra})")
    vtxt = (" Phiên bản: " + ", ".join(parts) + ".") if parts else ""
    price = _fmt_price(p["price"], p["currency"]) or "liên hệ"
    desc = p["description"] or ""
    body = f"Giá {price}. {desc}{vtxt}".strip()
    return ContextBlock(source="product", title=p["name"], body=body)


def handle_incoming(org_id: str, shop_id: str, channel_id: str, customer_ref: str, text: str) -> dict:
    started = time.time()
    intent = classify(text)

    # 1) conversation + history + store the customer message
    with tenant_tx(org_id) as conn:
        shop = conn.execute("SELECT name FROM shop WHERE id=%s", (shop_id,)).fetchone()
        shop_name = shop["name"] if shop else "cửa hàng"
        botrow = conn.execute(
            "SELECT b.id AS bot_id, b.persona, b.config FROM channel c LEFT JOIN bot b ON b.id = c.bot_id WHERE c.id=%s",
            (channel_id,),
        ).fetchone()
        persona = (botrow["persona"] or "") if botrow else ""
        bot_id = str(botrow["bot_id"]) if botrow and botrow["bot_id"] else None
        bot_cfg = (botrow["config"] or {}) if botrow else {}
        conv_id = _get_or_create_conversation(conn, org_id, shop_id, channel_id, customer_ref)
        hist_rows = conn.execute(
            "SELECT role, content FROM message WHERE conversation_id=%s ORDER BY created_at DESC LIMIT %s",
            (conv_id, _HISTORY),
        ).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in reversed(hist_rows)]
        conn.execute(
            "INSERT INTO message (organization_id, conversation_id, role, content) VALUES (%s,%s,'customer',%s)",
            (org_id, conv_id, text),
        )

    # 2) quota gate (ADR-004) — enforced server-side
    quota = check_ai_quota(org_id)
    if not quota["allowed"]:
        reply = ("Cảm ơn bạn! Cửa hàng tạm thời đã đạt giới hạn trả lời tự động, "
                 "nhân viên sẽ phản hồi bạn sớm nhất có thể nhé.")
        with tenant_tx(org_id) as conn:
            conn.execute(
                "INSERT INTO message (organization_id, conversation_id, role, content, meta) VALUES (%s,%s,'system',%s,%s)",
                (org_id, conv_id, reply, json.dumps({"reason": "quota_exceeded"})),
            )
            conn.execute("UPDATE conversation SET status='needs_human', last_at=now() WHERE id=%s", (conv_id,))
        return {"conversation_id": conv_id, "reply": reply, "status": "needs_human", "quota_exceeded": True}

    # 3) retrieve tenant-scoped context (per-bot RAG top_k, default 3, clamped 1..8)
    top_k = max(1, min(8, int(bot_cfg.get("rag_top_k") or 3)))
    embedder = get_embedder()
    qvec = embedder.embed_one(text)
    context: List[ContextBlock] = []
    with tenant_tx(org_id) as conn:
        for p in search_products(conn, shop_id, qvec, query_text=text, k=top_k, bot_id=bot_id):
            if _relevant(p):
                context.append(_product_block(conn, p))
        for c in search_chunks(conn, shop_id, qvec, query_text=text, k=top_k, bot_id=bot_id):
            if _relevant(c):
                context.append(ContextBlock(source="knowledge", title=c["title"], body=c["content"]))

    # 4) LLM answer (per-bot model override if set)
    llm = get_llm(org_id, model=(bot_cfg.get("model") or None))
    res = llm.answer(question=text, context=context, history=history, shop_name=shop_name, persona=persona)
    latency_ms = int((time.time() - started) * 1000)

    # 5) handoff policy: nothing retrieved for an info-seeking intent -> human
    handoff_no_context = bool(bot_cfg.get("handoff_no_context", True))
    needs_human = handoff_no_context and len(context) == 0 and intent in ("product", "order")
    status = "needs_human" if needs_human else "ai"

    # 6) persist AI reply + update conversation
    with tenant_tx(org_id) as conn:
        conn.execute(
            "INSERT INTO message (organization_id, conversation_id, role, content, meta) VALUES (%s,%s,'ai',%s,%s)",
            (org_id, conv_id, res.text, json.dumps({
                "intent": intent, "retrieval_count": len(context), "model": res.model,
            })),
        )
        conn.execute("UPDATE conversation SET status=%s, last_at=now() WHERE id=%s", (status, conv_id))

    # 7) meter usage & cost (always)
    usage.record_ai_message(
        org_id, shop_id=shop_id, channel_id=channel_id, conversation_id=conv_id,
        model=res.model, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
        retrieval_count=len(context), latency_ms=latency_ms, customer_ref=customer_ref,
    )

    return {"conversation_id": conv_id, "reply": res.text, "status": status,
            "intent": intent, "retrieval_count": len(context), "model": res.model}
