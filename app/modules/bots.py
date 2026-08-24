"""Bots: a shop can have many bots, each with its own persona (custom prompt),
greeting and appearance. Channels bind to a bot. Includes a live test-chat."""
from __future__ import annotations
import json
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..db import tenant_tx
from ..errors import bad_request, not_found
from ..providers.llm import ContextBlock
from ..providers.registry import get_embedder, get_llm
from ..providers.vectorstore import search_chunks, search_products, variants_for
from ..tenancy import OrgContext, get_org_context, require_role

router = APIRouter(prefix="/api/bots", tags=["bots"])


class BotBody(BaseModel):
    shop_id: Optional[str] = None
    name: str = ""
    persona: str = ""
    greeting: str = "Xin chào! Mình có thể giúp gì cho bạn?"
    avatar_url: str = ""
    accent_color: str = "#6d7cff"
    config: dict = {}   # {handoff_no_context, business_hours:{enabled,start,end,off_message}}


class TestBody(BaseModel):
    text: str
    history: list = []   # [{role, content}] for multi-turn memory


def get_or_create_default_bot(conn, org_id: str, shop_id: str) -> str:
    row = conn.execute("SELECT id FROM bot WHERE shop_id=%s ORDER BY created_at LIMIT 1", (shop_id,)).fetchone()
    if row:
        return str(row["id"])
    row = conn.execute(
        "INSERT INTO bot (organization_id, shop_id, name) VALUES (%s,%s,'Trợ lý mặc định') RETURNING id",
        (org_id, shop_id),
    ).fetchone()
    return str(row["id"])


def _bot_dict(r) -> dict:
    return {"id": str(r["id"]), "shop_id": str(r["shop_id"]), "name": r["name"], "persona": r["persona"] or "",
            "greeting": r["greeting"] or "", "avatar_url": r["avatar_url"] or "", "accent_color": r["accent_color"],
            "config": r["config"] or {}}


@router.get("")
def list_bots(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        rows = conn.execute(
            """SELECT b.*, (SELECT count(*) FROM channel c WHERE c.bot_id=b.id) AS channels
               FROM bot b WHERE b.shop_id=%s ORDER BY b.created_at""", (shop_id,)
        ).fetchall()
    return [dict(_bot_dict(r), channels=int(r["channels"])) for r in rows]


@router.post("")
def create_bot(body: BotBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not body.shop_id:
        raise bad_request("shop_id là bắt buộc")
    if not body.name.strip():
        raise bad_request("tên trợ lý là bắt buộc")
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM shop WHERE id=%s", (body.shop_id,)).fetchone():
            raise bad_request("shop not found")
        row = conn.execute(
            """INSERT INTO bot (organization_id, shop_id, name, persona, greeting, avatar_url, accent_color, config)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (ctx.org_id, body.shop_id, body.name, body.persona, body.greeting, body.avatar_url, body.accent_color,
             json.dumps(body.config or {})),
        ).fetchone()
    audit.record("bot.create", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=str(row["id"]))
    return {"id": str(row["id"])}


@router.get("/{bot_id}")
def get_bot(bot_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute("SELECT * FROM bot WHERE id=%s", (bot_id,)).fetchone()
    if not r:
        raise not_found("bot not found")
    return _bot_dict(r)


@router.put("/{bot_id}")
def update_bot(bot_id: str, body: BotBody, ctx: OrgContext = Depends(require_role("admin"))):
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM bot WHERE id=%s", (bot_id,)).fetchone():
            raise not_found("bot not found")
        conn.execute(
            """UPDATE bot SET name=%s, persona=%s, greeting=%s, avatar_url=%s, accent_color=%s, config=%s WHERE id=%s""",
            (body.name, body.persona, body.greeting, body.avatar_url, body.accent_color, json.dumps(body.config or {}), bot_id),
        )
    audit.record("bot.update", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=bot_id)
    return {"ok": True}


@router.delete("/{bot_id}")
def delete_bot(bot_id: str, ctx: OrgContext = Depends(require_role("admin"))):
    with tenant_tx(ctx.org_id) as conn:
        used = conn.execute("SELECT count(*) AS n FROM channel WHERE bot_id=%s", (bot_id,)).fetchone()["n"]
        if int(used) > 0:
            raise bad_request("Trợ lý đang gắn với kênh. Hãy gỡ khỏi kênh trước khi xoá.")
        conn.execute("DELETE FROM bot WHERE id=%s", (bot_id,))
    audit.record("bot.delete", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=bot_id)
    return {"ok": True}


@router.post("/{bot_id}/test")
def test_bot(bot_id: str, body: TestBody, ctx: OrgContext = Depends(get_org_context)):
    if not body.text.strip():
        raise bad_request("nhập câu hỏi để thử")
    with tenant_tx(ctx.org_id) as conn:
        b = conn.execute("SELECT shop_id, persona FROM bot WHERE id=%s", (bot_id,)).fetchone()
        if not b:
            raise not_found("bot not found")
        shop_id = str(b["shop_id"]); persona = b["persona"] or ""
        shop = conn.execute("SELECT name FROM shop WHERE id=%s", (shop_id,)).fetchone()
        shop_name = shop["name"] if shop else "cửa hàng"
    qvec = get_embedder().embed_one(body.text)
    context = []
    with tenant_tx(ctx.org_id) as conn:
        for p in search_products(conn, shop_id, qvec, k=3, bot_id=bot_id):
            if float(p["score"]) >= 0.05:
                vs = variants_for(conn, p["id"])
                vt = ("; ".join(f"{v['name']}: còn {v['stock']}" for v in vs)) if vs else ""
                context.append(ContextBlock(source="product", title=p["name"],
                                            body=f"{p['description'] or ''} {vt}".strip()))
        for c in search_chunks(conn, shop_id, qvec, k=3, bot_id=bot_id):
            if float(c["score"]) >= 0.05:
                context.append(ContextBlock(source="knowledge", title=c["title"], body=c["content"]))
    history = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in (body.history or [])][-8:]
    res = get_llm(ctx.org_id).answer(question=body.text, context=context, history=history, shop_name=shop_name, persona=persona)
    return {"reply": res.text, "model": res.model, "retrieved": len(context)}
