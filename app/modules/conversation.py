"""Conversation endpoints: public widget (no auth) + agent inbox (auth)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..db import no_tenant, tenant_tx
from ..errors import bad_request, not_found
from ..tenancy import OrgContext, get_org_context, require_role
from . import orchestrator

router = APIRouter(prefix="/api", tags=["conversation"])


# ---- public widget (no auth; keyed by channel public_key) -----------------

class WidgetMessage(BaseModel):
    session_id: str
    text: str


def _resolve_channel(public_key: str) -> dict:
    with no_tenant() as conn:
        row = conn.execute(
            "SELECT * FROM resolve_channel_by_public_key(%s)", (public_key,)
        ).fetchone()
    if not row:
        raise not_found("unknown widget key")
    return {"channel_id": str(row["channel_id"]), "organization_id": str(row["organization_id"]),
            "shop_id": str(row["shop_id"]), "status": row["status"], "config": row["config"]}


@router.get("/widget/{public_key}/config")
def widget_config(public_key: str):
    ch = _resolve_channel(public_key)
    return {"greeting": (ch["config"] or {}).get("greeting", "Xin chào!"),
            "status": ch["status"]}


@router.post("/widget/{public_key}/message")
def widget_message(public_key: str, body: WidgetMessage):
    if not body.text.strip():
        raise bad_request("empty message")
    ch = _resolve_channel(public_key)
    result = orchestrator.handle_incoming(
        ch["organization_id"], ch["shop_id"], ch["channel_id"], body.session_id, body.text
    )
    return {"reply": result["reply"], "status": result["status"]}


@router.get("/widget/{public_key}/poll")
def widget_poll(public_key: str, session_id: str, after: str = ""):
    """Widget polls for agent/AI messages (so human handoff replies appear)."""
    ch = _resolve_channel(public_key)
    with tenant_tx(ch["organization_id"]) as conn:
        conv = conn.execute(
            "SELECT id FROM conversation WHERE channel_id=%s AND customer_ref=%s ORDER BY created_at DESC LIMIT 1",
            (ch["channel_id"], session_id),
        ).fetchone()
        if not conv:
            return {"messages": []}
        params = [str(conv["id"])]
        clause = ""
        if after:
            clause = "AND created_at > %s"
            params.append(after)
        rows = conn.execute(
            f"""SELECT role, content, created_at FROM message
                WHERE conversation_id=%s AND role IN ('ai','agent','system') {clause}
                ORDER BY created_at""",
            tuple(params),
        ).fetchall()
    return {"messages": [{"role": r["role"], "content": r["content"],
                          "at": r["created_at"].isoformat()} for r in rows]}


# ---- agent inbox (auth) ----------------------------------------------------

class AgentReply(BaseModel):
    text: str


@router.get("/conversations")
def list_conversations(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        rows = conn.execute(
            """SELECT c.id, c.customer_ref, c.status, c.last_at,
                      (SELECT content FROM message m WHERE m.conversation_id=c.id ORDER BY created_at DESC LIMIT 1) AS last_message
               FROM conversation c WHERE c.shop_id=%s ORDER BY c.last_at DESC LIMIT 100""",
            (shop_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "customer_ref": r["customer_ref"], "status": r["status"],
             "last_at": r["last_at"].isoformat(), "last_message": r["last_message"]} for r in rows]


@router.get("/conversations/{conv_id}/messages")
def conversation_messages(conv_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM conversation WHERE id=%s", (conv_id,)).fetchone():
            raise not_found()
        rows = conn.execute(
            "SELECT role, content, meta, created_at FROM message WHERE conversation_id=%s ORDER BY created_at",
            (conv_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "meta": r["meta"],
             "at": r["created_at"].isoformat()} for r in rows]


@router.post("/conversations/{conv_id}/reply")
def agent_reply(conv_id: str, body: AgentReply, ctx: OrgContext = Depends(require_role("agent"))):
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM conversation WHERE id=%s", (conv_id,)).fetchone():
            raise not_found()
        conn.execute(
            "INSERT INTO message (organization_id, conversation_id, role, content) VALUES (%s,%s,'agent',%s)",
            (ctx.org_id, conv_id, body.text),
        )
        conn.execute("UPDATE conversation SET status='human', last_at=now() WHERE id=%s", (conv_id,))
    return {"ok": True}


@router.post("/conversations/{conv_id}/close")
def close_conversation(conv_id: str, ctx: OrgContext = Depends(require_role("agent"))):
    with tenant_tx(ctx.org_id) as conn:
        conn.execute("UPDATE conversation SET status='closed', last_at=now() WHERE id=%s", (conv_id,))
    return {"ok": True}
