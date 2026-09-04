"""Conversation endpoints: public widget (no auth) + agent inbox (auth)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import ratelimit
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
            "shop_id": str(row["shop_id"]), "status": row["status"], "config": row["config"],
            "bot_id": str(row["bot_id"]) if row["bot_id"] else None}


@router.get("/widget/{public_key}/config")
def widget_config(public_key: str):
    ch = _resolve_channel(public_key)
    greeting = (ch["config"] or {}).get("greeting", "Xin chào!")
    appearance = {"name": "Trợ lý", "avatar_url": "", "accent_color": "#6d7cff"}
    if ch["bot_id"]:
        with tenant_tx(ch["organization_id"]) as conn:
            b = conn.execute("SELECT name, greeting, avatar_url, accent_color FROM bot WHERE id=%s", (ch["bot_id"],)).fetchone()
        if b:
            appearance = {"name": b["name"], "avatar_url": b["avatar_url"] or "", "accent_color": b["accent_color"]}
            greeting = b["greeting"] or greeting
    return {"greeting": greeting, "status": ch["status"], **appearance}


@router.post("/widget/{public_key}/message")
def widget_message(public_key: str, body: WidgetMessage):
    ratelimit.check(f"widget:{public_key}", limit=30, window_s=60)
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
            """SELECT c.id, c.customer_ref, c.status, c.last_at, c.assigned_user_id, c.channel_id,
                      (SELECT email FROM app_user u WHERE u.id=c.assigned_user_id) AS assignee,
                      (SELECT kind FROM channel ch WHERE ch.id=c.channel_id) AS channel_kind,
                      (SELECT content FROM message m WHERE m.conversation_id=c.id ORDER BY created_at DESC LIMIT 1) AS last_message
               FROM conversation c WHERE c.shop_id=%s ORDER BY c.last_at DESC LIMIT 100""",
            (shop_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "customer_ref": r["customer_ref"], "status": r["status"],
             "last_at": r["last_at"].isoformat(), "last_message": r["last_message"],
             "channel_kind": r["channel_kind"],
             "assigned_user_id": str(r["assigned_user_id"]) if r["assigned_user_id"] else None,
             "assignee": r["assignee"]} for r in rows]


@router.get("/conversations/{conv_id}/messages")
def conversation_messages(conv_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM conversation WHERE id=%s", (conv_id,)).fetchone():
            raise not_found()
        rows = conn.execute(
            """SELECT m.role, m.content, m.meta, m.created_at,
                      (SELECT email FROM app_user u WHERE u.id=m.sender_user_id) AS sender
               FROM message m WHERE m.conversation_id=%s ORDER BY m.created_at""",
            (conv_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "meta": r["meta"],
             "sender": r["sender"], "at": r["created_at"].isoformat()} for r in rows]


class AssignBody(BaseModel):
    user_id: str | None = None   # None = claim to self


@router.post("/conversations/{conv_id}/assign")
def assign_conversation(conv_id: str, body: AssignBody, ctx: OrgContext = Depends(require_role("agent"))):
    """Claim a conversation (or, for admin/owner, assign it to another member).
    Controls who is responsible for replying in a shared inbox."""
    target = body.user_id or ctx.user.id
    # only admin/owner may assign to someone else
    if target != ctx.user.id and ctx.role not in ("admin", "owner"):
        raise bad_request("chỉ Quản trị/Chủ sở hữu mới gán cho người khác")
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM conversation WHERE id=%s", (conv_id,)).fetchone():
            raise not_found()
        if target and not conn.execute("SELECT 1 FROM membership WHERE user_id=%s", (target,)).fetchone():
            raise bad_request("người được gán phải là thành viên của workspace")
        conn.execute("UPDATE conversation SET assigned_user_id=%s WHERE id=%s", (target or None, conv_id))
        email = None
        if target:
            row = conn.execute("SELECT email FROM app_user WHERE id=%s", (target,)).fetchone()
            email = row["email"] if row else None
    from .. import audit
    audit.record("conversation.assign", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=conv_id, detail={"assigned_to": email})
    return {"ok": True, "assigned_user_id": target, "assignee": email}


@router.post("/conversations/{conv_id}/reply")
def agent_reply(conv_id: str, body: AgentReply, ctx: OrgContext = Depends(require_role("agent"))):
    if not body.text.strip():
        raise bad_request("nội dung trả lời trống")
    with tenant_tx(ctx.org_id) as conn:
        conv = conn.execute(
            "SELECT channel_id, customer_ref, assigned_user_id FROM conversation WHERE id=%s", (conv_id,)
        ).fetchone()
        if not conv:
            raise not_found()
        conn.execute(
            "INSERT INTO message (organization_id, conversation_id, role, content, sender_user_id) VALUES (%s,%s,'agent',%s,%s)",
            (ctx.org_id, conv_id, body.text, ctx.user.id),
        )
        # replying auto-claims an unassigned conversation to the replier
        assign_clause = "" if conv["assigned_user_id"] else ", assigned_user_id=%s"
        params = [conv_id]
        if not conv["assigned_user_id"]:
            params = [ctx.user.id, conv_id]
        conn.execute(f"UPDATE conversation SET status='human', last_at=now(){assign_clause} WHERE id=%s", tuple(params))
    # push the reply OUT to the customer's channel (Telegram/Messenger/Zalo/WhatsApp)
    from . import channel as channel_mod
    delivered, note = channel_mod.deliver_agent_reply(
        ctx.org_id, str(conv["channel_id"]), conv["customer_ref"], body.text)
    return {"ok": True, "delivered": bool(delivered), "note": note}


@router.post("/conversations/{conv_id}/close")
def close_conversation(conv_id: str, ctx: OrgContext = Depends(require_role("agent"))):
    with tenant_tx(ctx.org_id) as conn:
        conn.execute("UPDATE conversation SET status='closed', last_at=now() WHERE id=%s", (conv_id,))
    return {"ok": True}


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, ctx: OrgContext = Depends(require_role("agent"))):
    # GDPR-style deletion: removes the conversation and its messages (cascade).
    from .. import audit
    with tenant_tx(ctx.org_id) as conn:
        conn.execute("DELETE FROM conversation WHERE id=%s", (conv_id,))
    audit.record("conversation.delete", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=conv_id)
    return {"ok": True}


@router.delete("/customers")
def delete_customer_data(shop_id: str, customer_ref: str, ctx: OrgContext = Depends(require_role("admin"))):
    # Erase all data for one customer (all their conversations in a shop).
    from .. import audit
    with tenant_tx(ctx.org_id) as conn:
        n = conn.execute("DELETE FROM conversation WHERE shop_id=%s AND customer_ref=%s", (shop_id, customer_ref)).rowcount
    audit.record("customer.delete", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=customer_ref, detail={"conversations": n})
    return {"ok": True, "deleted_conversations": n}
