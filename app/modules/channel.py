"""Channels: connect a sales channel to a shop. MVP ships the website widget
(instant, no app review). Meta/TikTok/Shopee are added as ChannelProviders later.
"""
from __future__ import annotations
import json
import secrets

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..db import tenant_tx
from ..errors import bad_request
from ..tenancy import OrgContext, get_org_context, require_role
from .billing import channel_allowed

router = APIRouter(prefix="/api", tags=["channel"])


class ChannelBody(BaseModel):
    shop_id: str
    kind: str = "website"
    name: str = "Website widget"
    greeting: str = "Xin chào! Mình có thể giúp gì cho bạn?"


def _assert_shop(conn, shop_id: str):
    if not conn.execute("SELECT 1 FROM shop WHERE id=%s", (shop_id,)).fetchone():
        raise bad_request("shop not found in this organization")


@router.get("/channels")
def list_channels(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, shop_id)
        rows = conn.execute(
            "SELECT id, kind, name, public_key, status, config FROM channel WHERE shop_id=%s ORDER BY created_at",
            (shop_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "kind": r["kind"], "name": r["name"],
             "public_key": r["public_key"], "status": r["status"], "config": r["config"]} for r in rows]


@router.post("/channels")
def create_channel(body: ChannelBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not channel_allowed(ctx.org_id, body.kind):
        raise bad_request(f"channel '{body.kind}' not included in your plan")
    if body.kind != "website":
        # Meta/TikTok/Shopee require OAuth + app review (see platform matrix).
        raise bad_request(f"channel '{body.kind}' not yet available in this build (website only)")
    public_key = "web_" + secrets.token_urlsafe(16)
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, body.shop_id)
        row = conn.execute(
            """INSERT INTO channel (organization_id, shop_id, kind, name, public_key, config)
               VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
            (ctx.org_id, body.shop_id, body.kind, body.name, public_key,
             json.dumps({"greeting": body.greeting})),
        ).fetchone()
    audit.record("channel.create", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=str(row["id"]), detail={"kind": body.kind})
    return {"id": str(row["id"]), "public_key": public_key, "kind": body.kind,
            "greeting": body.greeting}
