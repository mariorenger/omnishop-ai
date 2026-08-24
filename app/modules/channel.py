"""Channels: connect a sales channel to a shop.

- website  : instant chat widget (public_key), no credentials.
- messenger / instagram : Meta Graph API (real send/verify/webhook). Needs the
  platform Facebook App + Meta App Review for messaging; tenant supplies a Page
  access token.
- tiktok / shopee : credential capture + storage now; live messaging is gated by
  partner approval (scaffold).

Credentials are encrypted at rest; non-secret routing keys go in `config`.
"""
from __future__ import annotations
import json
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .. import audit
from ..config import config
from ..db import no_tenant, tenant_tx
from ..errors import bad_request, not_found
from ..providers.channels import meta
from ..security import decrypt_secret, encrypt_secret
from ..tenancy import OrgContext, get_org_context, require_role
from . import orchestrator
from .billing import channel_allowed

router = APIRouter(prefix="/api", tags=["channel"])

# Per-kind connection spec: which fields to collect, which are secret/routing.
KIND_SPECS = {
    "website": {"label": "Tiện ích website", "live": True, "note": "", "fields": []},
    "messenger": {
        "label": "Facebook Messenger", "live": True,
        "note": "Cần một Facebook App đã được duyệt quyền nhắn tin. Nhập Page ID và Page Access Token.",
        "fields": [
            {"key": "page_id", "label": "Page ID", "secret": False, "required": True},
            {"key": "page_access_token", "label": "Page Access Token", "secret": True, "required": True},
        ],
    },
    "instagram": {
        "label": "Instagram", "live": True,
        "note": "Dùng chung Facebook App. Nhập IG Page ID và Page Access Token.",
        "fields": [
            {"key": "page_id", "label": "IG Page ID", "secret": False, "required": True},
            {"key": "page_access_token", "label": "Page Access Token", "secret": True, "required": True},
        ],
    },
    "tiktok": {
        "label": "TikTok Shop", "live": False,
        "note": "Đang chờ phê duyệt đối tác TikTok Shop. Lưu thông tin để kích hoạt sau.",
        "fields": [
            {"key": "shop_id", "label": "Shop ID", "secret": False, "required": True},
            {"key": "app_key", "label": "App Key", "secret": False, "required": True},
            {"key": "app_secret", "label": "App Secret", "secret": True, "required": True},
            {"key": "access_token", "label": "Access Token", "secret": True, "required": False},
        ],
    },
    "shopee": {
        "label": "Shopee", "live": False,
        "note": "Đang chờ phê duyệt đối tác Shopee Open Platform. Lưu thông tin để kích hoạt sau.",
        "fields": [
            {"key": "shop_id", "label": "Shop ID", "secret": False, "required": True},
            {"key": "partner_id", "label": "Partner ID", "secret": False, "required": True},
            {"key": "partner_key", "label": "Partner Key", "secret": True, "required": True},
            {"key": "access_token", "label": "Access Token", "secret": True, "required": False},
        ],
    },
}


class ChannelBody(BaseModel):
    shop_id: str
    kind: str = "website"
    name: str = ""
    greeting: str = "Xin chào! Mình có thể giúp gì cho bạn?"
    credentials: dict = {}


def _assert_shop(conn, shop_id: str):
    if not conn.execute("SELECT 1 FROM shop WHERE id=%s", (shop_id,)).fetchone():
        raise bad_request("shop not found in this organization")


@router.get("/channels/kinds")
def channel_kinds(ctx: OrgContext = Depends(get_org_context)):
    out = []
    for kind, spec in KIND_SPECS.items():
        out.append({"kind": kind, "label": spec["label"], "fields": spec["fields"],
                    "live": spec["live"], "note": spec["note"],
                    "allowed": channel_allowed(ctx.org_id, kind)})
    return out


@router.get("/channels")
def list_channels(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, shop_id)
        rows = conn.execute(
            "SELECT id, kind, name, public_key, status, config FROM channel WHERE shop_id=%s ORDER BY created_at",
            (shop_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "kind": r["kind"], "name": r["name"], "public_key": r["public_key"],
             "status": r["status"], "config": r["config"]} for r in rows]


@router.post("/channels")
def create_channel(body: ChannelBody, ctx: OrgContext = Depends(require_role("admin"))):
    kind = body.kind
    spec = KIND_SPECS.get(kind)
    if not spec:
        raise bad_request("unknown channel type")
    if not channel_allowed(ctx.org_id, kind):
        raise bad_request(f"channel '{kind}' not included in your plan")

    public_key = None
    creds: dict = {}
    cfg: dict = {}
    status = "connected"
    note = ""

    if kind == "website":
        public_key = "web_" + secrets.token_urlsafe(16)
        cfg["greeting"] = body.greeting
    else:
        for f in spec["fields"]:
            v = str(body.credentials.get(f["key"], "")).strip()
            if f["required"] and not v:
                raise bad_request(f"thiếu trường bắt buộc: {f['label']}")
            (creds if f["secret"] else cfg)[f["key"]] = v
        if not spec["live"]:
            status = "pending"
        elif kind in ("messenger", "instagram"):
            ok, info = meta.verify_page_token(creds.get("page_access_token", ""))
            status = "connected" if ok else "degraded"
            note = info

    enc = encrypt_secret(json.dumps(creds)) if creds else None
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, body.shop_id)
        row = conn.execute(
            """INSERT INTO channel (organization_id, shop_id, kind, name, public_key, credentials_enc, status, config)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (ctx.org_id, body.shop_id, kind, body.name or spec["label"], public_key, enc, status, json.dumps(cfg)),
        ).fetchone()
    audit.record("channel.connect", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=str(row["id"]), detail={"kind": kind, "status": status})
    return {"id": str(row["id"]), "kind": kind, "status": status, "note": note, "public_key": public_key}


class ChannelUpdate(BaseModel):
    name: Optional[str] = None
    greeting: Optional[str] = None
    credentials: dict = {}


@router.get("/channels/{channel_id}")
def get_channel(channel_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute(
            "SELECT id, kind, name, public_key, status, config, credentials_enc FROM channel WHERE id=%s", (channel_id,)
        ).fetchone()
    if not r:
        raise not_found("channel not found")
    has_creds = bool(r["credentials_enc"])
    return {"id": str(r["id"]), "kind": r["kind"], "name": r["name"], "public_key": r["public_key"],
            "status": r["status"], "config": r["config"], "has_credentials": has_creds}


@router.post("/channels/{channel_id}/verify")
def verify_channel(channel_id: str, ctx: OrgContext = Depends(require_role("admin"))):
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute("SELECT kind, credentials_enc FROM channel WHERE id=%s", (channel_id,)).fetchone()
        if not r:
            raise not_found("channel not found")
        kind = r["kind"]
        if kind not in ("messenger", "instagram"):
            return {"status": r and "connected", "note": "Kênh này không cần kiểm tra token."}
        token = ""
        if r["credentials_enc"]:
            try:
                token = json.loads(decrypt_secret(bytes(r["credentials_enc"]))).get("page_access_token", "")
            except Exception:  # noqa: BLE001
                token = ""
        ok, info = meta.verify_page_token(token) if token else (False, "chưa có token")
        status = "connected" if ok else "degraded"
        conn.execute("UPDATE channel SET status=%s WHERE id=%s", (status, channel_id))
    audit.record("channel.verify", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=channel_id,
                 detail={"status": status})
    return {"status": status, "note": info}


@router.put("/channels/{channel_id}")
def update_channel(channel_id: str, body: ChannelUpdate, ctx: OrgContext = Depends(require_role("admin"))):
    with tenant_tx(ctx.org_id) as conn:
        r = conn.execute("SELECT kind, config, credentials_enc FROM channel WHERE id=%s", (channel_id,)).fetchone()
        if not r:
            raise not_found("channel not found")
        cfg = dict(r["config"] or {})
        if body.greeting is not None:
            cfg["greeting"] = body.greeting
        spec = KIND_SPECS.get(r["kind"], {"fields": []})
        # merge credentials: keep existing, overwrite provided; routing keys -> config
        creds = {}
        if r["credentials_enc"]:
            try:
                creds = json.loads(decrypt_secret(bytes(r["credentials_enc"])))
            except Exception:  # noqa: BLE001
                creds = {}
        for f in spec["fields"]:
            if f["key"] in body.credentials:
                v = str(body.credentials[f["key"]]).strip()
                (creds if f["secret"] else cfg)[f["key"]] = v
        enc = encrypt_secret(json.dumps(creds)) if creds else r["credentials_enc"]
        conn.execute(
            "UPDATE channel SET name=coalesce(%s,name), config=%s, credentials_enc=%s WHERE id=%s",
            (body.name, json.dumps(cfg), enc, channel_id),
        )
    audit.record("channel.update", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=channel_id)
    return {"ok": True}


@router.delete("/channels/{channel_id}")
def delete_channel(channel_id: str, ctx: OrgContext = Depends(require_role("admin"))):
    with tenant_tx(ctx.org_id) as conn:
        if not conn.execute("SELECT 1 FROM channel WHERE id=%s", (channel_id,)).fetchone():
            raise not_found("channel not found")
        conn.execute("DELETE FROM channel WHERE id=%s", (channel_id,))
    audit.record("channel.delete", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=channel_id)
    return {"ok": True}


# --------------------------- Meta webhook (platform-level) ------------------

@router.get("/channels/webhook/meta")
def meta_webhook_verify(request: Request):
    q = request.query_params
    if q.get("hub.mode") == "subscribe" and q.get("hub.verify_token") == config.META_VERIFY_TOKEN:
        return PlainTextResponse(q.get("hub.challenge", ""))
    return PlainTextResponse("forbidden", status_code=403)


@router.post("/channels/webhook/meta")
async def meta_webhook(request: Request):
    body = await request.body()
    if config.META_APP_SECRET and not meta.verify_signature(
        config.META_APP_SECRET, body, request.headers.get("x-hub-signature-256", "")
    ):
        return PlainTextResponse("bad signature", status_code=403)
    payload = json.loads(body or b"{}")
    for page_id, sender, text in meta.normalize_entries(payload):
        with no_tenant() as conn:
            ch = conn.execute("SELECT * FROM resolve_channel_by_meta(%s)", (page_id,)).fetchone()
        if not ch:
            continue
        result = orchestrator.handle_incoming(
            str(ch["organization_id"]), str(ch["shop_id"]), str(ch["channel_id"]), sender, text
        )
        token = ""
        if ch["credentials_enc"]:
            try:
                token = json.loads(decrypt_secret(bytes(ch["credentials_enc"]))).get("page_access_token", "")
            except Exception:  # noqa: BLE001
                token = ""
        if token:
            meta.send_message(token, sender, result["reply"])
    return {"ok": True}
