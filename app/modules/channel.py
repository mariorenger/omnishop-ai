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
from ..providers.channels import meta, telegram, whatsapp, zalo
from ..security import decrypt_secret, encrypt_secret
from ..tenancy import OrgContext, get_org_context, require_role
from . import orchestrator
from .billing import channel_allowed
from .bots import get_or_create_default_bot

router = APIRouter(prefix="/api", tags=["channel"])

# Per-kind connection spec: which fields to collect, which are secret/routing.
# Fields, notes and doc links follow each platform's official integration docs so a
# tenant can self-configure the connection to THEIR own endpoint. `hint` on a field
# tells the user exactly where to find that value.
KIND_SPECS = {
    "website": {"label": "Tiện ích website", "live": True, "note": "",
                "docs": "", "fields": []},
    "messenger": {
        "label": "Facebook Messenger", "live": True,
        "note": "Kết nối nhanh bằng nút \"Kết nối Facebook\" (OAuth), hoặc nhập thủ công Page ID và "
                "Page Access Token từ Facebook App của bạn (quyền pages_messaging).",
        "docs": "https://developers.facebook.com/docs/messenger-platform/get-started",
        "fields": [
            {"key": "page_id", "label": "Page ID", "secret": False, "required": True,
             "hint": "ID trang Facebook — xem trong Meta Business Suite → Cài đặt trang."},
            {"key": "page_access_token", "label": "Page Access Token", "secret": True, "required": True,
             "hint": "Tạo trong Facebook App → Messenger → Access Tokens (token của Trang)."},
        ],
    },
    "instagram": {
        "label": "Instagram", "live": True,
        "note": "Dùng chung Facebook App (Instagram Messaging). Instagram phải là tài khoản Doanh nghiệp "
                "liên kết với một Trang Facebook. Nhập IG Page ID và Page Access Token.",
        "docs": "https://developers.facebook.com/docs/messenger-platform/instagram",
        "fields": [
            {"key": "page_id", "label": "IG-linked Page ID", "secret": False, "required": True,
             "hint": "ID Trang Facebook được liên kết với tài khoản Instagram doanh nghiệp."},
            {"key": "page_access_token", "label": "Page Access Token", "secret": True, "required": True,
             "hint": "Token của Trang, có quyền instagram_manage_messages."},
        ],
    },
    "telegram": {
        "label": "Telegram", "live": True,
        "note": "Chat với @BotFather → /newbot → dán Bot Token vào đây. Webhook sẽ tự đăng ký. "
                "Không cần phê duyệt — chạy thật ngay.",
        "docs": "https://core.telegram.org/bots/api",
        "fields": [
            {"key": "bot_token", "label": "Bot Token", "secret": True, "required": True,
             "hint": "Dạng 123456789:AA... do @BotFather cấp khi tạo bot."},
        ],
    },
    "zalo": {
        "label": "Zalo OA", "live": True,
        "note": "Cần Official Account đã kết nối tới một Zalo App (developers.zalo.me). Access Token OA hết "
                "hạn ~1 ngày, cần làm mới bằng refresh token; nhập App Secret để hệ thống xác thực webhook.",
        "docs": "https://developers.zalo.me/docs/official-account/bat-dau",
        "fields": [
            {"key": "oa_id", "label": "OA ID", "secret": False, "required": True,
             "hint": "ID Official Account — trong Zalo OA Manager → Thông tin OA."},
            {"key": "access_token", "label": "OA Access Token", "secret": True, "required": True,
             "hint": "Lấy qua OAuth v4 (oauth.zaloapp.com/v4/oa/access_token). Hết hạn ~25 giờ."},
            {"key": "app_secret", "label": "App Secret (xác thực webhook)", "secret": True, "required": False,
             "hint": "App Secret của Zalo App — dùng kiểm tra chữ ký X-ZEvent-Signature. Tuỳ chọn."},
        ],
    },
    "whatsapp": {
        "label": "WhatsApp Cloud", "live": True,
        "note": "Dùng WhatsApp Business Account trên Meta. Webhook dùng chung cấu hình Meta (verify token + "
                "App Secret của nền tảng). Nhập Phone Number ID và Access Token.",
        "docs": "https://developers.facebook.com/docs/whatsapp/cloud-api/get-started",
        "fields": [
            {"key": "phone_number_id", "label": "Phone Number ID", "secret": False, "required": True,
             "hint": "ID nội bộ của số gửi (KHÔNG phải số điện thoại) — WhatsApp Manager → API Setup."},
            {"key": "access_token", "label": "Access Token", "secret": True, "required": True,
             "hint": "Token hệ thống (System User) dài hạn, hoặc token tạm 24 giờ khi thử nghiệm."},
        ],
    },
    "tiktok": {
        "label": "TikTok Shop", "live": False,
        "note": "Cần App trên TikTok Shop Partner Center + shop uỷ quyền (OAuth) để có access_token và "
                "shop_cipher. Mọi request phải ký (sign) bằng App Secret. Lưu thông tin để kích hoạt khi "
                "app được duyệt.",
        "docs": "https://partner.tiktokshop.com/docv2/page/authorization-overview-202407",
        "fields": [
            {"key": "shop_id", "label": "Shop ID", "secret": False, "required": True,
             "hint": "ID cửa hàng TikTok Shop nhận được sau khi uỷ quyền."},
            {"key": "shop_cipher", "label": "Shop Cipher", "secret": False, "required": False,
             "hint": "Mã shop_cipher trả về cùng access_token, dùng trong hầu hết API v2."},
            {"key": "app_key", "label": "App Key", "secret": False, "required": True,
             "hint": "Cấp khi tạo App trong Partner Center."},
            {"key": "app_secret", "label": "App Secret", "secret": True, "required": True,
             "hint": "Dùng để ký HMAC-SHA256 mọi request."},
            {"key": "access_token", "label": "Access Token", "secret": True, "required": False,
             "hint": "Nhận qua OAuth; cần refresh định kỳ."},
        ],
    },
    "shopee": {
        "label": "Shopee", "live": False,
        "note": "Cần App trên Shopee Open Platform + shop uỷ quyền (OAuth). Access token hết hạn 4 giờ (cần "
                "refresh). Mọi request ký HMAC-SHA256 từ Partner Key. Lưu thông tin để kích hoạt khi duyệt.",
        "docs": "https://open.shopee.com/documents",
        "fields": [
            {"key": "shop_id", "label": "Shop ID", "secret": False, "required": True,
             "hint": "ID cửa hàng nhận được sau khi shop uỷ quyền cho app."},
            {"key": "partner_id", "label": "Partner ID", "secret": False, "required": True,
             "hint": "Partner ID của app trong Shopee Open Platform Console."},
            {"key": "partner_key", "label": "Partner Key", "secret": True, "required": True,
             "hint": "Khoá ký HMAC-SHA256 (partner_id + path + timestamp + access_token + shop_id)."},
            {"key": "access_token", "label": "Access Token", "secret": True, "required": False,
             "hint": "Nhận qua OAuth; hết hạn 4 giờ, làm mới bằng refresh token."},
        ],
    },
}


class ChannelBody(BaseModel):
    shop_id: str
    kind: str = "website"
    name: str = ""
    greeting: str = "Xin chào! Mình có thể giúp gì cho bạn?"
    credentials: dict = {}
    bot_id: Optional[str] = None


def _assert_shop(conn, shop_id: str):
    if not conn.execute("SELECT 1 FROM shop WHERE id=%s", (shop_id,)).fetchone():
        raise bad_request("shop not found in this organization")


def _webhook_url(kind: str) -> str:
    """The public webhook URL a tenant must paste into the platform's console
    (empty for kinds that need none / auto-register)."""
    base = config.OAUTH_REDIRECT_BASE.rstrip("/")
    if not base:
        return ""
    if kind == "zalo":
        return f"{base}/api/channels/webhook/zalo"
    if kind in ("messenger", "instagram", "whatsapp"):
        return f"{base}/api/channels/webhook/meta"
    return ""   # website: none · telegram: auto-registered · tiktok/shopee: gated


@router.get("/channels/kinds")
def channel_kinds(ctx: OrgContext = Depends(get_org_context)):
    out = []
    for kind, spec in KIND_SPECS.items():
        out.append({"kind": kind, "label": spec["label"], "fields": spec["fields"],
                    "live": spec["live"], "note": spec["note"], "docs": spec.get("docs", ""),
                    "webhook_url": _webhook_url(kind),
                    "allowed": channel_allowed(ctx.org_id, kind)})
    return out


@router.get("/channels")
def list_channels(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, shop_id)
        rows = conn.execute(
            """SELECT c.id, c.kind, c.name, c.public_key, c.status, c.config, c.bot_id, b.name AS bot_name
               FROM channel c LEFT JOIN bot b ON b.id = c.bot_id
               WHERE c.shop_id=%s ORDER BY c.created_at""",
            (shop_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "kind": r["kind"], "name": r["name"], "public_key": r["public_key"],
             "status": r["status"], "config": r["config"],
             "bot_id": str(r["bot_id"]) if r["bot_id"] else None, "bot_name": r["bot_name"]} for r in rows]


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
            if ok and cfg.get("page_id"):
                sok, sinfo = meta.subscribe_page(creds.get("page_access_token", ""), cfg["page_id"])
                note = f"{info} · {'Trang đã đăng ký nhận tin' if sok else 'CHƯA đăng ký Trang: ' + sinfo}"
        elif kind == "telegram":
            public_key = "tg_" + secrets.token_urlsafe(16)
            ok, info = telegram.verify_token(creds.get("bot_token", ""))
            status = "connected" if ok else "degraded"
            note = info
            # auto-register the Telegram webhook — needs a PUBLIC HTTPS base so
            # Telegram can reach us (localhost/http can't receive messages).
            base = config.OAUTH_REDIRECT_BASE.rstrip("/")
            good_base = base.startswith("https://") and "localhost" not in base and "127.0.0.1" not in base
            if ok and good_base:
                hook = f"{base}/api/channels/webhook/telegram/{public_key}"
                wok, winfo = telegram.set_webhook(creds["bot_token"], hook)
                note = f"{info} · webhook: {'đã đặt' if wok else winfo}"
            elif ok:
                status = "degraded"
                note = (f"{info} · Token OK nhưng chưa nhận được tin: đặt OAUTH_REDIRECT_BASE = "
                        f"https://tên-miền rồi khởi động lại, sau đó bấm Kiểm tra kết nối.")
        elif kind == "zalo":
            ok, info = zalo.verify_token(creds.get("access_token", ""))
            status = "connected" if ok else "degraded"
            note = info
        elif kind == "whatsapp":
            ok, info = whatsapp.verify_token(creds.get("access_token", ""), cfg.get("phone_number_id", ""))
            status = "connected" if ok else "degraded"
            note = info

    enc = encrypt_secret(json.dumps(creds)) if creds else None
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, body.shop_id)
        # bind a bot: the one requested (must belong to the shop) or a default one
        bot_id = body.bot_id
        if bot_id:
            if not conn.execute("SELECT 1 FROM bot WHERE id=%s AND shop_id=%s", (bot_id, body.shop_id)).fetchone():
                raise bad_request("bot not found in this shop")
        else:
            bot_id = get_or_create_default_bot(conn, ctx.org_id, body.shop_id)
        # website greeting mirrors the bot greeting if not customized
        if kind == "website":
            b = conn.execute("SELECT greeting FROM bot WHERE id=%s", (bot_id,)).fetchone()
            if b and b["greeting"]:
                cfg["greeting"] = cfg.get("greeting") or b["greeting"]
        row = conn.execute(
            """INSERT INTO channel (organization_id, shop_id, kind, name, public_key, credentials_enc, status, config, bot_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (ctx.org_id, body.shop_id, kind, body.name or spec["label"], public_key, enc, status, json.dumps(cfg), bot_id),
        ).fetchone()
    audit.record("channel.connect", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=str(row["id"]), detail={"kind": kind, "status": status})
    return {"id": str(row["id"]), "kind": kind, "status": status, "note": note, "public_key": public_key}


class ChannelUpdate(BaseModel):
    name: Optional[str] = None
    greeting: Optional[str] = None
    credentials: dict = {}
    bot_id: Optional[str] = None


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
        r = conn.execute("SELECT kind, credentials_enc, config, public_key FROM channel WHERE id=%s", (channel_id,)).fetchone()
        if not r:
            raise not_found("channel not found")
        kind = r["kind"]
        if kind not in ("messenger", "instagram", "telegram", "zalo", "whatsapp"):
            return {"status": "connected", "note": "Kênh này không cần kiểm tra token."}
        creds = {}
        if r["credentials_enc"]:
            try:
                creds = json.loads(decrypt_secret(bytes(r["credentials_enc"])))
            except Exception:  # noqa: BLE001
                creds = {}
        cfg = dict(r["config"] or {})
        base = config.OAUTH_REDIRECT_BASE.rstrip("/")
        good_base = base.startswith("https://") and "localhost" not in base and "127.0.0.1" not in base
        if kind in ("messenger", "instagram"):
            tok = creds.get("page_access_token", "")
            if not tok:
                ok, info = False, "chưa có token"
            else:
                ok, info = meta.verify_page_token(tok)
                if ok:
                    # subscribe the Page to our app so it actually delivers messages
                    pid = cfg.get("page_id", "")
                    if pid:
                        sok, sinfo = meta.subscribe_page(tok, pid)
                        info = f"{info} · {'Trang đã đăng ký nhận tin' if sok else 'CHƯA đăng ký được Trang: ' + sinfo}"
                    info += (f" · Webhook (Admin cấu hình 1 lần trong Facebook App → {base}/api/channels/webhook/meta)"
                             if good_base else " · Cần OAUTH_REDIRECT_BASE=https://tên-miền để nhận tin")
        elif kind == "telegram":
            tok = creds.get("bot_token", "")
            if not tok:
                ok, info = False, "chưa có token"
            else:
                ok, info = telegram.verify_token(tok)
                # Only a valid, publicly-reachable HTTPS webhook lets the bot RECEIVE
                # messages. Re-register it here (the domain may have been set after
                # connecting) and report Telegram's own view of the webhook.
                if ok:
                    base = config.OAUTH_REDIRECT_BASE.rstrip("/")
                    good_base = base.startswith("https://") and "localhost" not in base and "127.0.0.1" not in base
                    if good_base and r["public_key"]:
                        hook = f"{base}/api/channels/webhook/telegram/{r['public_key']}"
                        telegram.set_webhook(tok, hook)
                        wi = telegram.webhook_info(tok)
                        if wi["last_error"]:
                            ok = False
                            info = f"{info} · webhook LỖI: {wi['last_error']} (URL: {wi['url'] or hook})"
                        elif wi["url"]:
                            info = f"{info} · webhook OK ({wi['pending']} tin đang chờ)"
                        else:
                            info = f"{info} · chưa đặt được webhook"
                    else:
                        ok = False
                        info = (f"{info} · Token hợp lệ NHƯNG chưa nhận được tin: cần đặt "
                                f"OAUTH_REDIRECT_BASE = https://tên-miền-của-bạn rồi khởi động lại "
                                f"(hiện tại: '{base or 'trống'}').")
        elif kind == "zalo":
            tok = creds.get("access_token", "")
            ok, info = zalo.verify_token(tok) if tok else (False, "chưa có token")
            if ok:
                info += (f" · Đặt Webhook URL trong Zalo OA/Developer Console → {base}/api/channels/webhook/zalo"
                         if good_base else " · Cần OAUTH_REDIRECT_BASE=https://tên-miền để nhận tin")
        else:  # whatsapp
            tok = creds.get("access_token", "")
            ok, info = whatsapp.verify_token(tok, cfg.get("phone_number_id", "")) if tok else (False, "chưa có token")
            if ok:
                info += (f" · Webhook (dùng chung Meta App → {base}/api/channels/webhook/meta)"
                         if good_base else " · Cần OAUTH_REDIRECT_BASE=https://tên-miền để nhận tin")
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
        if body.bot_id is not None:
            if body.bot_id and not conn.execute("SELECT 1 FROM bot WHERE id=%s", (body.bot_id,)).fetchone():
                raise bad_request("bot not found")
            conn.execute("UPDATE channel SET bot_id=%s WHERE id=%s", (body.bot_id or None, channel_id))
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
    # WhatsApp Cloud shares this app + endpoint but a different body shape.
    if payload.get("object") == "whatsapp_business_account":
        for pnid, sender, text, name in whatsapp.normalize_entries(payload):
            row = _resolve_by_cfg("whatsapp", "phone_number_id", pnid)
            if not row:
                continue
            result = orchestrator.handle_incoming(
                str(row["organization_id"]), str(row["shop_id"]), str(row["id"]), sender, text,
                customer_name=name or None
            )
            creds = _creds(row["credentials_enc"])
            if creds.get("access_token"):
                whatsapp.send_message(creds["access_token"], pnid, sender, result["reply"])
        return {"ok": True}
    for page_id, sender, text in meta.normalize_entries(payload):
        with no_tenant() as conn:
            ch = conn.execute("SELECT * FROM resolve_channel_by_meta(%s)", (page_id,)).fetchone()
        if not ch:
            continue
        result = orchestrator.handle_incoming(
            str(ch["organization_id"]), str(ch["shop_id"]), str(ch["channel_id"]), sender, text
        )
        token = _creds(ch["credentials_enc"]).get("page_access_token", "")
        if token:
            meta.send_message(token, sender, result["reply"])
    return {"ok": True}


# --------------------------- other channel webhooks -------------------------

def _creds(enc) -> dict:
    if not enc:
        return {}
    try:
        return json.loads(decrypt_secret(bytes(enc)))
    except Exception:  # noqa: BLE001
        return {}


def deliver_agent_reply(org_id: str, channel_id: str, customer_ref: str, text: str):
    """Push a human/agent reply OUT to the external channel the customer is on.
    Website returns (True, 'widget') because the widget polls for new messages."""
    with tenant_tx(org_id) as conn:
        r = conn.execute("SELECT kind, credentials_enc, config FROM channel WHERE id=%s", (channel_id,)).fetchone()
    if not r:
        return False, "không tìm thấy kênh"
    kind = r["kind"]
    creds = _creds(r["credentials_enc"])
    cfg = dict(r["config"] or {})
    try:
        if kind == "website":
            return True, "widget"
        if kind in ("messenger", "instagram"):
            return meta.send_message(creds.get("page_access_token", ""), customer_ref, text)
        if kind == "telegram":
            return telegram.send_message(creds.get("bot_token", ""), customer_ref, text)
        if kind == "zalo":
            return zalo.send_message(creds.get("access_token", ""), customer_ref, text)
        if kind == "whatsapp":
            return whatsapp.send_message(creds.get("access_token", ""), cfg.get("phone_number_id", ""), customer_ref, text)
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    return False, f"kênh {kind} chưa hỗ trợ gửi tin đi"


def _resolve_by_cfg(kind: str, key: str, value: str):
    """Resolve a live channel by a non-secret routing key in its config, without a
    tenant context (webhooks are anonymous). Uses the superuser connection."""
    from ..db import admin_tx
    if not value:
        return None
    with admin_tx() as conn:
        return conn.execute(
            "SELECT id, organization_id, shop_id, credentials_enc FROM channel "
            "WHERE kind=%s AND config->>%s = %s LIMIT 1",
            (kind, key, value),
        ).fetchone()


@router.post("/channels/webhook/telegram/{public_key}")
async def telegram_webhook(public_key: str, request: Request):
    from ..db import admin_tx
    update = json.loads(await request.body() or b"{}")
    chat_id, text = telegram.normalize_update(update)
    if not chat_id:
        return {"ok": True}
    with admin_tx() as conn:
        ch = conn.execute(
            "SELECT id, organization_id, shop_id, credentials_enc FROM channel "
            "WHERE kind='telegram' AND public_key=%s LIMIT 1",
            (public_key,),
        ).fetchone()
    if not ch:
        return {"ok": True}
    result = orchestrator.handle_incoming(
        str(ch["organization_id"]), str(ch["shop_id"]), str(ch["id"]), chat_id, text,
        customer_name=telegram.sender_name(update) or None
    )
    token = _creds(ch["credentials_enc"]).get("bot_token", "")
    if token:
        telegram.send_message(token, chat_id, result["reply"])
    return {"ok": True}


@router.post("/channels/webhook/zalo")
async def zalo_webhook(request: Request):
    payload = json.loads(await request.body() or b"{}")
    oa_id, sender, text = zalo.normalize_event(payload)
    if not oa_id:
        return {"ok": True}
    row = _resolve_by_cfg("zalo", "oa_id", oa_id)
    if not row:
        return {"ok": True}
    result = orchestrator.handle_incoming(
        str(row["organization_id"]), str(row["shop_id"]), str(row["id"]), sender, text
    )
    creds = _creds(row["credentials_enc"])
    if creds.get("access_token"):
        zalo.send_message(creds["access_token"], sender, result["reply"])
    return {"ok": True}
