"""Facebook Login (OAuth) — the convenient one-button connect for Messenger/IG.

Flow: tenant clicks "Kết nối Facebook" -> we redirect to Facebook's OAuth dialog
-> Facebook calls our callback with a code -> we exchange it for a user token,
list the user's Pages, and auto-create a Messenger channel per Page (token
encrypted, bound to the shop's default bot).

Requires the platform Facebook App (META_APP_ID/SECRET, or admin config) and the
callback URL registered in the app. Manual token entry remains as a fallback.
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import time

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse

from ..config import config
from ..db import no_tenant, tenant_tx
from ..errors import bad_request
from ..providers import registry
from ..providers.channels import meta
from ..security import encrypt_secret
from ..tenancy import OrgContext, require_role
from .bots import get_or_create_default_bot

router = APIRouter(prefix="/api/channels/oauth/meta", tags=["oauth"])
GRAPH = "https://graph.facebook.com/v21.0"
SCOPES = "pages_show_list,pages_messaging,pages_manage_metadata"


def _meta_app() -> dict:
    cfg = registry._load("channel:meta") or {}
    return {"app_id": cfg.get("model") or config.META_APP_ID,
            "app_secret": cfg.get("api_key") or config.META_APP_SECRET}


def _sign_state(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    sig = hmac.new(config.APP_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{raw}.{sig}"


def _verify_state(state: str) -> dict | None:
    try:
        raw, sig = state.split(".")
        if hmac.new(config.APP_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16] != sig:
            return None
        payload = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        if int(payload.get("exp", 0)) < time.time():
            return None
        return payload
    except Exception:  # noqa: BLE001
        return None


@router.get("/start")
def start(shop_id: str, ctx: OrgContext = Depends(require_role("admin"))):
    app = _meta_app()
    if not app["app_id"]:
        raise bad_request("Chưa cấu hình Facebook App. Vào Quản trị hệ thống để nhập App ID/Secret.")
    redirect_uri = config.OAUTH_REDIRECT_BASE.rstrip("/") + "/api/channels/oauth/meta/callback"
    state = _sign_state({"org": ctx.org_id, "shop": shop_id, "exp": int(time.time()) + 600})
    url = (f"https://www.facebook.com/v21.0/dialog/oauth?client_id={app['app_id']}"
           f"&redirect_uri={redirect_uri}&state={state}&scope={SCOPES}&response_type=code")
    return {"url": url}


@router.get("/callback")
def callback(code: str = "", state: str = ""):
    data = _verify_state(state)
    if not data:
        return HTMLResponse("<h3>Liên kết không hợp lệ hoặc đã hết hạn.</h3>", status_code=400)
    app = _meta_app()
    redirect_uri = config.OAUTH_REDIRECT_BASE.rstrip("/") + "/api/channels/oauth/meta/callback"
    try:
        tok = httpx.get(f"{GRAPH}/oauth/access_token", params={
            "client_id": app["app_id"], "client_secret": app["app_secret"],
            "redirect_uri": redirect_uri, "code": code}, timeout=30).json()
        user_token = tok.get("access_token")
        if not user_token:
            return HTMLResponse(f"<h3>Không lấy được token: {tok.get('error',{}).get('message','')}</h3>", status_code=400)
        pages = httpx.get(f"{GRAPH}/me/accounts", params={"access_token": user_token, "fields": "id,name,access_token"},
                          timeout=30).json().get("data", [])
    except Exception as e:  # noqa: BLE001
        return HTMLResponse(f"<h3>Lỗi kết nối Facebook: {e}</h3>", status_code=502)

    org_id, shop_id = data["org"], data["shop"]
    connected = 0
    with tenant_tx(org_id) as conn:
        bot_id = get_or_create_default_bot(conn, org_id, shop_id)
        for pg in pages:
            exists = conn.execute("SELECT 1 FROM channel WHERE config->>'page_id'=%s", (pg["id"],)).fetchone()
            if exists:
                continue
            enc = encrypt_secret(json.dumps({"page_access_token": pg.get("access_token", "")}))
            conn.execute(
                """INSERT INTO channel (organization_id, shop_id, kind, name, credentials_enc, status, config, bot_id)
                   VALUES (%s,%s,'messenger',%s,%s,'connected',%s,%s)""",
                (org_id, shop_id, pg.get("name", "Facebook Page"), enc,
                 json.dumps({"page_id": pg["id"]}), bot_id),
            )
            connected += 1
    dest = config.OAUTH_REDIRECT_BASE.rstrip("/")
    return RedirectResponse(f"{dest}/?connected={connected}", status_code=302)
