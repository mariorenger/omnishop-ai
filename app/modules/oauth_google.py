"""Google Sign-In (OAuth 2.0). One-button login/signup with a Google account.

Flow: web calls /start -> we return Google's consent URL (signed state) -> Google
redirects to /callback with a code -> we exchange it for tokens, read the user's
email from the userinfo endpoint, find-or-create the user (+ a workspace on first
login), issue our JWT, and redirect back to the web app with the token.

Client id/secret come from env (GOOGLE_CLIENT_*) or admin config (scope
'auth:google'). The redirect URI must be registered in the Google Cloud console:
  <OAUTH_REDIRECT_BASE>/api/auth/google/callback
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import time

import httpx
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

from ..config import config
from ..db import no_tenant, tenant_tx
from ..providers import registry
from ..security import issue_token

router = APIRouter(prefix="/api/auth/google", tags=["oauth"])

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def _creds() -> dict:
    cfg = registry._load("auth:google") or {}
    return {"client_id": cfg.get("model") or config.GOOGLE_CLIENT_ID,
            "client_secret": cfg.get("api_key") or config.GOOGLE_CLIENT_SECRET}


def _redirect_uri() -> str:
    return registry.public_base() + "/api/auth/google/callback"


def _web_base() -> str:
    """Where to send the user after Google login: the public domain
    (OAUTH_REDIRECT_BASE) if configured, else a CORS origin, else localhost."""
    base = registry.public_base()
    if base and "localhost" not in base and "127.0.0.1" not in base:
        return base
    for o in (config.CORS_ORIGINS or []):
        if o and o != "*":
            return o.rstrip("/")
    return base or "http://localhost:3000"


def _sign_state(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    sig = hmac.new(config.APP_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{raw}.{sig}"


def _verify_state(state: str):
    try:
        raw, sig = state.split(".")
        if hmac.new(config.APP_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()[:16] != sig:
            return None
        payload = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        return payload if int(payload.get("exp", 0)) >= time.time() else None
    except Exception:  # noqa: BLE001
        return None


@router.get("/config")
def google_config():
    """Public — the login page shows the Google button only when configured."""
    return {"enabled": bool(_creds()["client_id"])}


@router.get("/start")
def start():
    c = _creds()
    if not c["client_id"]:
        return {"error": "Google Sign-In chưa được cấu hình."}
    state = _sign_state({"exp": int(time.time()) + 600})
    from urllib.parse import urlencode
    q = urlencode({"client_id": c["client_id"], "redirect_uri": _redirect_uri(),
                   "response_type": "code", "scope": "openid email profile",
                   "state": state, "access_type": "online", "prompt": "select_account"})
    return {"url": f"{AUTH_URL}?{q}"}


def _find_or_create_user(email: str, name: str = "") -> str:
    email_l = email.lower()
    promote = "admin" if email_l in config.PLATFORM_ADMIN_EMAILS else None
    with no_tenant() as conn:
        row = conn.execute("SELECT id FROM app_user WHERE lower(email)=lower(%s)", (email,)).fetchone()
        if row:
            if promote:
                conn.execute("UPDATE app_user SET platform_role='admin', is_platform_admin=true WHERE id=%s", (row["id"],))
            return str(row["id"])
        user = conn.execute(
            "INSERT INTO app_user (email, full_name, platform_role, is_platform_admin) VALUES (%s,%s,%s,%s) RETURNING id",
            (email, name, promote, promote == "admin"),
        ).fetchone()
        user_id = str(user["id"])
        org = conn.execute("INSERT INTO organization (name) VALUES (%s) RETURNING id",
                           (f"{email.split('@')[0]}'s workspace",)).fetchone()
        org_id = str(org["id"])
        conn.execute("INSERT INTO subscription (organization_id, plan_code) VALUES (%s,'free')", (org_id,))
    with tenant_tx(org_id) as conn:
        conn.execute("INSERT INTO membership (organization_id, user_id, role) VALUES (%s,%s,'owner')", (org_id, user_id))
    return user_id


@router.get("/callback")
def callback(code: str = "", state: str = ""):
    if not _verify_state(state):
        return HTMLResponse("<h3>Liên kết đăng nhập không hợp lệ hoặc đã hết hạn.</h3>", status_code=400)
    c = _creds()
    try:
        tok = httpx.post(TOKEN_URL, data={
            "code": code, "client_id": c["client_id"], "client_secret": c["client_secret"],
            "redirect_uri": _redirect_uri(), "grant_type": "authorization_code"}, timeout=30).json()
        access = tok.get("access_token")
        if not access:
            return HTMLResponse(f"<h3>Không lấy được token: {tok.get('error_description', tok.get('error',''))}</h3>", status_code=400)
        info = httpx.get(USERINFO_URL, headers={"Authorization": f"Bearer {access}"}, timeout=30).json()
    except Exception as e:  # noqa: BLE001
        return HTMLResponse(f"<h3>Lỗi kết nối Google: {e}</h3>", status_code=502)
    email = info.get("email")
    if not email or not info.get("email_verified", True):
        return HTMLResponse("<h3>Không lấy được email đã xác minh từ Google.</h3>", status_code=400)
    user_id = _find_or_create_user(email, info.get("name", ""))
    from .. import audit
    audit.record("auth.login", actor_user_id=user_id, detail={"via": "google"})
    token = issue_token(user_id)
    return RedirectResponse(f"{_web_base()}/?token={token}", status_code=302)
