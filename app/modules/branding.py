"""Platform branding — config-driven UI. The platform admin sets the app name,
logo (uploaded image) and accent colour once; every tenant's web app reads them
from the public /api/branding endpoint. No code change, no redeploy."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel

from .. import audit
from ..db import no_tenant
from ..errors import bad_request
from ..tenancy import CurrentUser, require_platform_admin

_LOGO_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/svg+xml"}

router = APIRouter(prefix="/api", tags=["branding"])

DEFAULTS = {"app_name": "OmniShop AI", "accent_color": "#818cf8"}


def _read() -> dict:
    with no_tenant() as conn:
        r = conn.execute(
            "SELECT app_name, logo_asset, accent_color FROM platform_settings WHERE id=1"
        ).fetchone()
    app_name = (r and r["app_name"]) or DEFAULTS["app_name"]
    accent = (r and r["accent_color"]) or DEFAULTS["accent_color"]
    logo_url = f"/api/files/{r['logo_asset']}" if r and r["logo_asset"] else None
    return {"app_name": app_name, "accent_color": accent, "logo_url": logo_url}


@router.get("/branding")
def get_branding():
    """Public — the web app calls this before login to theme itself."""
    return _read()


class BrandingBody(BaseModel):
    app_name: Optional[str] = None
    accent_color: Optional[str] = None
    logo_asset: Optional[str] = None   # file_asset id from /api/uploads; "" clears


@router.get("/admin/branding")
def admin_get_branding(_: CurrentUser = Depends(require_platform_admin)):
    return _read()


@router.put("/admin/branding")
def admin_set_branding(body: BrandingBody, admin: CurrentUser = Depends(require_platform_admin)):
    with no_tenant() as conn:
        cur = conn.execute("SELECT app_name, logo_asset, accent_color FROM platform_settings WHERE id=1").fetchone()
        app_name = body.app_name if body.app_name is not None else (cur and cur["app_name"])
        accent = body.accent_color if body.accent_color is not None else (cur and cur["accent_color"])
        logo = cur["logo_asset"] if cur else None
        if body.logo_asset is not None:
            logo = body.logo_asset or None
        conn.execute(
            "UPDATE platform_settings SET app_name=%s, accent_color=%s, logo_asset=%s WHERE id=1",
            (app_name, accent, logo),
        )
    audit.record("admin.branding.update", actor_user_id=admin.id, detail={"app_name": app_name})
    return _read()


@router.post("/admin/branding/logo")
async def admin_upload_logo(file: UploadFile = File(...), admin: CurrentUser = Depends(require_platform_admin)):
    """Platform admin uploads the system logo (no org context needed)."""
    data = await file.read()
    if not data:
        raise bad_request("tệp rỗng")
    if len(data) > 2 * 1024 * 1024:
        raise bad_request("ảnh tối đa 2MB")
    mime = file.content_type or "image/png"
    if mime not in _LOGO_MIME:
        raise bad_request("chỉ chấp nhận ảnh (PNG/JPG/WebP/GIF/SVG)")
    with no_tenant() as conn:
        row = conn.execute(
            "INSERT INTO file_asset (organization_id, mime, bytes) VALUES (NULL,%s,%s) RETURNING id",
            (mime, data),
        ).fetchone()
        conn.execute("UPDATE platform_settings SET logo_asset=%s WHERE id=1", (row["id"],))
    audit.record("admin.branding.logo", actor_user_id=admin.id)
    return _read()
