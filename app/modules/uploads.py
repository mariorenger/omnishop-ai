"""Minimal object storage for small assets (avatars/logos): stored in Postgres,
served publicly by unguessable id. A local ObjectStorage stand-in; swap for
S3/R2 behind the same URL contract later."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response

from ..db import no_tenant
from ..errors import bad_request, not_found
from ..tenancy import OrgContext, require_role

router = APIRouter(prefix="/api", tags=["uploads"])

_ALLOWED = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/svg+xml"}


@router.post("/uploads")
async def upload(file: UploadFile = File(...), ctx: OrgContext = Depends(require_role("admin"))):
    data = await file.read()
    if not data:
        raise bad_request("tệp rỗng")
    if len(data) > 2 * 1024 * 1024:
        raise bad_request("ảnh tối đa 2MB")
    mime = file.content_type or "image/png"
    if mime not in _ALLOWED:
        raise bad_request("chỉ chấp nhận ảnh (PNG/JPG/WebP/GIF/SVG)")
    with no_tenant() as conn:
        row = conn.execute(
            "INSERT INTO file_asset (organization_id, mime, bytes) VALUES (%s,%s,%s) RETURNING id",
            (ctx.org_id, mime, data),
        ).fetchone()
    return {"url": f"/api/files/{row['id']}", "mime": mime, "size": len(data)}


@router.get("/files/{file_id}")
def serve_file(file_id: str):
    with no_tenant() as conn:
        row = conn.execute("SELECT mime, bytes FROM file_asset WHERE id=%s", (file_id,)).fetchone()
    if not row:
        raise not_found("file not found")
    return Response(content=bytes(row["bytes"]), media_type=row["mime"],
                    headers={"Cache-Control": "public, max-age=86400"})
