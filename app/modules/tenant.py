"""Tenant resources: organizations (list), shops, members."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..db import no_tenant, tenant_tx
from ..errors import bad_request
from ..tenancy import CurrentUser, OrgContext, get_current_user, get_org_context, require_role
from .auth import _my_orgs
from .billing import resolve_entitlements

router = APIRouter(prefix="/api", tags=["tenant"])


class ShopBody(BaseModel):
    name: str


class OrgBody(BaseModel):
    name: str


@router.post("/orgs")
def create_org(body: OrgBody, user: CurrentUser = Depends(get_current_user)):
    """Create a new workspace owned by the current user (any logged-in user,
    including a platform admin who also wants a merchant workspace)."""
    name = (body.name or "").strip() or f"{user.email.split('@')[0]}'s workspace"
    with no_tenant() as conn:
        org = conn.execute("INSERT INTO organization (name) VALUES (%s) RETURNING id", (name,)).fetchone()
        org_id = str(org["id"])
        conn.execute("INSERT INTO subscription (organization_id, plan_code) VALUES (%s,'free')", (org_id,))
    with tenant_tx(org_id) as conn:
        conn.execute("INSERT INTO membership (organization_id, user_id, role) VALUES (%s,%s,'owner')",
                     (org_id, user.id))
    audit.record("org.create", organization_id=org_id, actor_user_id=user.id, detail={"name": name})
    return {"id": org_id, "name": name, "role": "owner"}


class MemberBody(BaseModel):
    email: str
    role: str = "agent"


@router.get("/orgs")
def list_orgs(user: CurrentUser = Depends(get_current_user)):
    return _my_orgs(user.id)


@router.get("/shops")
def list_shops(ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        rows = conn.execute(
            "SELECT id, name, created_at FROM shop ORDER BY created_at"
        ).fetchall()
    return [{"id": str(r["id"]), "name": r["name"]} for r in rows]


@router.post("/shops")
def create_shop(body: ShopBody, ctx: OrgContext = Depends(require_role("admin"))):
    ent = resolve_entitlements(ctx.org_id)
    with tenant_tx(ctx.org_id) as conn:
        count = conn.execute("SELECT count(*) AS n FROM shop").fetchone()["n"]
        if count >= int(ent.get("shops", 1)):
            raise bad_request(f"plan '{ent['_plan']}' allows {ent.get('shops',1)} shop(s)")
        row = conn.execute(
            "INSERT INTO shop (organization_id, name) VALUES (%s,%s) RETURNING id",
            (ctx.org_id, body.name),
        ).fetchone()
    audit.record("shop.create", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=str(row["id"]), detail={"name": body.name})
    return {"id": str(row["id"]), "name": body.name}


@router.delete("/org")
def delete_org(confirm: str = "", ctx: OrgContext = Depends(require_role("owner"))):
    # Permanent deletion of the organization and ALL its data (GDPR). Requires the
    # exact org name as confirmation.
    with no_tenant() as conn:
        row = conn.execute("SELECT name FROM organization WHERE id=%s", (ctx.org_id,)).fetchone()
        if not row:
            raise bad_request("organization not found")
        if confirm.strip() != row["name"]:
            raise bad_request("Nhập đúng tên tổ chức để xác nhận xoá.")
        conn.execute("DELETE FROM organization WHERE id=%s", (ctx.org_id,))
    audit.record("org.delete", actor_user_id=ctx.user.id, target=ctx.org_id, detail={"name": row["name"]})
    return {"ok": True}


@router.get("/members")
def list_members(ctx: OrgContext = Depends(require_role("admin"))):
    with tenant_tx(ctx.org_id) as conn:
        rows = conn.execute(
            """SELECT u.email, m.role, m.created_at
               FROM membership m JOIN app_user u ON u.id = m.user_id
               ORDER BY m.created_at"""
        ).fetchall()
    return [{"email": r["email"], "role": r["role"]} for r in rows]


@router.post("/members")
def add_member(body: MemberBody, ctx: OrgContext = Depends(require_role("admin"))):
    if body.role not in ("owner", "admin", "agent", "viewer"):
        raise bad_request("invalid role")
    with no_tenant() as conn:
        u = conn.execute("SELECT id FROM app_user WHERE email=%s", (body.email,)).fetchone()
    if not u:
        raise bad_request("user must sign up first")
    with tenant_tx(ctx.org_id) as conn:
        conn.execute(
            """INSERT INTO membership (organization_id, user_id, role) VALUES (%s,%s,%s)
               ON CONFLICT (organization_id, user_id) DO UPDATE SET role = EXCLUDED.role""",
            (ctx.org_id, str(u["id"]), body.role),
        )
    audit.record("member.add", organization_id=ctx.org_id, actor_user_id=ctx.user.id,
                 target=body.email, detail={"role": body.role})
    from ..providers.email import send_safe
    send_safe(body.email, "Bạn được mời vào workspace OmniShop AI",
              f"<p>Bạn vừa được thêm vào một workspace trên OmniShop AI với vai trò <b>{body.role}</b>. "
              f"Đăng nhập để bắt đầu.</p>")
    return {"ok": True}
