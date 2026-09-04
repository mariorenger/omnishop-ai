"""Authentication: signup (bootstraps org+membership+free subscription), login, me.
AuthProvider boundary (ADR-003) — local for MVP, swappable for Keycloak/OIDC."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..db import admin_tx, no_tenant, tenant_tx
from ..errors import bad_request, unauthorized
from ..security import hash_password, issue_token, verify_password
from ..tenancy import CurrentUser, get_current_user, require_platform_admin

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupBody(BaseModel):
    email: str
    password: str
    full_name: str = ""
    org_name: str = ""


class LoginBody(BaseModel):
    email: str
    password: str


def _my_orgs(user_id: str) -> list[dict]:
    # A user's own memberships span multiple orgs; read via superuser keyed to
    # the authenticated user id (their own data), since membership is RLS-scoped.
    with admin_tx() as conn:
        rows = conn.execute(
            """SELECT o.id, o.name, m.role
               FROM membership m JOIN organization o ON o.id = m.organization_id
               WHERE m.user_id = %s ORDER BY o.created_at""",
            (user_id,),
        ).fetchall()
    return [{"id": str(r["id"]), "name": r["name"], "role": r["role"]} for r in rows]


@router.post("/signup")
def signup(body: SignupBody):
    if len(body.password) < 8:
        raise bad_request("password must be at least 8 characters")
    with no_tenant() as conn:
        dup = conn.execute("SELECT 1 FROM app_user WHERE email=%s", (body.email,)).fetchone()
        if dup:
            raise bad_request("email already registered")
        user = conn.execute(
            "INSERT INTO app_user (email, password_hash, full_name) VALUES (%s,%s,%s) RETURNING id",
            (body.email, hash_password(body.password), body.full_name),
        ).fetchone()
        user_id = str(user["id"])
        org = conn.execute(
            "INSERT INTO organization (name) VALUES (%s) RETURNING id",
            (body.org_name or f"{body.email.split('@')[0]}'s workspace",),
        ).fetchone()
        org_id = str(org["id"])
        conn.execute(
            "INSERT INTO subscription (organization_id, plan_code) VALUES (%s,'free')", (org_id,)
        )
    # membership is RLS-protected — insert within the org's tenant context.
    with tenant_tx(org_id) as conn:
        conn.execute(
            "INSERT INTO membership (organization_id, user_id, role) VALUES (%s,%s,'owner')",
            (org_id, user_id),
        )
    audit.record("signup", organization_id=org_id, actor_user_id=user_id, target=body.email)
    return {"token": issue_token(user_id), "user": {"id": user_id, "email": body.email},
            "orgs": _my_orgs(user_id)}


@router.post("/login")
def login(body: LoginBody):
    from .. import ratelimit
    ratelimit.check(f"login:{body.email.lower()}", limit=10, window_s=300)
    with no_tenant() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM app_user WHERE email=%s", (body.email,)
        ).fetchone()
    if not row or not row["password_hash"] or not verify_password(body.password, row["password_hash"]):
        raise unauthorized("invalid credentials")
    user_id = str(row["id"])
    audit.record("auth.login", actor_user_id=user_id, detail={"via": "password"})
    return {"token": issue_token(user_id), "user": {"id": user_id, "email": body.email},
            "orgs": _my_orgs(user_id)}


@router.post("/logout-all")
def logout_all(user: CurrentUser = Depends(get_current_user)):
    """Revoke every session for this user (tokens issued before now stop working)."""
    with no_tenant() as conn:
        conn.execute("UPDATE app_user SET tokens_valid_after=now() WHERE id=%s", (user.id,))
    audit.record("auth.logout_all", actor_user_id=user.id)
    return {"ok": True}


@router.get("/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return {"user": {"id": user.id, "email": user.email, "is_platform_admin": user.is_platform_admin,
                     "platform_role": user.platform_role},
            "orgs": _my_orgs(user.id)}


# ---- platform staff management (admin grants manager/admin) -----------------

class StaffBody(BaseModel):
    email: str
    platform_role: str          # 'admin' | 'manager' | 'none'


@router.get("/staff")
def list_staff(_: CurrentUser = Depends(require_platform_admin)):
    with no_tenant() as conn:
        rows = conn.execute(
            "SELECT email, platform_role FROM app_user WHERE platform_role IS NOT NULL ORDER BY email"
        ).fetchall()
    return [{"email": r["email"], "platform_role": r["platform_role"]} for r in rows]


@router.put("/staff")
def set_staff(body: StaffBody, admin: CurrentUser = Depends(require_platform_admin)):
    role = body.platform_role if body.platform_role in ("admin", "manager") else None
    with no_tenant() as conn:
        row = conn.execute("SELECT id, platform_role FROM app_user WHERE lower(email)=lower(%s)", (body.email,)).fetchone()
        if not row:
            raise bad_request("người dùng chưa có tài khoản — họ cần đăng nhập/đăng ký trước")
        # never allow removing the LAST platform admin — that would lock everyone out
        if row["platform_role"] == "admin" and role != "admin":
            others = conn.execute(
                "SELECT count(*) AS n FROM app_user WHERE platform_role='admin' AND id<>%s", (row["id"],)
            ).fetchone()["n"]
            if int(others) == 0:
                raise bad_request("Không thể gỡ quyền của quản trị viên cuối cùng. Hãy cấp quyền admin cho người khác trước.")
        conn.execute("UPDATE app_user SET platform_role=%s, is_platform_admin=%s WHERE id=%s",
                     (role, role == "admin", row["id"]))
    audit.record("admin.staff.update", actor_user_id=admin.id, target=body.email,
                 detail={"platform_role": role})
    return {"email": body.email, "platform_role": role}


def bootstrap_admin() -> None:
    """Ensure the deployer's platform admin exists (BOOTSTRAP_ADMIN_EMAIL). Idempotent."""
    from ..config import config
    email = config.BOOTSTRAP_ADMIN_EMAIL.strip()
    if not email:
        return
    with no_tenant() as conn:
        row = conn.execute("SELECT id, platform_role FROM app_user WHERE lower(email)=lower(%s)", (email,)).fetchone()
        if row:
            if row["platform_role"] != "admin":
                conn.execute("UPDATE app_user SET platform_role='admin', is_platform_admin=true WHERE id=%s", (row["id"],))
            return
        pw = config.BOOTSTRAP_ADMIN_PASSWORD or "change-me-now-123"
        conn.execute(
            "INSERT INTO app_user (email, password_hash, full_name, is_platform_admin, platform_role) "
            "VALUES (%s,%s,'Platform Admin',true,'admin')",
            (email, hash_password(pw)),
        )
    print(f"[auth] bootstrapped platform admin: {email}", flush=True)
