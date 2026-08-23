"""Authentication: signup (bootstraps org+membership+free subscription), login, me.
AuthProvider boundary (ADR-003) — local for MVP, swappable for Keycloak/OIDC."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..db import admin_tx, no_tenant, tenant_tx
from ..errors import bad_request, unauthorized
from ..security import hash_password, issue_token, verify_password
from ..tenancy import CurrentUser, get_current_user

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
    with no_tenant() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM app_user WHERE email=%s", (body.email,)
        ).fetchone()
    if not row or not verify_password(body.password, row["password_hash"]):
        raise unauthorized("invalid credentials")
    user_id = str(row["id"])
    return {"token": issue_token(user_id), "user": {"id": user_id, "email": body.email},
            "orgs": _my_orgs(user_id)}


@router.get("/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return {"user": {"id": user.id, "email": user.email, "is_platform_admin": user.is_platform_admin},
            "orgs": _my_orgs(user.id)}
