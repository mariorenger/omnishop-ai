"""FastAPI dependencies for authN/authZ and tenant context.

Authorization is enforced server-side (never trust the frontend). Org membership
is checked *through* RLS: we open a tenant transaction for the requested org and
look up the caller's membership — a non-member simply gets no row.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header

from .db import no_tenant, tenant_tx
from .errors import unauthorized, forbidden
from .security import decode_token

# role hierarchy for RBAC checks
_ROLE_RANK = {"viewer": 0, "agent": 1, "admin": 2, "owner": 3}


@dataclass
class CurrentUser:
    id: str
    email: str
    is_platform_admin: bool
    platform_role: Optional[str] = None   # 'admin' | 'manager' | None


@dataclass
class OrgContext:
    org_id: str
    role: str
    user: CurrentUser


def get_current_user(authorization: str = Header(default="")) -> CurrentUser:
    if not authorization.lower().startswith("bearer "):
        raise unauthorized()
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    if not payload:
        raise unauthorized("invalid or expired token")
    user_id = str(payload["sub"])
    with no_tenant() as conn:
        row = conn.execute(
            "SELECT id, email, is_platform_admin, platform_role, extract(epoch from tokens_valid_after) AS tva "
            "FROM app_user WHERE id = %s", (user_id,)
        ).fetchone()
    if not row:
        raise unauthorized()
    if row["tva"] and float(payload.get("iat", 0)) < float(row["tva"]):
        raise unauthorized("session revoked — please sign in again")
    prole = row["platform_role"] or ("admin" if row["is_platform_admin"] else None)
    return CurrentUser(id=str(row["id"]), email=row["email"],
                       is_platform_admin=(prole == "admin"), platform_role=prole)


def get_org_context(
    x_org_id: str = Header(default=""),
    user: CurrentUser = Depends(get_current_user),
) -> OrgContext:
    if not x_org_id:
        raise forbidden("missing X-Org-Id header")
    with tenant_tx(x_org_id) as conn:
        row = conn.execute(
            "SELECT role FROM membership WHERE user_id = %s", (user.id,)
        ).fetchone()
    if not row:
        raise forbidden("not a member of this organization")
    return OrgContext(org_id=x_org_id, role=row["role"], user=user)


def require_role(min_role: str):
    """Dependency factory: require at least `min_role` in the org."""
    def _dep(ctx: OrgContext = Depends(get_org_context)) -> OrgContext:
        if _ROLE_RANK.get(ctx.role, -1) < _ROLE_RANK.get(min_role, 99):
            raise forbidden(f"requires role >= {min_role}")
        return ctx
    return _dep


def require_platform_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Write access to the control plane — platform admins only."""
    if user.platform_role != "admin":
        raise forbidden("platform admin only")
    return user


def require_platform_reader(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Read access to the control plane — platform admin OR manager (read-only)."""
    if user.platform_role not in ("admin", "manager"):
        raise forbidden("platform admin or manager only")
    return user
