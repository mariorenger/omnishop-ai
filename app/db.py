"""Database access with tenant-scoped Row-Level Security (ADR-008).

Two connection pools:
  - app pool  (omni_app, non-superuser) → RLS is enforced. Use for all
    tenant-scoped work; wrap in `tenant_tx(org_id)` which sets app.current_org.
  - admin pool (omni, superuser) → RLS bypassed. Use only for platform-admin
    cross-tenant reads and non-tenant tables when convenient.

`no_tenant()` uses the app pool without an org context; it is only safe for
non-RLS tables (app_user, organization, plan, subscription, job, audit_log).
Any query against an RLS table without an org context returns zero rows.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import config

_app_pool: Optional[ConnectionPool] = None
_admin_pool: Optional[ConnectionPool] = None


def _pool(dsn: str) -> ConnectionPool:
    return ConnectionPool(dsn, min_size=1, max_size=10, kwargs={"row_factory": dict_row}, open=True)


def app_pool() -> ConnectionPool:
    global _app_pool
    if _app_pool is None:
        _app_pool = _pool(config.PG_DSN)
    return _app_pool


def admin_pool() -> ConnectionPool:
    global _admin_pool
    if _admin_pool is None:
        _admin_pool = _pool(config.PG_DSN_ADMIN)
    return _admin_pool


@contextmanager
def tenant_tx(org_id: str) -> Iterator[psycopg.Connection]:
    """A transaction scoped to one organization; RLS restricts every row."""
    with app_pool().connection() as conn:
        with conn.transaction():
            # transaction-local; reset automatically at commit/rollback.
            conn.execute("SELECT set_config('app.current_org', %s, true)", (str(org_id),))
            yield conn


@contextmanager
def no_tenant() -> Iterator[psycopg.Connection]:
    """App-pool connection with no org context (non-RLS tables only)."""
    with app_pool().connection() as conn:
        with conn.transaction():
            yield conn


@contextmanager
def admin_tx() -> Iterator[psycopg.Connection]:
    """Superuser connection (RLS bypassed). Platform-admin use only."""
    with admin_pool().connection() as conn:
        with conn.transaction():
            yield conn


def wait_ready(timeout_s: int = 30) -> None:
    """Block until the DB accepts a trivial query (used at startup)."""
    import time
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            with admin_pool().connection() as conn:
                conn.execute("SELECT 1")
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1)
    raise RuntimeError(f"database not ready: {last}")
