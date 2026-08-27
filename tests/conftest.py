"""Test fixtures. Unit tests need no services; API tests use a FastAPI TestClient
against a real Postgres + Redis (skipped automatically if the DB is unreachable)."""
from __future__ import annotations
import os
import sys
import uuid

import pytest

# Sensible local defaults so the suite runs against the dev stack out of the box.
os.environ.setdefault("PG_DSN", "postgresql://omni_app:omni_app@localhost:5432/omnishop")
os.environ.setdefault("PG_DSN_ADMIN", "postgresql://omni:omni@localhost:5432/omnishop")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("APP_SECRET", "test-secret-please-change-32bytes-minimum")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _db_up() -> bool:
    try:
        import psycopg
        psycopg.connect(os.environ["PG_DSN_ADMIN"], connect_timeout=2).close()
        return True
    except Exception:  # noqa: BLE001
        return False


DB_UP = _db_up()
requires_db = pytest.mark.skipif(not DB_UP, reason="Postgres not reachable")


@pytest.fixture(scope="session")
def app():
    from app.main import app as fastapi_app
    if DB_UP:
        try:
            from app.db import run_migrations, wait_ready
            wait_ready(15)
            run_migrations()
        except Exception:  # noqa: BLE001
            pass
    return fastapi_app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture()
def tenant(client):
    """A fresh org + owner + shop, returning ready-to-use auth headers."""
    email = f"t_{uuid.uuid4().hex[:10]}@test.local"
    r = client.post("/api/auth/signup", json={"email": email, "password": "pw12345678", "org_name": "Test Co"})
    assert r.status_code == 200, r.text
    d = r.json()
    org_id = d["orgs"][0]["id"]
    headers = {"Authorization": f"Bearer {d['token']}", "X-Org-Id": org_id}
    s = client.post("/api/shops", json={"name": "Test Shop"}, headers=headers)
    assert s.status_code == 200, s.text
    return {"headers": headers, "org_id": org_id, "shop_id": s.json()["id"], "email": email}


def drain_jobs(max_jobs: int = 50):
    """Process queued worker jobs synchronously (embedding / ingestion)."""
    from app.providers.queue import pop
    from app.worker import process
    n = 0
    for _ in range(max_jobs):
        job = pop(timeout=1)
        if not job:
            break
        process(job)
        n += 1
    return n
