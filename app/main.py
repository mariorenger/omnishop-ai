"""FastAPI application factory: wires modules, static frontend, health, and a
correlation-id for request tracing (observability §12)."""
from __future__ import annotations
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .errors import new_correlation_id
from .modules import (admin, auth, billing, channel, conversation, knowledge,
                      product, settings, tenant, usage)

app = FastAPI(title="OmniShop AI", version="0.1.0")


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    cid = request.headers.get("x-correlation-id") or new_correlation_id()
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["x-correlation-id"] = cid
    return response


@app.get("/api/health")
def health():
    return {"ok": True, "service": "omnishop-ai"}


for r in (auth.router, tenant.router, billing.router, usage.router, channel.router,
          knowledge.router, product.router, conversation.router, admin.router, settings.router):
    app.include_router(r)


# Static frontend (dashboard + widget). Mounted last so /api/* wins.
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")


@app.on_event("startup")
def _startup():
    # Best-effort DB readiness wait (does not crash under --reload if DB is slow).
    try:
        from .db import run_migrations, wait_ready
        wait_ready(30)
        run_migrations()
    except Exception as e:  # noqa: BLE001
        print(f"[api] startup: database not ready / migration deferred: {e}", flush=True)
