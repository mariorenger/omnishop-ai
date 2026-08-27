"""FastAPI application factory: wires modules, static frontend, health, and a
correlation-id for request tracing (observability §12)."""
from __future__ import annotations
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import config
from .errors import new_correlation_id
from .modules import (admin, analytics, auth, billing, bots, branding, channel, conversation,
                      knowledge, oauth_meta, product, rag, settings, tenant, uploads, usage)

app = FastAPI(title="OmniShop AI", version="0.1.0")

# CORS so the standalone web frontend (dev server / other origin) can call the API.
# In production the web service reverse-proxies /api same-origin; this stays permissive
# for local dev and is configurable via CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
          knowledge.router, product.router, conversation.router, admin.router, settings.router,
          analytics.router, rag.router, bots.router, uploads.router, oauth_meta.router,
          branding.router):
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
