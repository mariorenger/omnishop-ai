"""SePay bank-reconciliation webhook.

SePay watches the merchant's bank account and POSTs every incoming transfer to
our endpoint. We match the transfer to a pending/submitted invoice by the OMNI
code embedded in the transfer content, activate the plan automatically, and log
every transaction so the platform admin can see incoming payments.

Endpoint path is fixed to /webhook/sepay-webhook (no /api prefix) as configured
in the SePay dashboard. Auth: SePay sends `Authorization: Apikey <API_KEY>`.
"""
from __future__ import annotations
import json
import re

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from ..db import admin_tx
from ..providers import registry
from ..providers.payment import invoice_code

router = APIRouter(tags=["sepay"])   # mounted at the app root (no prefix)


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


@router.get("/webhook/sepay-webhook")
def sepay_webhook_health():
    """SePay calls this endpoint with POST; a GET (browser check) returns a hint
    instead of a 405 so it's easy to confirm the URL is reachable."""
    cfg = registry.resolve_sepay()
    return {"ok": True, "method": "POST", "configured": bool(cfg["api_key"]),
            "note": "SePay gửi dữ liệu bằng POST tới URL này."}


@router.post("/webhook/sepay-webhook")
async def sepay_webhook(request: Request, authorization: str = Header(default="")):
    cfg = registry.resolve_sepay()
    # Auth: only enforced when an API key is configured on our side (SePay also
    # supports a no-auth webhook). A configured key must match the "Apikey <key>"
    # header SePay sends; otherwise 401. We never 400 on the test ping itself.
    if cfg["api_key"]:
        provided = authorization.replace("Apikey", "").replace("apikey", "").replace("Bearer", "").strip()
        if provided != cfg["api_key"]:
            return JSONResponse({"success": False, "error": "unauthorized"}, status_code=401)

    # Parse leniently: SePay sends JSON, but a test ping / empty / form body must
    # not break with a 400 — treat anything unparseable as an empty payload.
    raw = await request.body()
    p: dict = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                p = parsed
        except Exception:  # noqa: BLE001
            try:
                from urllib.parse import parse_qs
                p = {k: v[0] for k, v in parse_qs(raw.decode("utf-8", "ignore")).items()}
            except Exception:  # noqa: BLE001
                p = {}

    sepay_id = p.get("id")
    amount = float(p.get("transferAmount") or 0)
    content = p.get("content") or p.get("description") or ""
    transfer_type = (p.get("transferType") or "").lower()
    gateway = p.get("gateway") or ""
    account_no = p.get("accountNumber") or ""
    reference = p.get("referenceCode") or ""

    matched = None
    org_id = None
    with admin_tx() as conn:
        # idempotency: SePay can retry — a transaction we already saw is a no-op.
        if sepay_id is not None and conn.execute(
                "SELECT 1 FROM sepay_transaction WHERE sepay_id=%s", (sepay_id,)).fetchone():
            return {"success": True, "duplicate": True}
        # only incoming money can pay an invoice; match by the OMNI code in the memo
        if transfer_type in ("in", "") and content:
            norm = _norm(content)
            invs = conn.execute(
                "SELECT id, organization_id FROM invoice WHERE status in ('pending','submitted')"
            ).fetchall()
            for r in invs:
                if invoice_code(r["id"]) in norm:
                    matched = str(r["id"]); org_id = str(r["organization_id"]); break
        conn.execute(
            """INSERT INTO sepay_transaction
                 (sepay_id, gateway, account_no, amount, content, reference, transfer_type,
                  matched_invoice, organization_id, status)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (sepay_id, gateway, account_no, amount, content, reference, transfer_type,
             matched, org_id, "activated" if matched else "received"),
        )

    if matched:
        from .billing import _activate_invoice
        _activate_invoice(matched, provider="sepay")

    return {"success": True, "matched": bool(matched)}
