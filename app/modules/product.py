"""Products & variants (product-aware AI). Create/list; embeddings computed async.
In production these are synced from the sales channel; MVP supports manual entry."""
from __future__ import annotations
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .. import audit
from ..db import tenant_tx
from ..errors import bad_request
from ..providers.queue import enqueue
from ..tenancy import OrgContext, get_org_context, require_role

router = APIRouter(prefix="/api/products", tags=["product"])


class VariantBody(BaseModel):
    name: str
    sku: str = ""
    price: Optional[float] = None
    stock: int = 0


class ProductBody(BaseModel):
    shop_id: str
    name: str
    description: str = ""
    price: Optional[float] = None
    currency: str = "VND"
    sku: str = ""
    attributes: dict = {}
    variants: List[VariantBody] = []
    bot_id: Optional[str] = None


def _assert_shop(conn, shop_id: str):
    if not conn.execute("SELECT 1 FROM shop WHERE id=%s", (shop_id,)).fetchone():
        raise bad_request("shop not found in this organization")


@router.get("")
def list_products(shop_id: str, ctx: OrgContext = Depends(get_org_context)):
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, shop_id)
        rows = conn.execute(
            "SELECT id, name, description, price, currency, sku, attributes FROM product WHERE shop_id=%s ORDER BY created_at DESC",
            (shop_id,),
        ).fetchall()
        out = []
        for r in rows:
            variants = conn.execute(
                "SELECT name, sku, price, stock FROM product_variant WHERE product_id=%s ORDER BY name",
                (r["id"],),
            ).fetchall()
            out.append({
                "id": str(r["id"]), "name": r["name"], "description": r["description"],
                "price": float(r["price"]) if r["price"] is not None else None,
                "currency": r["currency"], "sku": r["sku"], "attributes": r["attributes"],
                "variants": [{"name": v["name"], "sku": v["sku"],
                              "price": float(v["price"]) if v["price"] is not None else None,
                              "stock": int(v["stock"])} for v in variants],
            })
    return out


@router.post("")
def create_product(body: ProductBody, ctx: OrgContext = Depends(require_role("admin"))):
    if not body.name.strip():
        raise bad_request("name is required")
    import json
    with tenant_tx(ctx.org_id) as conn:
        _assert_shop(conn, body.shop_id)
        bot_id = body.bot_id
        if bot_id and not conn.execute("SELECT 1 FROM bot WHERE id=%s AND shop_id=%s", (bot_id, body.shop_id)).fetchone():
            bot_id = None
        row = conn.execute(
            """INSERT INTO product (organization_id, shop_id, name, description, price, currency, sku, attributes, bot_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (ctx.org_id, body.shop_id, body.name, body.description, body.price,
             body.currency, body.sku, json.dumps(body.attributes), bot_id),
        ).fetchone()
        pid = str(row["id"])
        for v in body.variants:
            conn.execute(
                """INSERT INTO product_variant (organization_id, product_id, name, sku, price, stock)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (ctx.org_id, pid, v.name, v.sku, v.price, v.stock),
            )
    enqueue("embed_product", {"product_id": pid}, organization_id=ctx.org_id)
    audit.record("product.create", organization_id=ctx.org_id, actor_user_id=ctx.user.id, target=pid)
    return {"id": pid, "name": body.name, "variants": len(body.variants)}
