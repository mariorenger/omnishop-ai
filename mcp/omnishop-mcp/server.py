"""OmniShop MCP server — exposes the OmniShop API as MCP tools over Streamable HTTP.

It signs in to OmniShop as a normal merchant user (OMNISHOP_EMAIL/PASSWORD) and
works inside one organization (OMNISHOP_ORG_ID, or that user's first org), so
tenant isolation and role checks stay enforced by the OmniShop API itself.

Registered behind IBM ContextForge (see docker-compose.mcp.yml); the gateway
federates these tools with other MCP servers into one virtual server.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

API_URL = os.environ.get("OMNISHOP_API_URL", "http://api:8000").rstrip("/")
EMAIL = os.environ.get("OMNISHOP_EMAIL", "demo@omnishop.local")
PASSWORD = os.environ.get("OMNISHOP_PASSWORD", "demo12345")
ORG_ID = os.environ.get("OMNISHOP_ORG_ID", "")
# Agent replies are pushed out to real customers on real channels — opt in explicitly.
ALLOW_WRITE = os.environ.get("OMNISHOP_MCP_ALLOW_WRITE", "false").lower() in ("1", "true", "yes")
PORT = int(os.environ.get("PORT", "9100"))


class OmniShopClient:
    """Minimal OmniShop API client: lazy login, cached token, one re-login on 401."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(base_url=API_URL, timeout=60)
        self._token = ""
        self._org_id = ORG_ID

    async def _login(self) -> None:
        r = await self._http.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
        if r.status_code != 200:
            raise RuntimeError(f"OmniShop login failed for {EMAIL}: HTTP {r.status_code} {r.text[:200]}")
        data = r.json()
        self._token = data["token"]
        if not self._org_id:
            orgs = data.get("orgs") or []
            if not orgs:
                raise RuntimeError(f"{EMAIL} is not a member of any organization")
            self._org_id = orgs[0]["id"]

    async def request(self, method: str, path: str, **kw: Any) -> Any:
        for attempt in (1, 2):
            if not self._token:
                await self._login()
            headers = {"Authorization": f"Bearer {self._token}", "X-Org-Id": self._org_id}
            r = await self._http.request(method, path, headers=headers, **kw)
            if r.status_code == 401 and attempt == 1:
                self._token = ""
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"OmniShop {method} {path} → HTTP {r.status_code}: {r.text[:300]}")
            return r.json()

    async def default_shop_id(self, shop_id: str | None) -> str:
        if shop_id:
            return shop_id
        shops = await self.request("GET", "/api/shops")
        if not shops:
            raise RuntimeError("this organization has no shop yet")
        return shops[0]["id"]


client = OmniShopClient()
mcp = MCPServer(
    name="omnishop",
    title="OmniShop AI",
    instructions=(
        "Tools for an OmniShop merchant: shops, product catalog, hybrid RAG search over "
        "products + knowledge base, and the customer conversation inbox. "
        "shop_id is optional everywhere — the organization's first shop is used by default."
    ),
    version="1.0.0",
)


@mcp.tool(description="List the shops of the current OmniShop organization.")
async def list_shops() -> list[dict]:
    return await client.request("GET", "/api/shops")


@mcp.tool(description="List products (with variants, price, stock) of a shop, newest first.")
async def list_products(shop_id: str | None = None, limit: int = 20, offset: int = 0) -> dict:
    sid = await client.default_shop_id(shop_id)
    return await client.request("GET", "/api/products",
                                params={"shop_id": sid, "limit": limit, "offset": offset})


@mcp.tool(description=(
    "Hybrid RAG search (pgvector + keyword) over a shop's products and knowledge base. "
    "Set answer=true to also get the shop assistant's generated reply."
))
async def search_catalog(query: str, shop_id: str | None = None, top_k: int = 4,
                         answer: bool = False) -> dict:
    sid = await client.default_shop_id(shop_id)
    return await client.request("POST", "/api/rag/query",
                                json={"shop_id": sid, "query": query, "top_k": top_k, "answer": answer})


@mcp.tool(description="List recent customer conversations of a shop (all channels).")
async def list_conversations(shop_id: str | None = None, limit: int = 20, offset: int = 0) -> dict:
    sid = await client.default_shop_id(shop_id)
    return await client.request("GET", "/api/conversations",
                                params={"shop_id": sid, "limit": limit, "offset": offset})


@mcp.tool(description="Read the newest messages of one conversation (ascending order).")
async def get_conversation_messages(conversation_id: str, limit: int = 50) -> dict:
    return await client.request("GET", f"/api/conversations/{conversation_id}/messages",
                                params={"limit": limit})


if ALLOW_WRITE:
    @mcp.tool(description=(
        "Send a human-agent reply to a customer conversation. The message is delivered to the "
        "customer's real channel (Messenger/Zalo/Telegram/…) and hands the chat over to a human."
    ))
    async def reply_to_conversation(conversation_id: str, text: str) -> dict:
        return await client.request("POST", f"/api/conversations/{conversation_id}/reply",
                                    json={"text": text})


@mcp.prompt(description="Draft an answer to a customer question grounded in the shop's catalog.")
def customer_support_answer(question: str) -> str:
    return (
        "You are a helpful sales assistant for an OmniShop merchant. First call `search_catalog` "
        f"with the customer's question, then answer in the customer's language using only the "
        f"products and knowledge returned. Mention price and stock when relevant.\n\n"
        f"Customer question: {question}"
    )


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy"})


# Stateless + no DNS-rebinding host check: the gateway reaches us by container
# name (Host: omnishop-mcp:9100) and this port is only published on the compose network.
app = mcp.streamable_http_app(
    host="0.0.0.0",
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
