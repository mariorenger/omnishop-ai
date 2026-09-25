"""Register MCP servers with IBM ContextForge and publish one virtual server.

Runs once inside the ContextForge image (docker-compose.mcp.yml → mcp-register):
  1. wait for the gateway to be healthy and accept an admin JWT
  2. register (or reuse) every upstream MCP server in servers.json as a gateway
  3. refresh + collect their tools / resources / prompts
  4. (re)create the virtual server that bundles them
  5. smoke-test the whole path: client → gateway → upstream MCP server → OmniShop API
  6. write the bearer token + client config to /out for MCP clients

Idempotent: safe to re-run (`docker compose ... run --rm mcp-register`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

GATEWAY = os.environ.get("GATEWAY_URL", "http://mcpgateway:4444").rstrip("/")
PUBLIC_GATEWAY = os.environ.get("PUBLIC_GATEWAY_URL", "http://localhost:4444").rstrip("/")
ADMIN_EMAIL = os.environ.get("PLATFORM_ADMIN_EMAIL", "admin@example.com")
JWT_SECRET = os.environ["JWT_SECRET_KEY"]
TOKEN_EXP_MIN = os.environ.get("TOKEN_EXPIRY_MINUTES", "10080")  # 7 days
CONFIG = os.environ.get("SERVERS_CONFIG", "/config/servers.json")
OUT = os.environ.get("OUT_DIR", "/out")


def log(msg: str) -> None:
    print(f"[register] {msg}", flush=True)


def make_token() -> str:
    return subprocess.check_output(
        [sys.executable, "-m", "mcpgateway.utils.create_jwt_token",
         "--username", ADMIN_EMAIL, "--admin", "--exp", TOKEN_EXP_MIN,
         "--secret", JWT_SECRET, "--algo", "HS256"],
        text=True, stderr=subprocess.DEVNULL,
    ).strip()


TOKEN = ""


def api(method: str, path: str, data=None, headers=None, raw=False):
    req = urllib.request.Request(f"{GATEWAY}{path}", method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    if data is not None:
        req.data = json.dumps(data).encode()
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode()
        return (body, dict(r.headers)) if raw else (json.loads(body) if body else None)


def retry(fn, what: str, tries: int = 60, delay: float = 2, ok_codes=(401, 404, 409, 502, 503)):
    for i in range(1, tries + 1):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code not in ok_codes or i == tries:
                detail = e.read().decode(errors="replace")[:300]
                raise RuntimeError(f"{what}: HTTP {e.code} {detail}") from e
            log(f"{what}: HTTP {e.code}, retry {i}/{tries}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if i == tries:
                raise
            log(f"{what}: {e}, retry {i}/{tries}")
        time.sleep(delay)


def items(resp):
    """List endpoints return a list, or {'items': [...]} when paginated."""
    if isinstance(resp, dict):
        return resp.get("items") or resp.get("data") or []
    return resp or []


def gw_of(obj) -> str | None:
    return obj.get("gatewayId") or obj.get("gateway_id")


def wait_http(url: str, what: str, tries: int = 90) -> None:
    for i in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    log(f"{what} is up")
                    return
        except Exception:
            pass
        if i % 5 == 0:
            log(f"waiting for {what} ({i}/{tries})")
        time.sleep(2)
    raise RuntimeError(f"{what} did not become healthy at {url}")


def register_gateway(spec: dict) -> str:
    existing = [g for g in items(retry(lambda: api("GET", "/gateways"), "list gateways"))
                if g.get("name") == spec["name"]]
    if existing:
        gid = existing[0]["id"]
        log(f"gateway '{spec['name']}' already registered ({gid})")
    else:
        body = {"name": spec["name"], "url": spec["url"],
                "transport": spec.get("transport", "STREAMABLEHTTP"),
                "description": spec.get("description", "")}
        gid = retry(lambda: api("POST", "/gateways", body), f"register {spec['name']}")["id"]
        log(f"registered gateway '{spec['name']}' → {spec['url']} ({gid})")
    try:
        retry(lambda: api("POST", f"/gateways/{gid}/tools/refresh?include_resources=true&include_prompts=true"),
              f"refresh {spec['name']}", tries=10)
    except Exception as e:  # older gateways sync on registration; refresh is best-effort
        log(f"refresh {spec['name']} skipped: {e}")
    return gid


def collect(kind: str, gateway_ids: set[str]) -> list[dict]:
    return [o for o in items(api("GET", f"/{kind}")) if gw_of(o) in gateway_ids]


def mcp_call(server_id: str, method: str, params: dict | None, session: str | None, rid: int):
    """One JSON-RPC call on the virtual server's Streamable HTTP endpoint."""
    headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
    if session:
        headers["Mcp-Session-Id"] = session
    msg = {"jsonrpc": "2.0", "method": method}
    if rid:
        msg["id"] = rid
    if params is not None:
        msg["params"] = params
    body, hdrs = api("POST", f"/servers/{server_id}/mcp", msg, headers=headers, raw=True)
    sid = next((v for k, v in hdrs.items() if k.lower() == "mcp-session-id"), session)
    if not rid or not body.strip():
        return None, sid
    if body.lstrip().startswith("{"):
        return json.loads(body), sid
    for line in body.splitlines():  # SSE: take the JSON-RPC response for our id
        if line.startswith("data:"):
            m = json.loads(line[5:].strip())
            if m.get("id") == rid:
                return m, sid
    raise RuntimeError(f"no JSON-RPC response for {method}: {body[:300]}")


def smoke_test(server_id: str, calls: list[dict]) -> bool:
    init, sid = mcp_call(server_id, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "omnishop-register", "version": "1"}}, None, 1)
    log(f"smoke: initialize → {init['result']['serverInfo']}")
    mcp_call(server_id, "notifications/initialized", None, sid, 0)
    tools, _ = mcp_call(server_id, "tools/list", {}, sid, 2)
    names = [t["name"] for t in tools["result"]["tools"]]
    log(f"smoke: tools/list → {names}")
    ok = True
    for n, call in enumerate(calls, start=3):
        # the gateway prefixes tool names with the gateway slug (e.g. omnishop-list-shops)
        suffix = call["tool"].replace("_", "-")
        match = next((t for t in names if t == call["tool"] or t.endswith(suffix)
                      or t.endswith(call["tool"])), None)
        if not match:
            log(f"smoke: ✗ tool {call['tool']} not exposed by the virtual server")
            ok = False
            continue
        res, _ = mcp_call(server_id, "tools/call", {"name": match, "arguments": call.get("arguments", {})}, sid, n)
        result = res.get("result") or {}
        text = " ".join(c.get("text", "") for c in result.get("content", []))[:300]
        if res.get("error") or result.get("isError"):
            log(f"smoke: ✗ {match} → {res.get('error') or text}")
            ok = False
        else:
            log(f"smoke: ✓ {match} → {text}")
    return ok


def main() -> int:
    global TOKEN
    cfg = json.load(open(CONFIG))
    wait_http(f"{GATEWAY}/health", "ContextForge gateway")
    for spec in cfg["gateways"]:
        if spec.get("health"):
            wait_http(spec["health"], spec["name"])

    TOKEN = make_token()
    retry(lambda: api("GET", "/gateways"), "authenticated readiness")
    log(f"authenticated as {ADMIN_EMAIL}")

    gids = {register_gateway(spec) for spec in cfg["gateways"]}

    tools: list[dict] = []
    for i in range(60):
        tools = collect("tools", gids)
        if {gw_of(t) for t in tools} >= gids:
            break
        time.sleep(1)
    else:
        missing = gids - {gw_of(t) for t in tools}
        log(f"warning: no tools synced yet from gateways {missing}")
    resources, prompts = collect("resources", gids), collect("prompts", gids)
    log(f"collected {len(tools)} tools, {len(resources)} resources, {len(prompts)} prompts")

    vs = cfg["virtual_server"]
    try:
        api("DELETE", f"/servers/{vs['id']}")
        log(f"replaced existing virtual server {vs['id']}")
    except urllib.error.HTTPError:
        pass
    retry(lambda: api("POST", "/servers", {"server": {
        "id": vs["id"], "name": vs["name"], "description": vs.get("description", ""),
        "associated_tools": [t["id"] for t in tools],
        "associated_resources": [r["id"] for r in resources],
        "associated_prompts": [p["id"] for p in prompts],
    }}), "create virtual server")
    log(f"virtual server '{vs['name']}' ready")

    smoke_ok = smoke_test(vs["id"], cfg.get("smoke_test", []))

    url = f"{PUBLIC_GATEWAY}/servers/{vs['id']}/mcp"
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/gateway-token.txt", "w") as f:
        f.write(TOKEN + "\n")
    client_cfg = {"mcpServers": {"omnishop": {
        "type": "http", "url": url, "headers": {"Authorization": f"Bearer {TOKEN}"}}}}
    with open(f"{OUT}/mcp-client.json", "w") as f:
        json.dump(client_cfg, f, indent=2)

    log("=" * 60)
    log(f"Admin UI      : {PUBLIC_GATEWAY}/admin  ({ADMIN_EMAIL})")
    log(f"MCP endpoint  : {url}")
    log(f"Bearer token  : {OUT}/gateway-token.txt (valid {TOKEN_EXP_MIN} min)")
    log(f"Client config : {OUT}/mcp-client.json")
    log(f"Smoke test    : {'PASSED' if smoke_ok else 'FAILED (see above)'}")
    log("=" * 60)
    return 0 if smoke_ok else 1


if __name__ == "__main__":
    sys.exit(main())
