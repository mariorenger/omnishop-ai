# OmniShop MCP qua IBM ContextForge

Chạy **full luồng MCP** bằng Docker Compose: client MCP (Claude Desktop/Code, Cursor,
MCP Inspector…) → **IBM ContextForge MCP Gateway** → các MCP server phía sau → OmniShop API.

```
MCP client ──(Streamable HTTP + Bearer JWT)──► mcpgateway :4444
                                                 │  virtual server "OmniShop Assistant"
                                                 ├──► omnishop-mcp :9100 ──► api :8000 (RLS theo tenant)
                                                 └──► fast-time    :9080   (server demo của IBM)
                     cf-postgres (catalog của gateway) · cf-redis (cache/session)
```

| Service | Vai trò |
|---|---|
| `cf-postgres`, `cf-redis` | DB + cache riêng của gateway (tách khỏi `db`/`cache` của OmniShop) |
| `cf-migration` | Chạy một lần: tạo schema + tài khoản admin của gateway |
| `mcpgateway` | ContextForge: Admin UI, REST API, JWT auth, federation các MCP server |
| `omnishop-mcp` | MCP server của OmniShop ([`omnishop-mcp/server.py`](omnishop-mcp/server.py)) |
| `fast-time` | MCP server demo của IBM, để thấy gateway gộp nhiều server thành một |
| `omnishop-seed` | Chạy một lần: tạo tenant demo (`demo@omnishop.local`) để MCP server đăng nhập |
| `mcp-register` | Chạy một lần: đăng ký server → tạo virtual server → **smoke test end-to-end** → ghi token ra `mcp/out/` |

## Chạy

```bash
docker compose -f docker-compose.yml -f docker-compose.mcp.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.mcp.yml logs -f mcp-register
```

Khi xong, log của `mcp-register` sẽ in:

```
[register] smoke: tools/list → ['omnishop-list-shops', 'omnishop-list-products', 'omnishop-search-catalog',
                                'omnishop-list-conversations', 'omnishop-get-conversation-messages',
                                'fast-time-get-system-time']
[register] smoke: ✓ omnishop-search-catalog → {"products": [{"name": "Áo thun cotton basic", ...
[register] Admin UI      : http://localhost:4444/admin  (admin@example.com)
[register] MCP endpoint  : http://localhost:4444/servers/0a15b6c4e1d34f2c9e7a6b3d5f8c2e10/mcp
[register] Smoke test    : PASSED
```

- **Admin UI**: http://localhost:4444/admin — mặc định `admin@example.com` / `changeme-Admin123!`
- **Token cho client**: `mcp/out/gateway-token.txt` (hết hạn sau 7 ngày; chạy lại `mcp-register` để lấy token mới)
- **Cấu hình client dựng sẵn**: `mcp/out/mcp-client.json`

Chạy lại bước đăng ký (idempotent — dùng lại gateway cũ, dựng lại virtual server):

```bash
docker compose -f docker-compose.yml -f docker-compose.mcp.yml run --rm mcp-register
```

## Kết nối client

**Claude Code**

```bash
claude mcp add --transport http omnishop \
  http://localhost:4444/servers/0a15b6c4e1d34f2c9e7a6b3d5f8c2e10/mcp \
  --header "Authorization: Bearer $(cat mcp/out/gateway-token.txt)"
```

**Claude Desktop / Cursor / client đọc JSON**: chép nội dung `mcp/out/mcp-client.json`
vào file cấu hình MCP của client.

**Gọi tay bằng curl**

```bash
TOKEN=$(cat mcp/out/gateway-token.txt)
curl -s http://localhost:4444/servers/0a15b6c4e1d34f2c9e7a6b3d5f8c2e10/mcp \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
```

## Tool của OmniShop

Gateway thêm tiền tố theo tên server (`omnishop-…`). `shop_id` luôn tuỳ chọn — mặc định là shop đầu tiên.

| Tool | API gọi phía sau |
|---|---|
| `list_shops` | `GET /api/shops` |
| `list_products(shop_id?, limit, offset)` | `GET /api/products` |
| `search_catalog(query, shop_id?, top_k, answer)` | `POST /api/rag/query` — RAG hybrid; `answer=true` để lấy câu trả lời của trợ lý |
| `list_conversations(shop_id?, limit, offset)` | `GET /api/conversations` |
| `get_conversation_messages(conversation_id, limit)` | `GET /api/conversations/{id}/messages` |
| `reply_to_conversation(conversation_id, text)` | `POST /api/conversations/{id}/reply` — **chỉ bật khi** `OMNISHOP_MCP_ALLOW_WRITE=true`, vì tin nhắn được gửi thật tới khách |

Prompt: `customer_support_answer(question)`.

MCP server đăng nhập OmniShop như **một thành viên bình thường** của tổ chức
(`OMNISHOP_MCP_EMAIL/PASSWORD`, `OMNISHOP_MCP_ORG_ID`), nên cô lập tenant và phân quyền
theo vai trò vẫn do OmniShop API kiểm soát.

## Thêm MCP server khác

Thêm service vào `docker-compose.mcp.yml`, khai báo nó trong [`servers.json`](servers.json)
(`name`, `url`, `transport`, `health`), rồi chạy lại `mcp-register`. Mọi tool của các server
trong `servers.json` được gộp vào cùng virtual server.

## Cấu hình & production

Mọi biến đều có giá trị dev mặc định — xem [`.env.example`](.env.example). Trước khi dùng thật:

- Đặt `CF_JWT_SECRET_KEY`, `CF_AUTH_ENCRYPTION_SECRET`, `CF_POSTGRES_PASSWORD`, `CF_ADMIN_PASSWORD` (`openssl rand -hex 32`).
- Ghim phiên bản: `CONTEXTFORGE_VERSION=1.0.10` (hoặc bản bạn đã kiểm thử) thay cho `latest`.
- Đặt gateway sau HTTPS, `CF_PUBLIC_URL=https://…`, `CF_ENVIRONMENT=production`, `CF_SECURE_COOKIES=true`, `CF_FORCE_PASSWORD_CHANGE=true`.
- Bỏ `omnishop-seed` và trỏ `OMNISHOP_MCP_EMAIL/PASSWORD` tới tài khoản merchant thật (vai trò thấp nhất đủ dùng).
- Token trong `mcp/out/` là token **admin** của gateway, chỉ để thử; cấp cho người dùng token riêng
  có phạm vi hẹp trong Admin UI (API Tokens).

Dừng / xoá dữ liệu gateway:

```bash
docker compose -f docker-compose.yml -f docker-compose.mcp.yml down        # dừng
docker compose -f docker-compose.yml -f docker-compose.mcp.yml down -v     # xoá cả volume (DB OmniShop + gateway)
```
