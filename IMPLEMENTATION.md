# Implementation Notes — Milestone 1 (runnable MVP)

This is the first runnable slice of OmniShop AI: the **core loop** end-to-end —
sign up → create shop → add products/knowledge → get a website chat widget →
customer asks → **product-aware RAG** answer → metering + quota → human handoff.
It runs **with no API keys** (a retrieval-grounded stub LLM + a local embedder);
add `ANTHROPIC_API_KEY` for real Claude answers.

## Run it

```bash
cd omnishop-ai
docker compose up -d --build          # postgres+pgvector, valkey, api, worker
docker compose exec api python -m scripts.seed   # demo tenant + products + knowledge
# open http://localhost:8000
```

Seed prints logins and a widget URL:

- Merchant dashboard: `demo@omnishop.local` / `demo12345`
- Platform admin: `admin@omnishop.local` / `admin12345`
- Widget demo: `http://localhost:8000/widget.html?key=<printed>`

Try asking the widget: *"Áo thun size M màu đen còn không?"* or
*"Chính sách đổi trả thế nào?"*.

Turn on real AI: set `ANTHROPIC_API_KEY` in `.env`, `docker compose up -d`.

### Verify (tenant isolation + async path)

```bash
docker compose exec api python -m scripts.test_isolation   # ADR-008 RLS checks
```

## What works (verified end-to-end against real Postgres+pgvector+RLS+Redis)

- **Auth** (signup/login/JWT, PBKDF2) and **RBAC** (owner/admin/agent/viewer).
- **Multi-tenancy with 3-layer isolation** (ADR-008): app scoping + **Postgres
  RLS** + vector metadata scoping. `test_isolation` proves cross-tenant reads and
  writes are blocked; the API rejects a spoofed `X-Org-Id` with 403.
- **Plans / entitlements / quota** (ADR-004): free/starter/growth seeded; AI
  quota enforced server-side; plan change endpoint.
- **Usage metering & cost** (ADR-007): every AI reply + embedding batch recorded
  (model, tokens, retrieval count, latency, estimated cost) → per-tenant COGS in
  the dashboard and admin overview.
- **Knowledge**: paste text → chunk → **async embed via Valkey queue + worker** →
  retrievable. (Verified: upload → job → worker → document ready → AI cites it.)
- **Product-aware AI**: products + variants (stock/price); a size/stock question
  routes to product data, not just the vector store (architecture §6).
- **RAG engine**: hybrid-ready retrieval over pgvector, tenant-filtered, behind a
  swappable interface (ADR-002).
- **Conversation orchestrator** (architecture §7): intent classify → retrieve →
  LLM (quota-checked) → response/handoff policy → persist → meter.
- **Channels**: website widget live (public key, embeddable iframe) behind the
  `ChannelProvider` abstraction; Fake provider for tests; Meta/TikTok/Shopee are
  future providers (guarded off with a clear message).
- **Unified inbox**: agent sees conversations, replies (human handoff), closes;
  the widget polls and shows agent replies.
- **Admin control plane**: cross-tenant tenants list + overview (cost/usage).
- **Audit log** on privileged actions; **encrypted-at-rest** helper for channel
  credentials (R-17).

## How it maps to the milestones

| Milestone | Status here |
|---|---|
| 1 Foundation (repo, compose, PG, cache, auth, tenant, RBAC, audit) | ✅ |
| 2 Billing (plans, entitlements, usage, quota) | ✅ (payments deferred to a `BillingProvider` adapter) |
| 3 Channel core (`ChannelProvider`, connection, fake provider) | ✅ |
| 4 First real channel | ⏳ website shipped; **Meta next** (needs OAuth + app review) |
| 5 Knowledge (upload→chunk→embed→retrieve) | ✅ (PDF/DOCX parsing = OSS parser later) |
| 6 AI chatbot (conversation, RAG, LLM, handoff) | ✅ (guardrails/eval to deepen) |
| 7 Product AI (sync, product-aware retrieval) | ✅ manual entry; channel sync later |
| 8 Unified inbox | ✅ minimal own inbox; Chatwoot integration is ADR-005 |
| 9 Operations (admin, health, usage, cost, audit) | ◑ admin+usage+audit done; health states scaffolded |
| 10 Scale | deferred (as designed — no premature optimization) |

## Deliberate deviations from the ADRs (documented, not accidental)

- **Language/queue:** the ADRs sketched TypeScript/BullMQ; the MVP is **Python +
  FastAPI** with a **Valkey-backed queue** behind `QueueProvider`. Same interface
  principle; BullMQ/Temporal remain the documented scale target (ADR-006). Chosen
  for fastest reliable first-run.
- **Auth:** ADR-003 targets **Keycloak**; the MVP uses a thin local
  `AuthProvider` (PBKDF2 + HS256 JWT, zero native deps) behind the same boundary,
  swappable for Keycloak/OIDC without touching callers.
- **Object storage / secrets:** not yet wired (knowledge is pasted text for now);
  interfaces and ADRs stand for when file upload lands.
- **LLM default model:** `claude-opus-5` (configurable via `LLM_MODEL`). For a
  high-volume support bot, route a cheaper model in production per ADR-007
  (`LLM_MODEL=claude-haiku-4-5`), and effort is set `low` for interactive latency.

## Known limitations (good first feedback targets)

- Website is the only live channel (Meta/TikTok/Shopee gated off — see the
  platform matrix; they need OAuth + app-review/partner access).
- Knowledge ingestion is pasted text (no PDF/DOCX/OCR yet — that's an OSS parser).
- Local embeddings are lexical-ish (great offline; use OpenAI/`EMBEDDING_PROVIDER`
  or a hosted embedder for stronger semantics).
- Guardrails/prompt-injection defenses are minimal; handoff is confidence-lite.
- No payment processing yet (entitlements enforced; `BillingProvider` is the seam).

## Layout

```
app/
  config.py db.py security.py tenancy.py audit.py errors.py main.py worker.py
  providers/  llm.py embeddings.py vectorstore.py queue.py  channels/{base,website,fake}.py
  modules/    auth tenant billing usage knowledge product channel conversation admin orchestrator
  static/     index.html (dashboard)  widget.html (customer chat)
db/001_init.sql          # schema + RLS + pgvector + seed plans
scripts/seed.py          # demo tenant
scripts/test_isolation.py# ADR-008 verification (runnable)
```
