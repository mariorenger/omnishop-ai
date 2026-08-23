# Architecture Overview

This document defines the target architecture for OmniShop AI: the control
plane / data plane split, the multi-tenancy and isolation model, the channel
adapter abstraction, the RAG engine, the conversation orchestrator, and the
MVP → 5-year scaling roadmap.

Companion: [`diagrams.md`](diagrams.md) collects the diagrams referenced here.

---

## 1. First principles

1. **Modular monolith first.** One deployable API + a worker process, organized
   into strict modules. No microservices in the MVP.
2. **Control plane ≠ data plane.** Business/tenant/billing logic is separated
   from message/AI processing so they can scale and fail independently.
3. **Everything replaceable is behind a provider interface.** `LLMProvider`,
   `EmbeddingProvider`, `VectorStore`, `ObjectStorage`, `BillingProvider`,
   `PaymentProvider`, `QueueProvider`, `ChannelProvider`, `SupportProvider`,
   `EmailProvider`.
4. **Every resource is tenant-owned.** No query touches tenant data without an
   `organization_id` (and usually `shop_id`) filter enforced by an authorization
   layer — never by the frontend.
5. **The LLM never touches the database directly.** It calls a small set of
   audited, tenant-scoped tools. This is the hard security boundary.

---

## 2. Control plane vs data plane

```
                          CONTROL PLANE
        (who / what / plan / permission / quota / config / billing)
                                │
        ┌───────────────┬───────┴────────┬────────────────┐
     Tenants          Billing        Configuration      Admin
     Users            Plans          Integrations       Support
     Shops            Usage          AI settings        Audit
     Memberships      Quotas         Permissions        Health
                                │
                                ▼
                           DATA PLANE
        (messages / AI / RAG / products / orders / webhooks)
                                │
        ┌───────────────┬───────┴────────┬────────────────┐
     Webhook intake   RAG engine      LLM / agents      Conversation
     Message I/O      Retrieval       Tools             Product/Order svc
     Product sync     Embeddings      Response policy    Handoff
```

- **Control plane** answers: *who is this, what plan, what are they allowed to
  do, are they within quota, how is billing, what is configured.* Consistency and
  correctness matter more than latency.
- **Data plane** answers: *a message arrived — classify it, retrieve context,
  call the LLM within policy, send a reply, meter the cost.* Throughput and
  latency matter; it reads config/entitlements from the control plane but does
  not own them.

They share a database in the MVP (separate schemas/modules) and can be split into
separate services later (§8) without rewriting business logic.

---

## 3. Multi-tenancy model

```
User ─< Membership >─ Organization ─< Shop ─< Channel ─< Chatbot
                                         │        │
                                         │        └─ Conversation ─< Message
                                         └─ KnowledgeBase ─< Document ─< Chunk
                                         └─ Product ─< Variant
```

- **Organization** is the tenant/billing boundary. A user belongs to one or more
  organizations via **Membership** (role-scoped).
- **Shop** is a merchant storefront within an org; **Channel** is a connected
  platform account (a Facebook Page, an IG account, a Shopee shop…).
- Every data-plane row carries `organization_id` (+ `shop_id` / `channel_id` /
  `knowledge_base_id` where relevant).

**Isolation is enforced at three levels** (detail in
[ADR-008](../decisions/ADR-008-multi-tenancy-isolation.md)):

1. **Logical** — mandatory `organization_id` filter on every query, via a shared
   data-access layer that refuses unscoped tenant queries.
2. **Database** — PostgreSQL **Row-Level Security** as a defense-in-depth backstop
   (session `SET app.current_org`), so a missed filter cannot leak across tenants.
3. **Vector/search** — every embedding and search doc is tagged with
   `organization_id` / `shop_id` / `knowledge_base_id`; retrieval **always**
   applies the metadata filter. Tenant A can never retrieve Tenant B's data.

---

## 4. Channel adapter architecture

All platform-specific logic lives behind one interface. Facebook logic must never
leak into the RAG, conversation, or LLM services.

```ts
interface ChannelProvider {
  connect(): Promise<Connection>;
  disconnect(): Promise<void>;
  refreshCredentials(): Promise<Credentials>;
  getAccounts(): Promise<Account[]>;
  getShops(): Promise<Shop[]>;
  getProducts(): Promise<Product[]>;
  getOrders(): Promise<Order[]>;
  sendMessage(to: Recipient, msg: OutboundMessage): Promise<void>;
  registerWebhook(): Promise<void>;
  unregisterWebhook(): Promise<void>;
  verifyWebhook(req: Request): boolean;              // signature check
  normalizeWebhookEvent(raw: unknown): ChannelEvent; // → canonical event
}
```

Implementations: `MetaMessengerProvider`, `InstagramProvider`,
`TikTokShopProvider`, `ShopeeProvider`, plus a `FakeProvider` for tests
(Milestone 3). The rest of the system consumes only the **canonical**
`ChannelEvent` / `OutboundMessage` — it never sees a raw platform payload.

Channel **connection state** and **health** (`CONNECTED`, `DEGRADED`,
`TOKEN_EXPIRED`, `WEBHOOK_FAILED`, `RATE_LIMITED`, `DISCONNECTED`) are core IP in
the control plane, with token-refresh workers and merchant alerts.

---

## 5. RAG engine (backend-agnostic)

`RagEngine` is our orchestration over OSS libraries (LlamaIndex/Haystack for
ingest/retrieve, OSS parsers for documents) — **not** a license-blocked platform.

```
User query
   ↓ normalize
   ↓ intent classification
   ↓ retrieve:  product retrieval  ┐
                knowledge retrieval ┼─ hybrid (BM25 + vector) with tenant filter
   ↓ rerank
   ↓ context builder (token-budget aware)
   ↓ LLM (via LLMProvider, policy-checked)
   ↓ response
```

The `VectorStore` backend is swappable (pgvector → Qdrant → …) with no change to
application logic. Every retrieval applies the tenant metadata filter.

---

## 6. Product-aware AI

Not just document RAG. The orchestrator routes by intent to the right source:

```
Customer message
      ↓ classify intent
   ┌──┴───────────────┬──────────────────┬─────────────┐
Product query?     Knowledge query?    Order query?   Chit-chat/other
   ↓ yes              ↓ yes               ↓ yes           ↓
Product Service    RAG (knowledge)     Order Service   LLM (guardrailed)
(SKU/stock/price)                      (status/track)
   └──────────────────┴──────────────────┴─────────────┘
                        ↓ context builder → LLM → response policy → channel
```

"Áo này còn size M không?" hits the **Product Service** (live inventory), not a
vector store. Product data is synced per channel (Shopee/TikTok Shop first, since
their commerce APIs are richest — see the
[platform matrix](../platform/platform-capability-matrix.md)).

---

## 7. Conversation orchestrator

The LLM does not decide everything. A deterministic orchestrator owns routing,
policy, and handoff:

```
IncomingMessage (canonical)
      ↓ classifier (intent + language + urgency)
      ↓ router ── product / knowledge / order / smalltalk
      ↓ retrieve context (tenant-scoped)
      ↓ LLM call (LLMProvider; entitlement + quota checked; window-aware)
      ↓ response policy (guardrails, PII, confidence, handoff rules)
      ↓ send via ChannelProvider   OR   hand off to human (inbox)
      ↓ meter usage & cost (always)
```

- **Window-aware:** on Meta, replies outside the 24h window need tags/OTN — the
  orchestrator knows and behaves accordingly.
- **Human handoff:** low confidence, explicit request, or policy triggers route
  the conversation to the unified inbox (Chatwoot or our own — ADR-005).
- **Metering is not optional:** every LLM/embedding/retrieval call writes a usage
  record for cost accounting (§ operations/cost-model).

---

## 8. Scaling roadmap (do not over-engineer the MVP)

| Stage | Tenants | Shape | What changes |
|---|---|---|---|
| **1** | ~10 | Single deployment: 1 API + 1 worker + Postgres + Valkey + object storage; pgvector | Prove the product. One Docker Compose / one small cluster. |
| **2** | ~100 | Horizontal API + worker replicas behind a load balancer; read replica for Postgres | No architectural change — just replicas + autoscaling. |
| **3** | ~1,000 | Split **workers by class**: webhook workers, RAG/embedding workers, channel-sync workers, LLM workers; introduce Qdrant for vectors; Temporal for durable sync flows | Queues per class; isolate the hot path (webhook→reply) from batch (embedding/sync). |
| **4** | 10,000+ | Extract services **at real bottlenecks only**: Channel service, RAG service, Conversation service, Billing service; consider Postgres sharding by `organization_id`, dedicated search cluster | Data-driven, from benchmarks — not "enterprise architecture" for its own sake. |

The modular monolith's module boundaries (§9) are drawn so that any module can
become a service later without rewriting business logic.

---

## 9. Module boundaries (modular monolith)

```
/auth          /tenant        /billing      /usage        /shop
/channel       /customer      /conversation /inbox        /product
/order         /knowledge     /rag          /llm          /agent
/notification  /support       /admin        /analytics
```

Each module exposes: `api` (transport) · `application` (use-cases) · `domain`
(entities/rules) · `infrastructure` (DB/providers). Cross-module calls go through
application interfaces, never by reaching into another module's tables.

---

## 10. Async processing

Never block an HTTP request on: PDF/OCR parsing, chunking, embedding, product
sync, historical message backfill, webhook processing, analytics aggregation.
All of these run on the queue (`QueueProvider` → BullMQ/Valkey; Temporal later).
Webhook intake is: *verify signature → enqueue → 200 OK fast → process async.*

---

## 11. Security boundaries (summary; full list in risk register)

- OAuth tokens **encrypted at rest** (KMS/OpenBao) — never plaintext.
- Webhook **signature verification** + **idempotency keys** on every inbound event.
- **RBAC enforced server-side**; frontend is never trusted for authorization.
- **LLM tool-use only** — no arbitrary DB access from the model; tools are
  tenant-scoped and audited.
- **Prompt-injection & RAG-poisoning** defenses in the response policy layer.
- **PII handling & data retention/deletion** per org (delete org / customer /
  conversation / knowledge).
- **Every privileged action audited.**

---

## 12. Observability

One `correlation_id` follows a request across `webhook → queue → RAG → LLM →
response`. Infra traces/metrics/logs via OpenTelemetry + Prometheus + Grafana +
Loki + Tempo; LLM traces/cost/quality via Langfuse. Not all of this is in the
MVP — `correlation_id` propagation and structured logs are; the full stack is
added as load justifies it.
