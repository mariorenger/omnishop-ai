# Build vs Buy vs OSS Matrix

**Decision key:** `BUILD` (our IP) · `OSS` (reuse as-is behind an interface) ·
`OSS+CUSTOM` (reuse + our layer) · `EXTERNAL` (managed SaaS) · `BUY` (paid EE) ·
`DEFER` (not in MVP).

The rule applied throughout: **can OSS solve it → can it be safely integrated →
can we wrap it behind an interface → can we replace it later → only then BUILD.**
But **core business logic (tenancy, billing state, entitlements, usage/cost,
orchestration, admin) is always BUILD**, regardless of what OSS exists, because
it is the competitive moat and the thing we must never lose control of.

## Decision tree

```
                          DECISION
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                     │
   SaaS Core            AI / RAG Core          Platform / Infra
   (business IP)        (differentiator)       (commodity)
        │                    │                     │
      BUILD              OSS + our              OSS or EXTERNAL
   tenancy, billing,     orchestration         behind provider
   entitlements,         (RagEngine over       interfaces
   usage, cost,          MIT/Apache libs)      (replaceable)
   orchestration,
   admin
```

## Matrix

| Module | Decision | Technology | Why |
|---|---|---|---|
| **Tenant / org / membership** | **BUILD** | Postgres + our domain model | Core IP; every resource's ownership & isolation depend on it. |
| **Billing state / subscription / entitlements** | **BUILD** (state) + **EXTERNAL** (payments) | Our entitlement engine + Stripe/Paddle/local | Payment rails are commodity; *what a plan grants and quota enforcement* is our IP. See ADR-004. |
| **Usage metering & cost accounting** | **BUILD** | Postgres (+ time-series later) | Per-tenant COGS/margin is a core differentiator; no OSS owns our cost model. |
| **Auth / identity** | **OSS** behind our boundary | Keycloak (Apache-2.0) | Permissive, self-hostable, has Organizations. Wrap so it's replaceable. See ADR-003. |
| **RBAC / permissions** | **BUILD** (policy) on OSS primitives | Our policy layer (+ Keycloak roles / optional Cerbos/OpenFGA) | Authorization is business logic; must never trust the frontend. |
| **Audit log** | **BUILD** | Append-only Postgres table | Every admin/privileged action recorded; core to operability & compliance. |
| **Channel adapters** | **BUILD** | `ChannelProvider` interface + per-platform impls | The abstraction that keeps platform logic out of RAG/LLM/conversation. Core IP. See ADR-... /architecture. |
| **Channel connection & health state** | **BUILD** | Postgres + workers | Token lifecycle, webhook health, reconnect = operability moat. |
| **Conversation orchestrator** | **BUILD** | Our classifier + router | Decides RAG vs product vs order vs handoff; the brain of the product. |
| **RAG orchestration (`RagEngine`)** | **BUILD** | Our pipeline over OSS libs | Business-critical; must be backend-agnostic. **Not** Dify (license-blocked). |
| **RAG libraries (ingest/retrieve)** | **OSS** | LlamaIndex (MIT) / Haystack (Apache-2.0) | Reuse chunking/retrieval; keep behind `RagEngine`. |
| **Document parsing / OCR** | **OSS** | Apache Tika / Docling / RAGFlow / Unstructured | Don't hand-write PDF/DOCX parsers; verify Unstructured component licenses. |
| **Vector store** | **OSS** behind `VectorStore` | pgvector (MVP) → Qdrant (scale) | Start with no extra infra; migrate when it hurts. See ADR-002. |
| **Lexical / hybrid search** | **OSS** | OpenSearch (Apache-2.0) | BM25 + kNN + faceting; **not** Elasticsearch (ELv2 blocks SaaS). |
| **LLM access** | **EXTERNAL** behind `LLMProvider` | Claude / OpenAI / others | Commodity capability, huge cost lever; must be swappable per-tenant/route. See ADR-007. |
| **Embeddings** | **EXTERNAL/OSS** behind `EmbeddingProvider` | Hosted embeddings or self-host (bge/e5) | Swappable; cost-metered. |
| **LLM observability / eval / cost** | **OSS** | Langfuse (MIT core) | Trace every LLM call for cost & quality; self-host. |
| **Queue / background jobs** | **OSS** behind `QueueProvider` | BullMQ (MIT) on Valkey (BSD) | All async work (parsing, embedding, sync, webhooks). See ADR-006. |
| **Durable long-running workflows** | **DEFER → OSS** | Temporal (MIT) | Add when sync/retry flows outgrow simple queues. |
| **Object storage** | **EXTERNAL/OSS** behind `ObjectStorage` | S3-compatible cloud (R2/S3/B2) → SeaweedFS if self-host | Avoid MinIO AGPL as embedded dep. |
| **Relational DB** | **OSS** | PostgreSQL | Relational + JSONB + RLS + pgvector in one. See ADR-001. |
| **Cache / broker** | **OSS** | Valkey (BSD) | Redis-compatible without license churn. |
| **Support / unified inbox** | **OSS+CUSTOM** or **BUILD** (decide Milestone 8) | Chatwoot MIT core behind `SupportProvider` **or** own inbox | Reuse mature inbox UI; keep tenancy/AI ours. Custom branding/SSO are Chatwoot EE (paid). See ADR-005. |
| **Merchant support (tickets)** | **BUILD** (thin) | Our ticketing on core model | Ties to tenant/channel/usage health for diagnosis. |
| **Admin control plane** | **BUILD** | Our dashboard on core model | Operates our tenancy/billing/cost IP; never a vendor's model. |
| **Observability (infra)** | **OSS** | OpenTelemetry + Prometheus + Grafana + Loki + Tempo | Standard; trace by `correlation_id` across webhook→queue→RAG→LLM. |
| **Error tracking** | **OSS** | GlitchTip (MIT) or self-host Sentry (as-is) | Internal tool; watch Sentry FSL if modifying. |
| **Email (transactional)** | **EXTERNAL** behind `EmailProvider` | Resend / Postmark / SES | Commodity. |
| **Secrets management** | **OSS/EXTERNAL** | OpenBao (MPL-2.0) / cloud KMS / Infisical (MIT) | Encrypt OAuth tokens at rest; never plaintext. |
| **Local payments (VN/SEA)** | **EXTERNAL** behind `PaymentProvider` | VNPay / MoMo / ZaloPay | Needed for the target market; wrap uniformly. |

## What we deliberately do NOT build

- PDF/DOCX/OCR parsers → OSS.
- Vector search engine → OSS.
- Auth/IdP core → OSS (Keycloak).
- Inbox UI → likely Chatwoot MIT core (ADR-005).
- LLM/embeddings → external providers.
- Payment processing → external.

## What we always build (the moat)

Tenancy · billing state & entitlements · usage/cost metering · channel adapters &
health · conversation orchestration · RAG orchestration · admin control plane ·
audit log · authorization policy.

## What we blacklist (license/lock-in)

- **Dify** as core (multi-tenant SaaS forbidden by license).
- **n8n** embedded (hosted-service offering forbidden).
- **Elasticsearch** under ELv2 (managed-service forbidden) → OpenSearch.
- **MinIO** as embedded dep (AGPL) → cloud S3 / SeaweedFS.
- **Redis** as new dependency (license churn) → Valkey.

See [`oss-scorecard.md`](oss-scorecard.md) for the license evidence and
[`../decisions/`](../decisions/) for the load-bearing ADRs.
