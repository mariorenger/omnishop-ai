# Milestone 0 — Discovery

This is the index for the OmniShop AI discovery phase. **No application code is
written in this milestone.** The output is a set of decisions and an architecture
that later milestones implement.

## Reading order

1. [`product/vision.md`](product/vision.md) — what we are building and for whom.
2. [`research/oss-scorecard.md`](research/oss-scorecard.md) — OSS landscape with
   **verified licenses** (the licenses were checked live in Aug 2026; see the
   "as-of" note in that file).
3. [`platform/platform-capability-matrix.md`](platform/platform-capability-matrix.md)
   — what each sales channel's API can actually do.
4. [`research/build-vs-buy-matrix.md`](research/build-vs-buy-matrix.md) — the
   make/buy/OSS decision per module, driven by (2) and (3).
5. [`architecture/overview.md`](architecture/overview.md) — control/data plane
   split, multi-tenancy, channel adapters, RAG engine, conversation
   orchestrator, and the MVP → 5-year scaling roadmap.
6. [`decisions/`](decisions/) — the ADRs that lock in the load-bearing choices.
7. [`operations/cost-model.md`](operations/cost-model.md) and
   [`operations/risk-register.md`](operations/risk-register.md).

## Milestone map

| # | Milestone | Gate |
|---|---|---|
| **0** | **Discovery** (this) | Architecture & make/buy approved |
| 1 | Foundation — repo, Docker Compose, Postgres, Redis, storage, auth, tenant, RBAC, audit log | — |
| 2 | Billing — plans, subscriptions, entitlements, usage, quota | — |
| 3 | Channel core — `ChannelProvider` abstraction, connection state, webhook intake, health; fake provider for tests | — |
| 4 | First real channel — **Meta (Messenger + Instagram)**: OAuth, webhook, send/receive, health, reconnect | — |
| 5 | Knowledge — upload, parse, chunk, embed, index, retrieve, rerank | — |
| 6 | AI chatbot — conversation, RAG, LLM, guardrails, fallback, human handoff | — |
| 7 | Product AI — product sync, product-aware retrieval, inventory/price/variant | — |
| 8 | Unified inbox — integrate Chatwoot **or** build (decided in ADR-005) | — |
| 9 | Operations — admin, tenant/channel health, support, usage, cost, audit, monitoring | — |
| 10 | Scale — only after real benchmarks (load test → service extraction) | — |

## Guiding principles (applied throughout)

- **OSS-first, lock-in-averse:** Can OSS solve this? Can it be safely integrated?
  Can we wrap it behind an interface so it is replaceable? Only then build our own.
- **Core IP is always ours:** tenancy, billing, entitlements, usage/cost
  metering, conversation orchestration, and the admin control plane are never
  outsourced to an OSS project's data model.
- **Everything replaceable is behind a provider interface:** `LLMProvider`,
  `EmbeddingProvider`, `VectorStore`, `ObjectStorage`, `BillingProvider`,
  `QueueProvider`, `ChannelProvider`, `SupportProvider`.
- **Cheap at 10 tenants → efficient at 100 → scalable at 1,000 → separable at
  10,000+.** Do not microservice the MVP.
- **License is checked, not assumed.** "Open source" on a README is not
  permission to run a commercial multi-tenant SaaS.
