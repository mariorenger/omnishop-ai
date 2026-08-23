# OSS Scorecard

**Purpose:** decide, per module, what to reuse vs. build — and, critically,
whether each candidate's **license** actually permits use in a *commercial,
multi-tenant SaaS that we host and sell*.

> **Licenses verified as-of August 2026** via each project's own license
> file/docs. Licenses change (Redis, Elastic, MinIO, Dify and HashiCorp all
> changed within recent years). **Re-verify before committing**, and pin the
> exact version/commit whose license you audited. This file records findings and
> sources, not legal advice.

## The one test that matters here

Many "open source" projects are fine to *self-host for internal use* but forbid
**offering the software as a hosted service to third parties** — which is exactly
our business model. We tag each candidate:

- 🟢 **SAFE** — permissive (MIT / Apache-2.0 / BSD / PostgreSQL): use freely,
  including as the backend of a paid multi-tenant SaaS.
- 🟡 **CONDITIONAL** — usable but with obligations (copyleft/AGPL, or a
  dual-license where a needed feature sits in a paid EE tier). Legal review +
  architectural care required.
- 🔴 **BLOCKED for our use case** — license forbids offering it as a hosted
  service / multi-tenant SaaS without a written commercial agreement.

---

## Scorecard

| Module | Candidate | License | License verdict | Maturity | SaaS-suitability | Recommendation |
|---|---|---|---|---|---|---|
| **Support / Inbox** | Chatwoot | MIT core + commercial `enterprise/` dir | 🟢 core / 🟡 EE | High | Good (self-host MIT core) | **OSS+CUSTOMIZE** — use MIT core; custom-branding & SSO are EE (paid). Wrap behind `SupportProvider`. |
| **RAG platform** | Dify | Apache-2.0 **with additional conditions** | 🔴 | High | **Forbidden** — "may not use … to operate a multi-tenant environment" without written authorization | **DO NOT use as core.** Optionally study its UX. |
| **RAG library** | LlamaIndex | MIT | 🟢 | High | Excellent (library, not a platform) | **USE OSS** as ingestion/retrieval toolkit under our own `RagEngine`. |
| **RAG library** | Haystack (deepset) | Apache-2.0 | 🟢 | High | Good | Viable alternative/supplement to LlamaIndex. |
| **RAG platform** | RAGFlow | Apache-2.0 | 🟢 | Medium/High | Good | Candidate for document parsing/OCR-heavy ingestion; keep behind our interface. |
| **Vector store** | Qdrant | Apache-2.0 | 🟢 | High | Excellent | **USE OSS** (self-host) behind `VectorStore`. Strong default at scale. |
| **Vector store** | pgvector | PostgreSQL license | 🟢 | High | Excellent for MVP | **USE OSS** — start here (no extra infra). Migrate to Qdrant when it hurts. |
| **Vector store** | Weaviate | BSD-3-Clause | 🟢 | High | Good | Permissive alternative to Qdrant. |
| **Search / lexical** | OpenSearch | Apache-2.0 | 🟢 | High | Excellent | **USE OSS** for hybrid (BM25 + kNN) / faceting when needed. |
| **Search / lexical** | Elasticsearch | ELv2 / SSPL / AGPL-3 (tri-license) | 🔴 (ELv2 blocks SaaS) / 🟡 (AGPL) | High | ELv2 forbids offering as a managed service | **Avoid.** Use OpenSearch instead (Apache-2.0 fork). |
| **Auth / IdP** | Keycloak | Apache-2.0 | 🟢 | High | Good; has Organizations (multi-tenant) | **USE OSS** behind our auth boundary — permissive, self-hostable. |
| **Auth / IdP** | Supabase Auth (GoTrue) | Apache-2.0 | 🟢 | High | Good | Viable, lighter alternative to Keycloak. |
| **Auth / IdP** | Zitadel | AGPL-3.0 | 🟡 | High | Purpose-built multi-tenant, but AGPL | Only if legal accepts AGPL. Keycloak is the permissive default. |
| **Auth / IdP** | Ory (Kratos/Hydra) | Apache-2.0 | 🟢 | High | API-first, headless | Good if we want to compose our own flows. |
| **Workflow / automation** | n8n | Sustainable Use License (fair-code) | 🔴 | High | **Forbidden** — may not offer n8n as a hosted service to third parties | **DO NOT embed** in the product. Internal ops only, if at all. |
| **Durable workflow** | Temporal | MIT (server) | 🟢 | High | Excellent for long-running orchestration | **DEFER** to post-MVP; strong later choice for sync/retry-heavy flows. |
| **Queue / jobs** | BullMQ | MIT (on Redis/Valkey) | 🟢 | High | Excellent for MVP | **USE OSS** behind `QueueProvider`. |
| **In-memory / broker** | Valkey | BSD-3-Clause (Linux Foundation fork of Redis) | 🟢 | High | Excellent | **USE OSS** — permissive Redis-compatible. See Redis note below. |
| **In-memory / broker** | Redis | AGPL-3 option added (2025), after RSALv2/SSPL (2024) | 🟡 | High | Works, but license churn risk | Prefer **Valkey** to avoid license churn; Redis remains API-compatible. |
| **Object storage** | MinIO | AGPL-3.0 | 🟡/🔴 | High | AGPL obligations for a hosted service; commercial license sold separately | **Avoid as embedded dependency.** Prefer S3-compatible cloud (R2/S3/B2) behind `ObjectStorage`; SeaweedFS (Apache-2.0) if we must self-host. |
| **Object storage** | SeaweedFS | Apache-2.0 | 🟢 | Medium/High | Good self-host, S3-compatible | Permissive self-host fallback. |
| **LLM observability** | Langfuse | MIT core (EE thin) | 🟢 | High | Excellent self-host | **USE OSS** for tracing/eval/cost of LLM calls. (Acquired by ClickHouse Jan 2026; no license change stated.) |
| **Observability (infra)** | OpenTelemetry + Prometheus + Grafana + Loki + Tempo | Apache-2.0 (OTel/Prom/Tempo/Loki) / AGPL-3 (Grafana) | 🟢 / 🟡 (Grafana) | High | Standard stack | **USE OSS**; Grafana AGPL only affects modifying Grafana itself (we use it as-is). |
| **Error tracking** | Sentry (self-hosted) | FSL / BSL-style (Functional Source License) | 🟡 | High | Self-host allowed for internal use; competing-product restriction | Fine as an internal tool (we're not selling an error-tracker). GlitchTip (MIT) is a permissive alternative. |
| **Billing (subscriptions)** | Stripe Billing | External SaaS (not OSS) | n/a | High | We are Merchant of Record | **EXTERNAL** behind `BillingProvider`. See ADR-004 for MoR trade-off. |
| **Billing (MoR)** | Paddle / Lemon Squeezy | External SaaS | n/a | High | Handles global tax; weaker SEA coverage | **EXTERNAL** option; verify Vietnam/SEA payment-method coverage. |
| **Local payments (VN/SEA)** | VNPay / MoMo / ZaloPay | External | n/a | High | Needed for VN customers | **EXTERNAL** adapter(s) under `BillingProvider`/`PaymentProvider`. |
| **Email** | Resend / Postmark / SES | External | n/a | High | Transactional email | **EXTERNAL** behind `EmailProvider`. |
| **Secrets** | Infisical / OpenBao | MIT (Infisical core) / MPL-2.0 (OpenBao) | 🟢 | Medium/High | Good | **USE OSS** or cloud KMS; OpenBao is the permissive Vault fork. |
| **Admin control plane** | (Retool/Forest/etc.) | mixed / external | varies | — | Ties admin to a vendor's model | **BUILD** — admin operates our tenancy/billing/cost IP; must be ours. |
| **Doc parsing / OCR** | Unstructured / Docling / Apache Tika | Apache-2.0 (Tika/Docling) / mixed (Unstructured) | 🟢 / 🟡 | Medium/High | Good | **USE OSS** for parsing behind our ingestion; verify Unstructured's component licenses. |

---

## Red-flag summary (the ones that would bite us)

| Project | Why it's blocked/risky for us | What we do instead |
|---|---|---|
| **Dify** | License forbids operating a **multi-tenant environment** without written authorization — that *is* our product. | Build our own `RagEngine` on MIT/Apache libraries. |
| **n8n** | Sustainable Use License forbids **offering n8n as a hosted service** to third parties. | Don't embed. Use Temporal/BullMQ for orchestration. |
| **Elasticsearch (ELv2)** | ELv2 forbids providing it as a **managed/hosted service**. | Use **OpenSearch** (Apache-2.0). |
| **MinIO (AGPL-3)** | AGPL network-copyleft obligations when offered as a service; commercial license sold separately. | Cloud S3 (R2/S3/B2), or **SeaweedFS** (Apache-2.0). |
| **Redis (license churn)** | Moved RSALv2/SSPL (2024) → re-added AGPL (2025); future uncertain. | Use **Valkey** (BSD, API-compatible). |
| **Grafana / Sentry** | AGPL / FSL — fine to *use as-is* internally; obligations trigger if we *modify & redistribute* or build a competing product. | Use as-is; GlitchTip (MIT) as a Sentry alternative if needed. |

## Sources (verified Aug 2026)

- Chatwoot license (MIT core + `enterprise/`): <https://github.com/chatwoot/chatwoot>
- Dify license (Apache-2.0 + conditions, multi-tenant restriction): <https://github.com/langgenius/dify/blob/main/LICENSE>
- n8n Sustainable Use License: <https://docs.n8n.io/sustainable-use-license/>
- Qdrant (Apache-2.0), Weaviate (BSD-3), pgvector (PostgreSQL): <https://openalternative.co/compare/qdrant/vs/weaviate>
- Elasticsearch tri-license (ELv2/SSPL/AGPL): <https://www.elastic.co/pricing/faq/licensing>
- LlamaIndex (MIT) / Haystack (Apache-2.0) / RAGFlow (Apache-2.0): <https://atlan.com/know/enterprise-rag-platforms-comparison/>
- Langfuse (MIT core; ClickHouse acquisition Jan 2026): <https://dev.to/beton/langfuse-pricing-teardown-2026-2pi9>
- Keycloak (Apache-2.0) / Supabase Auth (Apache-2.0) / Zitadel (AGPL): <https://skycloak.io/blog/open-source-authentication-comparison-2026/>
- MinIO (AGPL-3): <https://github.com/minio/minio/blob/master/LICENSE>
- Stripe vs Paddle vs Lemon Squeezy (MoR, SEA coverage): <https://www.artisangrowthstrategies.com/blog/paddle-vs-stripe-vs-lemon-squeezy-2026>

> **Action before Milestone 1:** open each candidate's `LICENSE` at the exact
> version we plan to pin, and record the commit hash next to the verdict above.
