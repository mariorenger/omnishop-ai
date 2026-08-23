# Risk Register

Scored **Likelihood × Impact** (Low/Med/High). Owner is the role accountable.
Reviewed each milestone. IDs are stable.

| ID | Risk | L | I | Mitigation | Owner |
|---|---|---|---|---|---|
| **R-01 Platform API change** | Meta/TikTok/Shopee change APIs, deprecate versions, or tighten policy, breaking channels | High | High | Channel logic behind `ChannelProvider`; version-pin API calls; monitor deprecation notices; channel-health alerts; contract tests per provider | Platform Eng |
| **R-02 App-review / access denied** | Meta app review rejected; Shopee/TikTok chat API not granted per region/partner | High | High | Milestone 4 spike confirms access **before** roadmap commit; ship Meta first (most attainable); TikTok/Shopee as product/order first, chat later | CTO |
| **R-03 OAuth token lifecycle** | Tokens expire/revoke; refresh fails; webhooks silently stop | High | Med | Token-refresh workers; `TOKEN_EXPIRED`/`WEBHOOK_FAILED` health states; merchant alerts; retry+backoff; encrypt tokens at rest | SRE |
| **R-04 License violation** | Using an OSS project whose license forbids multi-tenant SaaS / hosted service | Med | High | OSS scorecard verdicts enforced; **Dify/n8n/Elasticsearch-ELv2/MinIO blacklisted**; pin & record audited commit; legal review before adopting any 🟡 | CTO |
| **R-05 AI cost blowout** | LLM spend outruns revenue; an unprofitable tenant goes unnoticed | High | High | Meter every call; per-tenant COGS in admin; model routing; context budgeting; quota enforcement; cost-anomaly alerts | AI Architect |
| **R-06 Tenant data leak** | Cross-tenant read via missed filter / vector bleak / prompt injection | Med | High | 3-layer isolation (app + RLS + metadata filter) — ADR-008; cross-tenant tests assert failure; LLM tool-only access; audit log | Security |
| **R-07 Prompt injection / RAG poisoning** | Malicious content in a customer message or ingested doc manipulates the AI | Med | High | Response-policy guardrails; tool-only model access; untrusted-content handling; no secrets in prompts; eval suite | AI Architect |
| **R-08 Data privacy / retention (PII, GDPR-style)** | Storing customer PII indefinitely; no deletion path | Med | High | Retention policy; delete org/customer/conversation/knowledge; encrypt at rest; data-processing terms; region awareness | Security |
| **R-09 Vendor lock-in** | Hard dependency on one LLM/vector/billing/inbox vendor | Med | Med | Provider interfaces for all replaceables; standard protocols (OIDC/S3); documented migration paths in ADRs | Architect |
| **R-10 Scaling / bottleneck** | pgvector or shared Postgres saturates; webhook storm overwhelms sync path | Med | High | Async-first; separate worker classes at Stage 3; read replicas → sharding by org; load tests before extraction (Stage 4) | SRE |
| **R-11 Operational complexity** | Too many moving parts too early for a small team | Med | Med | Modular monolith first; defer Temporal/Qdrant/OpenSearch/microservices until workload justifies | CTO |
| **R-12 Payment / MoR coverage (VN/SEA)** | Chosen provider lacks VN methods or tax handling | Med | Med | `PaymentProvider` abstraction; Stripe + local (VNPay/MoMo/ZaloPay) adapters; verify MoR SEA coverage before Milestone 2 | Product Eng |
| **R-13 Chatwoot integration misfit** | Chatwoot multi-tenancy/API/branding (EE) doesn't fit; sync complexity | Med | Med | `SupportProvider` boundary; our conversation model is source of truth; Milestone 8 spike; fallback = build inbox (ADR-005) | Product Eng |
| **R-14 Webhook reliability / idempotency** | Duplicate/out-of-order/lost webhooks cause double-replies or missed messages | Med | Med | Signature verify + idempotency keys; fast-ack then async; dead-letter queue; replay tooling | SRE |
| **R-15 24-hour window / messaging policy** | Product promises out-of-window messaging that platforms forbid | Med | Med | Window-aware orchestrator; product messaging honest about limits; tags/OTN where allowed | Product Eng |
| **R-16 Model quality / hallucination** | AI gives wrong product/stock/price/order info to customers | Med | High | Product-aware routing to live data (not vectors); confidence-based handoff; guardrails; eval suite with Langfuse | AI Architect |
| **R-17 Key/secret exposure** | OAuth tokens or API keys leaked (repo, logs, plaintext DB) | Low | High | Secrets in KMS/OpenBao; encrypt tokens at rest; secret scanning in CI; never log secrets; least-privilege | Security |
| **R-18 Regional/compliance (VN/SEA)** | Local data/commerce regulations, platform regional rules | Med | Med | Region-aware config; legal review per market; data residency options at enterprise tier | CTO |

## Top risks to watch (the ones that can sink the project)

1. **R-02 / R-01** — if we can't get *messaging* access on target channels, the
   product's core value is blocked. **Meta-first + Milestone 4 spike** is the
   direct mitigation.
2. **R-05** — AI cost is the margin. Metering + routing + quotas from day one.
3. **R-06 / R-07** — a cross-tenant leak or a manipulated AI is existential.
   Three-layer isolation + tool-only LLM + guardrails, tested.
4. **R-04** — a license misstep (Dify/n8n) could force a costly rebuild. Enforced
   by the scorecard blacklist.

## Review cadence

Re-score at every milestone gate; re-verify all OSS licenses and platform-API
capabilities (they drift) before Milestone 1 and before each channel ships.
