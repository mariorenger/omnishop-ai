# Cost Model

**Purpose:** answer "how much does this cost at 100 / 1,000 / 10,000 / 100,000
tenants?" so architecture decisions are cost-aware and we always know per-tenant
gross margin.

> **These are planning estimates (order-of-magnitude), not quotes.** Actual cost
> depends on message volume, AI usage per tenant, model choice, region, and
> negotiated rates. The **dominant, most volatile cost is LLM/AI usage** — model
> it per-message and re-check with real Langfuse data after Milestone 6. All
> figures are illustrative USD/month.

## Cost drivers

| Category | Driver | Notes |
|---|---|---|
| **LLM inference** | messages × tokens × model price | **Dominant & variable.** Route cheap models for simple replies. |
| **Embeddings** | documents/products × tokens; queries × tokens | One-time (ingest) + ongoing (queries). Self-host to cut. |
| **Vector store** | vectors stored + QPS | pgvector (free-ish) → Qdrant (infra) at scale. |
| **Postgres** | storage + IOPS + replicas | Grows with conversations/products. |
| **Cache/broker (Valkey)** | memory | Small until high concurrency. |
| **Object storage** | GB stored + egress | Documents/media; cheap on R2/B2. |
| **Search (OpenSearch)** | cluster size | Only when hybrid/faceted search added. |
| **Queue/workers** | compute for async | Scales with parsing/embedding/sync volume. |
| **Logs/metrics/traces** | ingest + retention | Sample aggressively; Langfuse for LLM. |
| **Platform APIs** | mostly free; some paid msg types | Meta Sponsored Msgs/OTN have cost. |
| **Payment fees** | % of revenue | 2.9%+30¢ (Stripe) … 5%+50¢ (MoR). |

## Illustrative cost per tenant (assumptions)

Baseline assumptions per **active** tenant (tune later):

- ~3,000 customer messages/month, ~60% answered by AI (~1,800 AI replies).
- Avg AI reply: ~1.5k input tokens (context+RAG) + ~0.3k output tokens.
- ~500 products + ~50 knowledge docs embedded once; ~1,800 query embeddings/mo.
- Mixed model routing (cheap model for ~70% of replies, stronger for ~30%).

Under these assumptions, **AI cost per active tenant lands roughly in the
$3–$12/month range** depending heavily on model mix and context size — this is the
number to protect and to measure with Langfuse, not to trust from this table.

## Scale bands (illustrative monthly infrastructure, excluding LLM & payment fees)

> LLM/embedding cost scales ~linearly with active tenants (use the per-tenant
> range above). The table below is **platform/infra** cost, which scales
> sub-linearly.

| Component | 100 tenants | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| Postgres (managed, + replica from 1k) | ~$100 | ~$500 | ~$3k | ~$20k (sharded) |
| Vector store | pgvector (in PG) | Qdrant small ~$300 | Qdrant ~$2k | Qdrant cluster ~$15k |
| Valkey (cache/broker) | ~$30 | ~$150 | ~$800 | ~$6k |
| Object storage (R2/B2) | ~$20 | ~$100 | ~$800 | ~$6k |
| Workers/API compute | ~$150 | ~$800 | ~$6k | ~$50k |
| OpenSearch (when added) | — | ~$300 | ~$2k | ~$15k |
| Observability + Langfuse | ~$50 | ~$250 | ~$1.5k | ~$10k |
| Keycloak + misc infra | ~$50 | ~$150 | ~$800 | ~$5k |
| **Infra subtotal (≈)** | **~$0.5k** | **~$3k** | **~$18k** | **~$130k** |
| **+ LLM/embeddings (≈, active)** | per-tenant × active | " | " | " |
| **+ Payment fees** | % of revenue | " | " | " |

**Per-tenant infra cost falls with scale** (~$5 → ~$3 → ~$1.8 → ~$1.3), while
**LLM cost stays roughly per-tenant-constant** — so at scale, *AI usage is the
margin lever*, not infrastructure.

## Unit economics example (illustrative)

```
Tenant on a $49/mo plan
  Revenue:            $49.00
  AI (LLM+embeddings): -$7.00   ← the lever; measure with Langfuse
  Infra (amortized):   -$2.00
  Payment fee (~3%):   -$1.50
  Platform msg costs:  -$0.50
  ─────────────────────────────
  Gross margin:       ~$38.00  (~78%)
```

The admin control plane must compute this **per tenant, per shop, per
conversation** from usage records — so we can answer *"which customer is
unprofitable?"* and act (plan change, model routing, quota).

## Cost-control levers (built into the architecture)

1. **Model routing** (ADR-007): cheap model for classification/simple replies.
2. **Context budgeting**: cap tokens per RAG context; rerank to fewer, better chunks.
3. **Caching**: cache embeddings and frequent answers; dedupe identical queries.
4. **Self-hosted embeddings** at volume.
5. **Quota enforcement** (ADR-004): plans cap AI messages; overage = upsell, not loss.
6. **Metering everything** (ADR-007): no blind spots; alerts on cost anomalies.

## What to measure post-MVP (replace estimates with reality)

- Actual tokens/reply and model mix (Langfuse) → true AI cost/tenant.
- pgvector latency vs tenant/vector growth → when to move to Qdrant (ADR-002).
- Worker cost per 1k messages → when to split worker classes (Stage 3).
- Per-tenant margin distribution → pricing & plan design feedback.
