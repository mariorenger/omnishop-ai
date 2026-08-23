# OmniShop AI

**Multi-tenant SaaS AI chatbot platform for social- and e-commerce merchants.**

A merchant signs up, picks a plan, pays, connects their sales channels
(Facebook, Instagram, TikTok Shop, Shopee…), imports products/knowledge, and an
AI RAG chatbot starts answering their customers automatically — across every
channel, with human handoff when needed.

> **Pay → Connect → AI works.**
> The merchant never has to know there is a vector DB, an LLM, OAuth, webhooks,
> Docker or Kubernetes behind it.

---

## Status: Milestone 1 — runnable MVP (core loop working)

The **discovery docs** (Milestone 0) are in [`docs/`](docs/). On top of them, a
first **runnable product** now implements the core loop end-to-end and has been
verified against real Postgres+pgvector+RLS+Redis. See
[`IMPLEMENTATION.md`](IMPLEMENTATION.md) for what's built and how it maps to the
milestones.

### Run it (no API key required)

```bash
docker compose up -d --build                       # postgres+pgvector, valkey, api, worker
docker compose exec api python -m scripts.seed     # demo tenant, products, knowledge
# open http://localhost:8000   (login: demo@omnishop.local / demo12345)
```

The seed prints a widget URL — open it and ask *"Áo thun size M màu đen còn
không?"*. It answers from the merchant's **product + inventory** data. Runs with a
retrieval-grounded stub LLM out of the box; set `ANTHROPIC_API_KEY` in `.env` for
real Claude. Verify tenant isolation: `docker compose exec api python -m
scripts.test_isolation`.

> **Guiding principle:** Do not build from scratch what good OSS already solves.
> Do not depend on OSS where it costs us business control, tenant isolation, cost
> visibility, or core differentiation. Do not optimize for scale before there is
> real workload.

### Where to start reading

| Document | What it is |
|---|---|
| [`docs/README.md`](docs/README.md) | **Milestone 0 index** — read this first |
| [`docs/product/vision.md`](docs/product/vision.md) | Product vision, target users, the "Pay → Connect → AI" journey |
| [`docs/research/oss-scorecard.md`](docs/research/oss-scorecard.md) | OSS candidates evaluated per module, **with verified licenses** |
| [`docs/platform/platform-capability-matrix.md`](docs/platform/platform-capability-matrix.md) | Facebook / Instagram / TikTok Shop / Shopee capability matrix |
| [`docs/research/build-vs-buy-matrix.md`](docs/research/build-vs-buy-matrix.md) | Per-module BUILD / OSS / BUY / DEFER decisions |
| [`docs/architecture/overview.md`](docs/architecture/overview.md) | Control plane / data plane, multi-tenancy, MVP → 5-year scaling |
| [`docs/decisions/`](docs/decisions/) | Architecture Decision Records (ADR-001 … ADR-008) |
| [`docs/operations/cost-model.md`](docs/operations/cost-model.md) | Cost model at 100 / 1k / 10k / 100k tenants |
| [`docs/operations/risk-register.md`](docs/operations/risk-register.md) | Risk register (platform, license, AI cost, isolation, privacy…) |

### The headline decisions (details in the docs)

- **Core SaaS IP — BUILD:** tenant/org model, billing state, entitlements,
  usage metering, cost accounting, channel-connection state, conversation
  orchestration, admin control plane.
- **AI/RAG — OSS libraries + our own orchestration:** build the `RagEngine`
  ourselves on top of MIT/Apache libraries (LlamaIndex, parsers), **not** on a
  platform whose license forbids multi-tenant SaaS.
- **Two licenses block our exact use case:** **Dify** (no multi-tenant SaaS
  without a written commercial license) and **n8n** (no offering as a hosted
  service). Both are RED-FLAGGED for the core product.
- **First real channel — Meta (Facebook Messenger + Instagram):** the most
  mature *messaging* APIs, which is where the chatbot's value lives.
  TikTok Shop & Shopee are commerce-first; their messaging APIs are
  partner-gated — layer them in later for product/order sync.

---

## Repository layout

```
omnishop-ai/
├── README.md              ← you are here
├── IMPLEMENTATION.md      ← what the running app does + how to run/verify
├── docker-compose.yml     ← postgres+pgvector, valkey, api, worker
├── app/                   ← modular monolith (FastAPI) + worker
│   ├── providers/         ← LLM, embeddings, vector, queue, channel (swappable)
│   ├── modules/           ← auth, tenant, billing, usage, knowledge, product,
│   │                        channel, conversation (+orchestrator), admin
│   └── static/            ← dashboard + customer chat widget
├── db/001_init.sql        ← schema + Row-Level Security + pgvector + seed plans
├── scripts/               ← seed.py, test_isolation.py
└── docs/                  ← Milestone 0 discovery
    ├── product/  research/  platform/  architecture/  decisions/  operations/
```
