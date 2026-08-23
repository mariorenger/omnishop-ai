# ADR-007: LLM & embeddings — provider-abstracted, multi-model, cost-metered

- **Status:** Accepted (Milestone 0)
- **Date:** 2026-08-23
- **Deciders:** CTO / AI Architect

## Context

The LLM is a commodity capability, our single largest variable cost, and a fast-
moving market. We must (a) never hard-code one vendor, (b) route different tasks
to different models by cost/quality, (c) meter every call for per-tenant COGS,
and (d) keep the model on a **short leash** — it may only call audited,
tenant-scoped tools, never the database directly.

## Options

- **Single hosted provider** (e.g. Anthropic Claude, or OpenAI) — simplest, but
  lock-in and no cost routing.
- **Multi-provider behind `LLMProvider`** — Claude / OpenAI / others + optional
  self-hosted open models for cheap/bulk tasks.
- **Gateway/proxy** (e.g. an OSS LLM gateway) in front of providers — central
  routing, keys, rate limits, fallback.

## Decision

- **`LLMProvider` and `EmbeddingProvider` interfaces**; no vendor name appears in
  domain code.
- **Multi-model routing:** cheap/fast model for classification & simple replies;
  stronger model for complex reasoning / low-confidence cases; **self-hosted open
  embeddings (e.g. bge/e5) or hosted embeddings** chosen by cost.
- **Every call metered** (model, input/output tokens, embedding tokens, retrieval
  count, latency, estimated cost) → usage/cost accounting.
- **Observability via Langfuse** (MIT) for traces, evals, and cost attribution.
- **Tool-use only:** the model invokes a fixed set of tenant-scoped tools
  (product lookup, order status, knowledge retrieval); it has **no arbitrary DB
  access**. Prompt-injection/RAG-poisoning defenses live in the response policy.

Default first providers can be picked at Milestone 6; the point of this ADR is
that the *architecture* is provider-agnostic and cost-aware from day one.

## Why

LLM pricing and capability change monthly; abstraction + routing lets us chase
cost/quality without rewrites and protects margin (the biggest lever on per-tenant
profitability). Metering + Langfuse make cost visible per tenant/shop/conversation
(see cost model). Tool-only access is the security boundary that keeps a
jailbroken prompt from reaching another tenant's data.

## Trade-offs

- Multi-provider routing adds complexity (prompt/format differences, eval per
  model) → mitigated by keeping the interface narrow and testing routes with
  Langfuse evals.
- Self-hosting embeddings trades infra for per-token savings → decide per volume.

## Migration path

Swapping or adding a model/provider is an adapter behind `LLMProvider` /
`EmbeddingProvider`. Routing policy is config, not code. If we later adopt an OSS
LLM gateway, it slots in behind the same interface. Metering schema is
provider-neutral (records model name as a string), so historical cost data
survives provider changes.

> Follow the project's LLM guidance (`claude-api` skill / provider docs) for exact
> model IDs, pricing, and features at implementation time — do not hard-code from
> memory.
