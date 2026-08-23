# ADR-006: Queue / async processing — BullMQ on Valkey now, Temporal later

- **Status:** Accepted (Milestone 0)
- **Date:** 2026-08-23
- **Deciders:** CTO / SRE

## Context

Many operations must not block HTTP requests: PDF/OCR parsing, chunking,
embedding, product sync, historical message backfill, webhook processing,
analytics. We need reliable retries, scheduling, and rate limiting, without
heavy infra in the MVP. Later, some flows become **long-running and stateful**
(multi-step channel sync, token-refresh sagas) where a durable-execution engine
pays off.

## Options

- **BullMQ** (MIT, 🟢) on **Valkey** (BSD, 🟢) — Redis-compatible job queue for
  Node/TS; priorities, rate limiting, scheduling, retries.
- **Temporal** (MIT server, 🟢) — durable execution runtime for long, versioned,
  multi-step workflows. Heavier to run.
- **RabbitMQ / Kafka** — brokers; more ops; Kafka for high-throughput streams.
- **Cloud (SQS/etc.)** — managed; vendor coupling.

## Evaluation

| Criterion | BullMQ+Valkey | Temporal | RabbitMQ | Kafka |
|---|---|---|---|---|
| License | 🟢 MIT/BSD | 🟢 MIT | 🟢 | 🟢 |
| MVP simplicity | high (reuse Valkey) | medium/low | medium | low |
| Retries/scheduling | ✅ | ✅ (durable) | ✅ | manual |
| Long-running sagas | limited | excellent | limited | n/a |
| Throughput ceiling | high | high | high | very high |
| Ops burden | low | medium | medium | high |

## Decision

**MVP:** **BullMQ on Valkey** behind a `QueueProvider` interface — we already run
Valkey for cache/broker, so this adds no new datastore. **Stage 3+:** introduce
**Temporal** for durable, long-running workflows (channel sync, token-refresh
sagas, historical backfill), keeping simple fire-and-forget jobs on BullMQ.

## Why

BullMQ+Valkey gives us reliable async with near-zero added infra for the MVP.
Temporal is the right tool for durability/versioning of long flows but is overkill
early; deferring it avoids premature operational weight. Kafka/RabbitMQ aren't
needed until stream volume demands them.

## Trade-offs

- BullMQ ties queueing to Valkey availability → mitigated by Valkey HA at scale.
- Running both BullMQ and Temporal later means two mental models → we draw a clear
  line: **jobs** (BullMQ) vs **workflows/sagas** (Temporal).

## Migration path

`QueueProvider` (`enqueue`, `schedule`, `process`) abstracts the backend. Moving a
flow from BullMQ to Temporal is a per-flow migration, not a global rewrite.
Worker classes (webhook/RAG/sync/LLM) are separable at Stage 3 without code
changes to producers.
