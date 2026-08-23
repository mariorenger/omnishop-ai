# ADR-002: Vector store — pgvector now, Qdrant at scale

- **Status:** Accepted (Milestone 0)
- **Date:** 2026-08-23
- **Deciders:** CTO / AI Architect

## Context

RAG needs vector similarity search with **hard tenant isolation** (metadata
filter on `organization_id` / `shop_id` / `knowledge_base_id`). We want to avoid
new infrastructure in the MVP but not paint ourselves into a corner at scale.

## Options

- **pgvector** (PostgreSQL license, 🟢) — vectors inside our existing Postgres.
- **Qdrant** (Apache-2.0, 🟢) — dedicated Rust vector DB, strong filtering/scale.
- **Weaviate** (BSD-3, 🟢) — dedicated, permissive.
- **Elasticsearch kNN** (ELv2, 🔴 for SaaS) — excluded on license.
- **Pinecone** (external SaaS) — managed, but adds a vendor + per-vector cost.

## Evaluation

| Criterion | pgvector | Qdrant | Weaviate | Pinecone |
|---|---|---|---|---|
| License | 🟢 | 🟢 | 🟢 | external |
| New infra in MVP | none | yes | yes | yes (SaaS) |
| Metadata filtering | good | excellent | excellent | good |
| Scale (vectors/QPS) | moderate | high | high | high |
| Cost at small scale | ~free | small | small | pay-per-vector |
| Lock-in | none | low | low | high |

## Decision

Start with **pgvector** (MVP, Stages 1–2). Migrate the vector workload to
**Qdrant** (self-hosted, Apache-2.0) at **Stage 3 (~1,000 tenants)** or earlier
if retrieval latency/QPS on pgvector degrades. Both sit behind a `VectorStore`
interface; **Elasticsearch is excluded on license**, Pinecone avoided for lock-in.

## Why

pgvector removes an entire moving part for the MVP and reuses our isolation model
(RLS + metadata columns). Qdrant is the permissive, high-scale successor with
excellent payload filtering, so the migration is a swap behind the interface, not
a rewrite.

## Trade-offs

- pgvector shares Postgres resources — heavy embedding search can compete with
  OLTP; mitigated by read replicas and by migrating before it hurts.
- Running Qdrant later adds ops; accepted as a scale-stage cost.

## Migration path

`VectorStore` interface (`upsert`, `query(filter)`, `delete`) implemented by both
`PgVectorStore` and `QdrantStore`. Migration = dual-write → backfill → cut reads
over → drop pgvector index. No change to `RagEngine` or application code.
