# ADR-001: Primary database — PostgreSQL

- **Status:** Accepted (Milestone 0)
- **Date:** 2026-08-23
- **Deciders:** CTO / Principal Architect

## Context

We need one primary datastore for control-plane data (tenants, billing,
entitlements, usage) and much of the data-plane (conversations, products,
knowledge metadata). We need: relational integrity, JSONB for raw/flexible
payloads, strong multi-tenant isolation, vector search for the MVP, and fuzzy
matching — without standing up five datastores on day one. Team is small; ops
budget is tight.

## Options

- **PostgreSQL** (OSI/PostgreSQL license, 🟢) — relational + JSONB + Row-Level
  Security + `pgvector` + `pg_trgm` in one engine.
- **MySQL/MariaDB** — relational, weaker JSONB/RLS/vector story.
- **MongoDB** (SSPL, 🔴 for offering as a service) — document-first, license risk,
  weaker relational integrity for billing.
- **Split datastores from day one** (Postgres + dedicated vector DB + search) —
  more power, much more ops.

## Evaluation

| Criterion | Postgres | MySQL | MongoDB |
|---|---|---|---|
| License / SaaS-suitability | 🟢 permissive | 🟢 | 🔴 SSPL |
| Relational integrity (billing) | Strong | Strong | Weak |
| JSONB / flexible payloads | Strong | OK | Native |
| Row-Level Security (isolation) | ✅ | ❌ | ❌ |
| Vector search built-in | ✅ pgvector | ❌ | limited |
| Fuzzy match | ✅ pg_trgm | limited | limited |
| Ops burden (MVP) | Low (one engine) | Low | Medium |

## Decision

**PostgreSQL** as the single primary database for the MVP, using `JSONB` for raw
payloads, **Row-Level Security** as an isolation backstop, `pgvector` for vectors,
and `pg_trgm` for fuzzy matching.

## Why

One engine covers relational + JSONB + RLS + vectors + fuzzy, minimizing ops for
a small team while giving us the strongest tenant-isolation primitive (RLS). It
defers the cost/complexity of a dedicated vector DB until workload justifies it
(see ADR-002).

## Trade-offs

- pgvector is excellent to a point but not as fast/scalable as a dedicated vector
  DB at very high vector counts/QPS → we plan to migrate vectors out (ADR-002).
- A single database is a shared failure/scaling domain → mitigated by read
  replicas (Stage 2) and later sharding by `organization_id` (Stage 4).

## Migration path

- Vectors: `VectorStore` interface lets us move pgvector → Qdrant with no app
  logic change (ADR-002).
- Scale: read replicas → partitioning → shard by `organization_id`. Keeping all
  tenant tables keyed by `organization_id` from day one makes sharding tractable.
- Search: heavy lexical/faceted search moves to OpenSearch behind a search
  interface when needed.
