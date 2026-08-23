# ADR-008: Multi-tenancy & tenant isolation

- **Status:** Accepted (Milestone 0)
- **Date:** 2026-08-23
- **Deciders:** CTO / Security Engineer

## Context

This is a commercial multi-tenant SaaS. A cross-tenant data leak — one merchant
reading another's conversations, products, or knowledge — is an existential
failure. We need isolation that is strong, affordable at 100k tenants, and
defensible in depth (a single missed `WHERE` clause must not leak data). We also
need it to hold across three surfaces: relational data, vector/search, and the
LLM's tool access.

## Options (isolation model)

- **Shared schema, `organization_id` column** (row-level) — cheapest, scales to
  many tenants; relies on query discipline.
- **Schema-per-tenant** — stronger separation; painful at thousands of tenants
  (migrations, connections).
- **Database-per-tenant** — strongest; unaffordable at our scale; reserved for
  rare enterprise isolation needs.

## Decision

**Shared-schema, row-level multi-tenancy keyed by `organization_id`**, hardened
with **three enforced layers**:

1. **Logical (application):** all tenant data flows through a shared data-access
   layer that **requires** an `organization_id` (and `shop_id` where relevant).
   Unscoped tenant queries are rejected in code review and by lint/tests.
2. **Database (defense-in-depth):** PostgreSQL **Row-Level Security** policies on
   tenant tables, keyed to a per-request `SET app.current_org = <id>`. Even if a
   query forgets the filter, RLS blocks cross-tenant rows.
3. **Vector/search:** every embedding/search document carries `organization_id` /
   `shop_id` / `knowledge_base_id`; retrieval **always** applies the metadata
   filter. No global `SELECT * FROM documents`-style access exists.

**LLM boundary:** the model reaches tenant data **only** through tenant-scoped
tools that themselves go through layers 1–3. The model never receives a raw DB
handle or another tenant's context.

Optional **per-tenant/large-enterprise upgrade path:** dedicated schema or DB for
customers who contractually require it, behind the same data-access interface.

## Why

Row-level shared schema is the only model that stays affordable at 10k–100k
tenants, while RLS + mandatory metadata filtering give defense-in-depth so a
single coding mistake is not a breach. Keeping everything keyed by
`organization_id` from day one also makes later **sharding by org** (Stage 4)
straightforward.

## Trade-offs

- Shared schema means noisy-neighbor and blast-radius considerations → mitigated
  by quotas (ADR-004), read replicas, and later sharding.
- RLS adds a small per-query cost and requires disciplined session context
  management → accepted as the price of a hard safety net.

## Migration path

- Scale: partition hot tables by `organization_id`, then shard by org across
  Postgres instances (Stage 4) — enabled by the consistent key.
- Isolation upgrade: move a specific tenant to a dedicated schema/DB without
  application changes, because access is already interface-mediated and org-keyed.

## Verification (must exist before Milestone 1 ships tenant data)

- Automated tests that attempt cross-tenant reads and **assert they fail** at both
  the application layer and via RLS.
- A retrieval test proving Tenant A cannot get Tenant B's vectors/search docs.
- Audit-logged access to any admin "impersonation"/cross-tenant support tooling.
