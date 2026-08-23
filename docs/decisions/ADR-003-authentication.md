# ADR-003: Authentication / identity — Keycloak behind our boundary

- **Status:** Accepted (Milestone 0)
- **Date:** 2026-08-23
- **Deciders:** CTO / Security Engineer

## Context

We need auth (signup/login, social login, sessions, OIDC/OAuth2, password reset,
MFA) and a place to anchor **organizations** for multi-tenancy. We do **not** want
to hand-roll credential storage. Authorization/RBAC *policy* is business logic and
stays ours (see ADR-008); this ADR is only about the **identity provider**.

## Options

- **Keycloak** (Apache-2.0, 🟢) — mature IdP, OIDC/OAuth2/SAML, **Organizations**
  feature for multi-tenant B2B, self-hostable.
- **Supabase Auth / GoTrue** (Apache-2.0, 🟢) — lighter, developer-friendly.
- **Ory (Kratos/Hydra)** (Apache-2.0, 🟢) — headless, API-first, compose-your-own.
- **Zitadel** (AGPL-3.0, 🟡) — purpose-built multi-tenant, but AGPL.
- **Auth0 / Clerk / WorkOS** (external SaaS) — fast, but per-MAU cost + lock-in.

## Evaluation

| Criterion | Keycloak | Supabase Auth | Ory | Zitadel | Auth0/Clerk |
|---|---|---|---|---|---|
| License | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 🟢 Apache-2.0 | 🟡 AGPL | external |
| Multi-tenant orgs | ✅ built-in | build it | build it | ✅ built-in | ✅ (paid) |
| Self-host | ✅ | ✅ | ✅ | ✅ | ❌ |
| Cost at 100k users | infra only | infra only | infra only | infra only | high per-MAU |
| Lock-in | low (OIDC) | low | low | low | high |
| Ops burden | medium | low/medium | medium | medium | none |

## Decision

**Keycloak** (Apache-2.0), self-hosted, accessed through our own thin auth
boundary (OIDC). We use Keycloak for identity and its **Organizations** feature as
the technical anchor for tenants; **RBAC policy and entitlements remain our code.**

## Why

Permissive license (no AGPL/SaaS concerns), built-in multi-tenant Organizations,
standard OIDC so we are not locked in, and no per-MAU external cost as we scale to
many merchant end-users. Supabase Auth/Ory are fine fallbacks; Zitadel only if
legal accepts AGPL; external SaaS rejected on cost + lock-in at our user scale.

## Trade-offs

- Keycloak has real operational weight (JVM, realm config) → mitigated by
  treating it as infra and keeping our boundary thin.
- We own more of the org/tenant mapping than a turnkey SaaS would give us — but
  that mapping is core IP we want to own anyway.

## Migration path

Because we consume **standard OIDC** behind our own `AuthProvider` boundary,
swapping Keycloak for Ory/Supabase Auth/an external IdP is a configuration +
adapter change, not an application rewrite. User identities are keyed by a stable
`user_id` in our DB, decoupled from the IdP's internal IDs.
