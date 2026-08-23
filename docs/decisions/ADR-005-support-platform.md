# ADR-005: Support / unified inbox — Chatwoot MIT core behind `SupportProvider`

- **Status:** Proposed (decision confirmed at Milestone 8 spike)
- **Date:** 2026-08-23
- **Deciders:** CTO / Product Engineer

## Context

When AI can't or shouldn't answer, a human agent needs a **unified inbox**:
conversation UI, agent assignment, labels, notes, customer profile, team
management. Building this well is months of work. Chatwoot already does it and is
MIT-licensed at its core — but we must not let its data model own our tenancy, AI,
or billing.

## Options

- **Chatwoot** (MIT core + commercial `enterprise/` dir, 🟢/🟡) — mature
  omnichannel inbox; FB/IG channels; API for integration.
- **Build our own inbox** — full control; large effort.
- **Other OSS helpdesks** (e.g. Zammad/FreeScout) — evaluate if Chatwoot misfits.

## Evaluation

| Criterion | Chatwoot (MIT core) | Build our own |
|---|---|---|
| License | 🟢 MIT core (branding/SSO are EE, paid) | n/a |
| Time-to-value | weeks (integrate) | months (build) |
| Inbox UX maturity | high | starts at zero |
| Multi-tenancy fit | must map to our orgs | native to our model |
| Custom branding | 🟡 EE feature | free |
| Lock-in risk | medium (its schema/API) | none |
| AI integration | via API/agent-bot | native |

## Decision

**Reuse Chatwoot's MIT core** behind a `SupportProvider` interface for the human
inbox — but keep **tenancy, AI orchestration, billing, and channel-connection
state in our control plane**. The AI is the "first responder"; handoff pushes the
conversation into the inbox. **Re-confirm at the Milestone 8 spike** whether
Chatwoot's multi-tenancy, API limits, branding constraints (EE), and scaling fit;
if not, **build a focused inbox** on our existing conversation model.

## Why

Don't spend months rebuilding a solved inbox UI. But don't let Chatwoot's schema
become our system of record for tenants/conversations — we already own a canonical
conversation model (channel-agnostic), so Chatwoot is a *view/agent surface*, not
the source of truth. The `SupportProvider` boundary keeps "build our own" open.

## Trade-offs

- Integrating Chatwoot means reconciling two conversation representations (ours =
  source of truth; Chatwoot = agent surface) → sync complexity; mitigated by
  treating Chatwoot as downstream.
- Custom branding / SSO are Chatwoot EE (paid) → budget for it or accept default
  branding for the inbox, or build our own if branding is critical.
- API/rate limits and multi-tenant mapping must be validated (Milestone 8 spike).

## Migration path

Because handoff and inbox access go through `SupportProvider`, and our canonical
conversation model is the source of truth, we can replace Chatwoot with our own
inbox (or another OSS helpdesk) without changing the orchestrator or losing data.
