# ADR-004: Billing — our entitlement engine + external payment provider(s)

- **Status:** Accepted (Milestone 0)
- **Date:** 2026-08-23
- **Deciders:** CTO / Product Engineer

## Context

Billing is core to the business, but it has two very different halves:

1. **Payment/subscription rails** (charge cards, handle renewals, invoices, tax,
   dunning) — a commodity best left to a payment provider.
2. **Entitlements, quota, usage, and plan → capability mapping** — this is our IP
   and must be enforced in the backend, never the frontend.

We also target Vietnam/SEA, where local payment methods (VNPay/MoMo/ZaloPay) and
tax handling matter, and where merchant-of-record (MoR) coverage varies.

## Options (payment rails)

- **Stripe** — you are Merchant of Record; lowest fees (~2.9%+30¢); you handle
  tax; weaker native VN methods.
- **Paddle / Lemon Squeezy** — MoR (~5%+50¢); handle global tax for you; **weaker
  SEA/Vietnam payment-method coverage** (verify).
- **Local (VNPay/MoMo/ZaloPay)** — needed for Vietnamese buyers; you handle more.
- **Polar / DodoPayments** — newer MoR with better India/APAC method coverage.

## Decision

- **BUILD** the entitlement/quota/usage engine ourselves (plans map to
  entitlements; backend enforces quotas). No `if (plan === 'premium')`
  hard-coding — an `EntitlementService` answers `canUse(feature)` /
  `withinQuota(metric)`.
- **EXTERNAL** payment rails behind a `BillingProvider` / `PaymentProvider`
  interface. Start with **Stripe** for cards/subscriptions; add **local VN
  methods** (VNPay/MoMo/ZaloPay) as adapters for the target market; keep **Paddle
  or a MoR** as an option if we want tax outsourced — pending SEA coverage check.

## Why

The commodity half (moving money) should never be reinvented; the differentiating
half (what a plan grants, quota enforcement, usage-based limits) must be ours so
we control the business model and can enforce it server-side. Wrapping payments
behind an interface lets us mix Stripe + local methods and switch MoR without
touching entitlement logic.

## Entitlement model (ours)

```
Plan ──> Entitlements (feature flags + numeric quotas)
User/Org ──> Subscription ──> resolved Entitlements
Data plane ──> EntitlementService.canUse(x) / withinQuota(metric) [server-side]
Usage records ──> quota counters ──> enforcement + billing-usage export
```

Example (illustrative): `Starter = 10,000 AI messages, 3 shops, 5 GB, 3 members`.
The backend enforces these; the frontend only *reflects* them.

## Trade-offs

- Building the entitlement engine is upfront work — but it is unavoidable IP.
- Mixing Stripe + local methods adds adapter complexity — accepted for the target
  market; hidden behind `PaymentProvider`.
- Being MoR with Stripe means we own tax/compliance; if that becomes heavy we can
  switch to a MoR provider behind the same interface.

## Migration path

`BillingProvider` (subscriptions/entitlement sync) and `PaymentProvider` (charge
methods) are separate interfaces. Swapping Stripe ↔ Paddle/Polar, or adding a
local method, is an adapter change. Entitlements live in our DB, independent of
any provider's product/price IDs (we map provider IDs → our plans).

> **Verify before Milestone 2:** current VN/SEA payment-method + tax coverage for
> each candidate provider; the 2026 landscape shifts (Lemon Squeezy is a Stripe
> company; new APAC-focused MoRs exist).
