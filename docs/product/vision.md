# Product Vision — OmniShop AI

## 1. One sentence

A multi-tenant SaaS that lets any shop connect its sales/social-commerce
channels and get an AI RAG chatbot that answers customers automatically — with
human handoff — across every channel.

## 2. The promise

> **Pay → Connect → AI works.**

The merchant's entire required journey:

```
Sign up → Choose plan → Pay → Connect shop → Import products / knowledge
        → AI is configured automatically → Enable chatbot
        → Customers chat → AI responds (human handoff when needed)
```

The merchant must **never** need to understand: Elasticsearch/OpenSearch, vector
DBs, embeddings, LLMs, OAuth, webhooks, Docker, Kubernetes, API keys, prompt
engineering, or RAG. Those are our problem, hidden behind the product.

## 3. Who it is for

- **Primary:** small/medium shops and social-commerce sellers who sell through
  Facebook/Instagram/TikTok Shop/Shopee and are drowning in repetitive customer
  DMs ("còn hàng không?", "size M còn không?", "bao giờ giao?").
- **Secondary:** growing brands that want one AI-assisted inbox across channels
  and human agents for escalations.

Their pain is **volume of repetitive questions** and **fragmentation across
channels**. Our value is **automated, product-aware, channel-agnostic answers**.

## 4. What makes it different (the core differentiators — we build these)

1. **Product-aware AI, not just document RAG.** "Áo này còn size M không?" must
   query the merchant's **product/inventory data**, not a PDF vector store. The
   AI understands SKU / variant / price / stock.
2. **True omnichannel with a channel-adapter architecture.** New channels are
   added as adapters without touching the RAG/LLM/conversation core.
3. **Zero-config onboarding.** Connecting a shop auto-imports products and
   auto-configures retrieval — no dashboards full of knobs.
4. **Cost-aware from day one.** Every AI interaction is metered so we always
   know per-tenant COGS and gross margin (which customer is unprofitable).
5. **Operable support.** When a merchant says "my TikTok isn't working," support
   can diagnose from tenant/channel health dashboards without SSH-ing a server.

## 5. Target channels

**Phase 1 (research now):** Facebook, Instagram, TikTok Shop, Shopee.
**Later:** Website widget, WhatsApp, Zalo, Telegram, Lazada, Shopify.

The architecture must **not** depend on the first four. Everything channel-specific
lives behind a `ChannelProvider` adapter (see the architecture overview). See the
[platform capability matrix](../platform/platform-capability-matrix.md) for what
each channel's API can actually do — it drives the order in which we ship them.

## 6. The merchant experience we are engineering toward

```
                        MERCHANT
                           │
                        SIGN UP
                           │
                        PAY PLAN
                           │
                      CONNECT SHOP
              ┌────────────┼────────────┐
           Facebook     TikTok        Shopee
           Instagram     Shop
              └────────────┼────────────┘
                           │
                      AUTO IMPORT (products, catalog)
                           │
                     AUTO RAG SETUP
                           │
                      AI CHATBOT
                           │
                   CUSTOMER MESSAGES
                           │
                     AI RESPONDS
                  ┌────────┴────────┐
               AI SUPPORT        HUMAN SUPPORT
```

## 7. Non-goals (for the MVP)

- Not a general-purpose chatbot builder / no-code workflow canvas.
- Not a full CRM or marketing-automation suite.
- Not a marketplace/storefront — we integrate with the merchant's existing shops.
- No microservices, no Kubernetes, no multi-region until real workload demands it.

## 8. Success signals

- Time-from-signup-to-first-AI-answer measured in **minutes**, not days.
- % of customer messages resolved by AI without human handoff.
- Positive **per-tenant gross margin** visible in the admin control plane.
- Adding channel #5 requires **one adapter**, zero changes to RAG/LLM/conversation.
