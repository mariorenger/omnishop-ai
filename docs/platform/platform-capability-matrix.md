# Platform Capability Matrix

**Purpose:** decide which channel to ship first, and what each channel's API can
*actually* do — so we never promise a feature a platform's API does not support.

> **Verified as-of August 2026** from public/official docs and vendor guides.
> Platform APIs change often and access is frequently **gated by app review,
> region, and partner status**. A ✅ means the capability is documented; it does
> **not** guarantee your app will be *granted* it without review. **Re-verify
> against the official developer portal before implementation**, and treat
> anything marked ⚠️ as "must confirm during Milestone 4 spike."

## Legend

- ✅ Documented & generally available (subject to app review)
- ⚠️ Exists but **gated** (partner approval / region / separate scope) or
  under-documented publicly
- ❌ Not available / not supported
- 🔑 Requires OAuth app review / business verification

## Matrix

| Capability | Facebook (Messenger) | Instagram (Messaging) | TikTok Shop | Shopee Open Platform |
|---|---|---|---|---|
| OAuth / authorization | ✅ Meta Login 🔑 | ✅ via Meta 🔑 | ✅ OAuth 2.0 (App Key/Secret/Service ID) 🔑 | ✅ partner OAuth 🔑 |
| Connect merchant account | ✅ Page connect | ✅ IG professional account | ✅ shop authorization | ✅ shop authorization |
| Receive customer message (webhook) | ✅ | ✅ | ⚠️ IM/customer-service API is **partner-gated**, not clearly public | ⚠️ Chat API exists but **separate scope + partner approval** |
| Send reply message | ✅ (24h window + tags) | ✅ (`instagram_business_manage_messages`) | ⚠️ gated (see above) | ⚠️ via Chat API (gated) |
| Messaging window / policy limits | ✅ **24h window**; outside → tags / One-Time Notifications / Sponsored | ✅ similar policy to Messenger | ⚠️ platform-defined; confirm | ⚠️ platform-defined; confirm |
| Product catalog sync | ⚠️ via Commerce/Catalog API | ⚠️ via Commerce/Catalog API | ✅ product listing API | ✅ product API |
| Inventory / stock | ⚠️ catalog-level | ⚠️ catalog-level | ✅ inventory API | ✅ inventory API |
| Price / variant | ⚠️ catalog-level | ⚠️ catalog-level | ✅ | ✅ |
| Orders (read) | ❌ (not a marketplace) | ❌ | ✅ order API | ✅ order API |
| Order status / fulfillment | ❌ | ❌ | ✅ | ✅ |
| Customer identity | ✅ PSID (page-scoped) | ✅ IGSID | ⚠️ order-scoped buyer info | ⚠️ order-scoped buyer info |
| Webhooks (events) | ✅ | ✅ | ✅ (shop/order/product contexts) | ✅ (order/product/chat) |
| Token lifetime | Long-lived page token ≈ 60 days; refresh needed | Long-lived → page token no expiry; short-lived → 1h | OAuth tokens with refresh | OAuth tokens with refresh |
| Rate limits | ✅ 200 × engaged users / 24h (Messenger) | ✅ Meta platform limits | ✅ per-app/per-shop limits | ✅ per-partner/per-shop limits |
| App review / verification | 🔑 required for production | 🔑 required for production | 🔑 required; partner center | 🔑 required; partner approval |
| Sandbox / test | ✅ (test users, ≤25 pre-review) | ✅ (≤25 test users pre-review) | ✅ (dev app) | ✅ (sandbox) |
| Regional restrictions | Global | Global | ⚠️ region-specific programs/features | ⚠️ strong in SEA; per-region rules |

## Key implications for our roadmap

### 1. The chatbot's value is in *messaging* — Meta wins the messaging race

Facebook Messenger and Instagram have the **most mature, best-documented
messaging APIs** (real-time webhooks for inbound DMs, a clear send API, a
well-known 24-hour window policy, tags for out-of-window sends). Since our core
product is *an AI that answers customer messages*, **Meta is the right first real
channel (Milestone 4).**

### 2. TikTok Shop & Shopee are commerce-first, messaging-gated

Both expose strong **product / inventory / order** APIs — excellent for
*product-aware AI* (Milestone 7) — but their **buyer-messaging APIs are gated**:
Shopee's Chat API needs a separate scope and partner approval and is not granted
to every partner; TikTok Shop's customer-service/IM API is partner-gated and not
clearly public. **Do not assume we can auto-reply on these channels until a spike
confirms access.**

Therefore:

- **Ship Meta first** for the messaging/chatbot value.
- **Integrate TikTok Shop & Shopee next for product/order sync** (feeding the
  product-aware AI), and enable auto-messaging on them only after confirming
  partner-gated chat access per merchant/region.

### 3. The 24-hour window is a product constraint, not a bug

On Meta, we can only freely reply within 24h of the customer's last message.
Outside it, we must use message tags / One-Time Notifications / Sponsored
Messages. The **conversation orchestrator must be window-aware** and the product
must not promise "AI messages your customers anytime."

### 4. Token lifecycle is an operational burden we must own

Page/shop tokens expire and must be refreshed; failures must surface as
**channel health** states (`TOKEN_EXPIRED`, `WEBHOOK_FAILED`, `RATE_LIMITED`, …)
and trigger merchant notifications — this is why channel-connection state is
**core IP we build**, not something we let an OSS inbox own.

## Recommended channel order

| Order | Channel | Why | Milestone |
|---|---|---|---|
| 1 | **Facebook Messenger** | Most mature messaging API; core chatbot value | 4 |
| 2 | **Instagram** | Same Meta stack; incremental cost | 4/6 |
| 3 | **Shopee** (product/order first) | Rich commerce APIs for product-aware AI; chat gated | 7 |
| 4 | **TikTok Shop** (product/order first) | Rich commerce APIs; chat gated | 7 |
| later | Website widget, WhatsApp, Zalo, Telegram, Lazada, Shopify | Adapter-by-adapter | 8+ |

## Sources (verified Aug 2026)

- Messenger send API, 24h window, tokens, rate limits: <https://developers.facebook.com/docs/messenger-platform/reference/send-api/> ; <https://developers.facebook.com/documentation/business-messaging/messenger-platform/send-messages>
- Instagram Messaging API (permissions, tokens, webhooks, app review): <https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/messaging-api/>
- TikTok Shop Partner API (OAuth, webhooks, product/order): <https://partner.tiktokshop.com/docv2/page/tts-webhooks-overview>
- Shopee Open Platform (Chat API scope/approval, product/order): <https://api2cart.com/api-technology/shopee-api/>

> **Milestone 4 spike checklist:** confirm (a) exact permissions/app-review path
> for Messenger + IG, (b) Shopee Chat API grant process, (c) TikTok Shop IM API
> availability in target regions, before committing the roadmap dates.
