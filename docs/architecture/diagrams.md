# Architecture Diagrams

Diagrams referenced from [`overview.md`](overview.md), collected here as Mermaid
(renders on GitHub) plus ASCII fallbacks.

## 1. System context — the merchant journey

```mermaid
flowchart TD
    M[Merchant] --> S[Sign up]
    S --> P[Pay plan]
    P --> C[Connect shop]
    C --> FB[Facebook / Instagram]
    C --> TT[TikTok Shop]
    C --> SP[Shopee]
    FB --> IMP[Auto import products/knowledge]
    TT --> IMP
    SP --> IMP
    IMP --> RAG[Auto RAG setup]
    RAG --> BOT[AI chatbot enabled]
    BOT --> CM[Customer messages]
    CM --> AI[AI responds]
    AI --> HS{Confident?}
    HS -- yes --> DONE[Auto-resolved]
    HS -- no --> HUMAN[Human support inbox]
```

## 2. Control plane / data plane

```mermaid
flowchart TB
    subgraph CP[Control Plane - who/what/plan/quota/config/billing]
        T[Tenants/Orgs/Users]
        B[Billing/Plans/Entitlements]
        U[Usage/Quotas]
        CFG[Config/Integrations/AI settings]
        ADM[Admin/Support/Audit/Health]
    end
    subgraph DP[Data Plane - messages/AI/RAG/products/orders]
        WH[Webhook intake]
        ORCH[Conversation orchestrator]
        RENG[RAG engine]
        LLM[LLM / tools]
        PS[Product/Order services]
    end
    CFG -->|reads config/entitlements| ORCH
    U -->|quota checks| ORCH
    WH --> ORCH
    ORCH --> RENG --> LLM
    ORCH --> PS
    ORCH -->|meter usage/cost| U
```

## 3. Multi-tenancy entity model

```mermaid
erDiagram
    USER ||--o{ MEMBERSHIP : has
    ORGANIZATION ||--o{ MEMBERSHIP : has
    ORGANIZATION ||--o{ SHOP : owns
    SHOP ||--o{ CHANNEL : has
    CHANNEL ||--o{ CHATBOT : runs
    CHANNEL ||--o{ CONVERSATION : receives
    CONVERSATION ||--o{ MESSAGE : contains
    SHOP ||--o{ KNOWLEDGEBASE : has
    KNOWLEDGEBASE ||--o{ DOCUMENT : has
    DOCUMENT ||--o{ CHUNK : split_into
    SHOP ||--o{ PRODUCT : has
    PRODUCT ||--o{ VARIANT : has
```

## 4. Channel adapter boundary

```mermaid
flowchart LR
    subgraph PLAT[Platforms]
        FBp[Facebook]
        IGp[Instagram]
        TTp[TikTok Shop]
        SPp[Shopee]
    end
    subgraph ADAPT[ChannelProvider implementations]
        FBa[MetaMessengerProvider]
        IGa[InstagramProvider]
        TTa[TikTokShopProvider]
        SPa[ShopeeProvider]
        FKa[FakeProvider - tests]
    end
    FBp --> FBa
    IGp --> IGa
    TTp --> TTa
    SPp --> SPa
    FBa --> CANON[Canonical ChannelEvent / OutboundMessage]
    IGa --> CANON
    TTa --> CANON
    SPa --> CANON
    FKa --> CANON
    CANON --> CORE[Conversation / RAG / LLM core - platform-agnostic]
```

## 5. Conversation orchestration flow

```mermaid
flowchart TD
    IN[Incoming canonical message] --> CL[Classifier: intent/lang/urgency]
    CL --> RT{Route}
    RT -->|product| PSVC[Product Service - live inventory]
    RT -->|knowledge| RAGe[RAG engine - tenant-filtered]
    RT -->|order| OSVC[Order Service - status/track]
    RT -->|other| CHAT[Guardrailed LLM]
    PSVC --> CB[Context builder]
    RAGe --> CB
    OSVC --> CB
    CHAT --> CB
    CB --> LLMc[LLM call - entitlement+quota+window checked]
    LLMc --> POL[Response policy: guardrails/PII/confidence]
    POL -->|confident| SEND[Send via ChannelProvider]
    POL -->|not confident| HAND[Human handoff -> inbox]
    SEND --> MET[Meter usage & cost]
    HAND --> MET
```

## 6. RAG pipeline

```mermaid
flowchart LR
    Q[User query] --> N[Normalize] --> I[Intent]
    I --> PR[Product retrieval]
    I --> KR[Knowledge retrieval]
    PR --> H[Hybrid search BM25+vector +tenant filter]
    KR --> H
    H --> RR[Rerank] --> CTX[Context builder token-budget] --> L[LLM] --> R[Response]
```

## 7. Scaling stages

```mermaid
flowchart LR
    S1[Stage 1 ~10 tenants: single deployment, pgvector] --> S2[Stage 2 ~100: horizontal API+worker, read replica]
    S2 --> S3[Stage 3 ~1k: workers by class, Qdrant, Temporal]
    S3 --> S4[Stage 4 10k+: extract services at real bottlenecks, shard by org]
```
