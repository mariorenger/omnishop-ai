-- OmniShop AI — initial schema (Milestone 1 foundation + core loop)
-- PostgreSQL 16 + pgvector. Multi-tenant, row-level-security hardened (ADR-008).
--
-- Isolation model (ADR-008):
--   1. Application always scopes by organization_id.
--   2. Row-Level Security is the defense-in-depth backstop: tenant tables carry
--      a policy keyed to current_setting('app.current_org'). The app sets this
--      per transaction (SET LOCAL app.current_org = '<uuid>').
--   3. Vector/search rows carry organization_id/shop_id/knowledge_base_id and
--      retrieval always filters by them.
--
-- Embedding dimension is fixed (see EMBEDDING_DIM in .env / config). Default 384.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- fuzzy text match

-- A dedicated app role used by the API/worker. RLS applies to non-superusers.
-- (In docker-compose the DB superuser is 'omni'; we also create a restricted
--  role the app connects as so RLS is actually enforced.)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omni_app') THEN
    CREATE ROLE omni_app LOGIN PASSWORD 'omni_app';
  END IF;
END$$;

-- ---------------------------------------------------------------------------
-- Control plane: identity, tenancy, billing
-- ---------------------------------------------------------------------------

CREATE TABLE app_user (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email         text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  full_name     text,
  is_platform_admin boolean NOT NULL DEFAULT false,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE organization (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name          text NOT NULL,
  status        text NOT NULL DEFAULT 'active',   -- active | suspended
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE membership (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  user_id         uuid NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  role            text NOT NULL DEFAULT 'owner',  -- owner | admin | agent | viewer
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (organization_id, user_id)
);

-- Plans & entitlements (ADR-004). Entitlements are JSONB: feature flags + quotas.
CREATE TABLE plan (
  code          text PRIMARY KEY,                 -- free | starter | growth
  name          text NOT NULL,
  price_month   numeric(10,2) NOT NULL DEFAULT 0,
  entitlements  jsonb NOT NULL DEFAULT '{}',      -- {"ai_messages_month":1000,"shops":1,...}
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE subscription (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  plan_code       text NOT NULL REFERENCES plan(code),
  status          text NOT NULL DEFAULT 'active',  -- active | past_due | canceled
  period_start    timestamptz NOT NULL DEFAULT date_trunc('month', now()),
  provider        text NOT NULL DEFAULT 'manual',  -- manual | stripe | vnpay ...
  provider_ref    text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (organization_id)
);

-- ---------------------------------------------------------------------------
-- Merchant resources: shops, channels
-- ---------------------------------------------------------------------------

CREATE TABLE shop (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  name            text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON shop (organization_id);

-- A connected channel (website widget now; meta/tiktok/shopee later behind ChannelProvider).
CREATE TABLE channel (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  shop_id         uuid NOT NULL REFERENCES shop(id) ON DELETE CASCADE,
  kind            text NOT NULL,                  -- website | messenger | instagram | tiktok | shopee
  name            text NOT NULL,
  public_key      text UNIQUE,                    -- for website widget embedding
  status          text NOT NULL DEFAULT 'connected', -- connected|degraded|token_expired|webhook_failed|rate_limited|disconnected
  -- OAuth/creds are encrypted at rest by the app before storage (never plaintext, R-17).
  credentials_enc bytea,
  config          jsonb NOT NULL DEFAULT '{}',    -- greeting, handoff rules, etc.
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON channel (organization_id);
CREATE INDEX ON channel (shop_id);

-- ---------------------------------------------------------------------------
-- Knowledge base (RAG) + products (product-aware AI)
-- ---------------------------------------------------------------------------

CREATE TABLE knowledge_base (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  shop_id         uuid NOT NULL REFERENCES shop(id) ON DELETE CASCADE,
  name            text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON knowledge_base (organization_id);

CREATE TABLE document (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id   uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  knowledge_base_id uuid NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
  title             text NOT NULL,
  source            text,
  status            text NOT NULL DEFAULT 'pending', -- pending | processing | ready | error
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON document (organization_id);
CREATE INDEX ON document (knowledge_base_id);

CREATE TABLE chunk (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id   uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  knowledge_base_id uuid NOT NULL REFERENCES knowledge_base(id) ON DELETE CASCADE,
  document_id       uuid NOT NULL REFERENCES document(id) ON DELETE CASCADE,
  ordinal           int NOT NULL,
  content           text NOT NULL,
  embedding         vector(384),
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON chunk (organization_id);
CREATE INDEX ON chunk (document_id);
-- Cosine distance index for retrieval (built lazily; fine for MVP scale).
CREATE INDEX chunk_embedding_idx ON chunk USING hnsw (embedding vector_cosine_ops);

CREATE TABLE product (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  shop_id         uuid NOT NULL REFERENCES shop(id) ON DELETE CASCADE,
  external_id     text,                           -- id on the source channel, if synced
  sku             text,
  name            text NOT NULL,
  description     text,
  price           numeric(12,2),
  currency        text NOT NULL DEFAULT 'VND',
  attributes      jsonb NOT NULL DEFAULT '{}',    -- {color, material, ...}
  embedding       vector(384),
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON product (organization_id);
CREATE INDEX ON product (shop_id);
CREATE INDEX product_name_trgm ON product USING gin (name gin_trgm_ops);
CREATE INDEX product_embedding_idx ON product USING hnsw (embedding vector_cosine_ops);

-- Product variants (size/color with own stock/price) — product-aware answers.
CREATE TABLE product_variant (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  product_id      uuid NOT NULL REFERENCES product(id) ON DELETE CASCADE,
  name            text NOT NULL,                  -- e.g. "Size M / Đen"
  sku             text,
  price           numeric(12,2),
  stock           int NOT NULL DEFAULT 0,
  attributes      jsonb NOT NULL DEFAULT '{}',
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON product_variant (organization_id);
CREATE INDEX ON product_variant (product_id);

-- ---------------------------------------------------------------------------
-- Conversations & messages
-- ---------------------------------------------------------------------------

CREATE TABLE conversation (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  shop_id         uuid NOT NULL REFERENCES shop(id) ON DELETE CASCADE,
  channel_id      uuid NOT NULL REFERENCES channel(id) ON DELETE CASCADE,
  customer_ref    text NOT NULL,                  -- channel-scoped customer id (e.g. widget session)
  status          text NOT NULL DEFAULT 'ai',     -- ai | needs_human | human | closed
  created_at      timestamptz NOT NULL DEFAULT now(),
  last_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON conversation (organization_id);
CREATE INDEX ON conversation (channel_id, customer_ref);

CREATE TABLE message (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  conversation_id uuid NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  role            text NOT NULL,                  -- customer | ai | agent | system
  content         text NOT NULL,
  meta            jsonb NOT NULL DEFAULT '{}',    -- intent, retrieved ids, confidence
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON message (organization_id);
CREATE INDEX ON message (conversation_id, created_at);

-- ---------------------------------------------------------------------------
-- Usage metering & cost accounting (ADR-007, cost model)
-- ---------------------------------------------------------------------------

CREATE TABLE usage_event (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id  uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  shop_id          uuid REFERENCES shop(id) ON DELETE SET NULL,
  channel_id       uuid REFERENCES channel(id) ON DELETE SET NULL,
  conversation_id  uuid REFERENCES conversation(id) ON DELETE SET NULL,
  kind             text NOT NULL,                 -- ai_message | embedding
  llm_model        text,
  input_tokens     int NOT NULL DEFAULT 0,
  output_tokens    int NOT NULL DEFAULT 0,
  embedding_tokens int NOT NULL DEFAULT 0,
  retrieval_count  int NOT NULL DEFAULT 0,
  latency_ms       int NOT NULL DEFAULT 0,
  estimated_cost   numeric(12,6) NOT NULL DEFAULT 0,
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON usage_event (organization_id, created_at);

-- ---------------------------------------------------------------------------
-- Async jobs (QueueProvider is Valkey-backed; this table is a durable mirror
-- for observability/retry). Worker consumes from Valkey; records land here.
-- ---------------------------------------------------------------------------

CREATE TABLE job (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid,
  kind            text NOT NULL,                  -- embed_document | embed_product
  payload         jsonb NOT NULL DEFAULT '{}',
  status          text NOT NULL DEFAULT 'queued', -- queued | running | done | error
  error           text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON job (status);

-- ---------------------------------------------------------------------------
-- Audit log (every privileged/admin action — R-06/R-17, ADR-008)
-- ---------------------------------------------------------------------------

CREATE TABLE audit_log (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid,
  actor_user_id   uuid,
  action          text NOT NULL,
  target          text,
  detail          jsonb NOT NULL DEFAULT '{}',
  correlation_id  text,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON audit_log (organization_id, created_at);

-- ---------------------------------------------------------------------------
-- Row-Level Security (defense-in-depth). Enabled on tenant tables; policy keyed
-- to app.current_org set by the app per transaction. Superuser bypasses RLS,
-- so the app connects as omni_app (non-superuser) for tenant-scoped work.
-- ---------------------------------------------------------------------------

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'shop','channel','knowledge_base','document','chunk','product',
    'product_variant','conversation','message','usage_event','membership'
  ] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY;', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY;', t);
    EXECUTE format($f$
      CREATE POLICY tenant_isolation ON %I
        USING (organization_id = current_setting('app.current_org', true)::uuid)
        WITH CHECK (organization_id = current_setting('app.current_org', true)::uuid);
    $f$, t);
  END LOOP;
END$$;

-- Grant the app role access (RLS still restricts rows).
GRANT USAGE ON SCHEMA public TO omni_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO omni_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO omni_app;

-- ---------------------------------------------------------------------------
-- SECURITY DEFINER helper: resolve a website widget channel by its public key.
-- The public widget endpoint has no org context yet, and `channel` is RLS-locked,
-- so a normal omni_app query would return zero rows. This function is owned by
-- the superuser that runs this script and therefore bypasses RLS to return just
-- the routing fields needed to establish tenant context. It exposes only the
-- lookup by public_key (no enumeration).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION resolve_channel_by_public_key(pk text)
RETURNS TABLE(channel_id uuid, organization_id uuid, shop_id uuid, status text, config jsonb)
LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$
  SELECT id, organization_id, shop_id, status, config
  FROM channel WHERE public_key = pk;
$$;
GRANT EXECUTE ON FUNCTION resolve_channel_by_public_key(text) TO omni_app;

-- ---------------------------------------------------------------------------
-- Seed plans
-- ---------------------------------------------------------------------------

INSERT INTO plan (code, name, price_month, entitlements) VALUES
  ('free',    'Free',    0,   '{"ai_messages_month": 200,   "shops": 1, "channels": 1, "storage_mb": 100,  "human_handoff": true, "channels_allowed": ["website"]}'),
  ('starter', 'Starter', 49,  '{"ai_messages_month": 10000, "shops": 3, "channels": 5, "storage_mb": 5000, "human_handoff": true, "channels_allowed": ["website","messenger","instagram","telegram","zalo","whatsapp"]}'),
  ('growth',  'Growth',  149, '{"ai_messages_month": 50000, "shops": 10,"channels": 20,"storage_mb": 20000,"human_handoff": true, "channels_allowed": ["website","messenger","instagram","telegram","zalo","whatsapp","tiktok","shopee"]}')
ON CONFLICT (code) DO NOTHING;
