-- OmniShop AI — Bots: a shop can have many bots (distinct persona/appearance);
-- each channel binds to one bot. Idempotent (initdb + startup).

CREATE TABLE IF NOT EXISTS bot (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  shop_id         uuid NOT NULL REFERENCES shop(id) ON DELETE CASCADE,
  name            text NOT NULL,
  persona         text,                              -- custom system prompt / instructions
  greeting        text DEFAULT 'Xin chào! Mình có thể giúp gì cho bạn?',
  avatar_url      text,
  accent_color    text NOT NULL DEFAULT '#6d7cff',
  config          jsonb NOT NULL DEFAULT '{}',
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS bot_org_idx ON bot (organization_id);
CREATE INDEX IF NOT EXISTS bot_shop_idx ON bot (shop_id);

ALTER TABLE channel ADD COLUMN IF NOT EXISTS bot_id uuid REFERENCES bot(id) ON DELETE SET NULL;

DO $$
BEGIN
  EXECUTE 'ALTER TABLE bot ENABLE ROW LEVEL SECURITY';
  EXECUTE 'ALTER TABLE bot FORCE ROW LEVEL SECURITY';
  EXECUTE 'DROP POLICY IF EXISTS tenant_isolation ON bot';
  EXECUTE $f$
    CREATE POLICY tenant_isolation ON bot
      USING (organization_id = current_setting('app.current_org', true)::uuid)
      WITH CHECK (organization_id = current_setting('app.current_org', true)::uuid)
  $f$;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='omni_app') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON bot TO omni_app;
  END IF;
END$$;

-- Resolver for the public widget: return the channel's bot too (return type
-- changed, so drop the old signature first).
DROP FUNCTION IF EXISTS resolve_channel_by_public_key(text);
CREATE OR REPLACE FUNCTION resolve_channel_by_public_key(pk text)
RETURNS TABLE(channel_id uuid, organization_id uuid, shop_id uuid, status text, config jsonb, bot_id uuid)
LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$
  SELECT id, organization_id, shop_id, status, config, bot_id
  FROM channel WHERE public_key = pk;
$$;
GRANT EXECUTE ON FUNCTION resolve_channel_by_public_key(text) TO omni_app;
