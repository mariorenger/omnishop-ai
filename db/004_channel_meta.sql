-- OmniShop AI — resolve a Meta channel by Facebook Page / IG id for inbound
-- webhooks (no org context yet; SECURITY DEFINER bypasses RLS). Idempotent.

CREATE OR REPLACE FUNCTION resolve_channel_by_meta(pid text)
RETURNS TABLE(channel_id uuid, organization_id uuid, shop_id uuid, status text, config jsonb, credentials_enc bytea)
LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$
  SELECT id, organization_id, shop_id, status, config, credentials_enc
  FROM channel
  WHERE kind IN ('messenger','instagram') AND config->>'page_id' = pid
  LIMIT 1;
$$;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='omni_app') THEN
    GRANT EXECUTE ON FUNCTION resolve_channel_by_meta(text) TO omni_app;
  END IF;
END$$;
