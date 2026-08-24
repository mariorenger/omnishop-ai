-- OmniShop AI — per-bot data scoping + file storage. Idempotent.

-- A document/product/chunk may belong to a specific bot (else shop-wide = NULL).
ALTER TABLE document ADD COLUMN IF NOT EXISTS bot_id uuid REFERENCES bot(id) ON DELETE SET NULL;
ALTER TABLE chunk    ADD COLUMN IF NOT EXISTS bot_id uuid REFERENCES bot(id) ON DELETE SET NULL;
ALTER TABLE product  ADD COLUMN IF NOT EXISTS bot_id uuid REFERENCES bot(id) ON DELETE SET NULL;

-- Small file store (avatars/logos). Served publicly by unguessable id (no RLS);
-- org_id kept for accounting/cleanup.
CREATE TABLE IF NOT EXISTS file_asset (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid,
  mime            text NOT NULL,
  bytes           bytea NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='omni_app') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON file_asset TO omni_app;
  END IF;
END$$;
