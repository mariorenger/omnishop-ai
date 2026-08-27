-- 007: async ingestion metadata on document + platform branding config.
-- Idempotent (runs at startup via run_migrations).

-- Document ingestion status detail. status flow: queued -> processing -> ready | error
ALTER TABLE document ADD COLUMN IF NOT EXISTS error         text;
ALTER TABLE document ADD COLUMN IF NOT EXISTS char_count    integer;
ALTER TABLE document ADD COLUMN IF NOT EXISTS mime          text;
ALTER TABLE document ADD COLUMN IF NOT EXISTS bytes         integer;
ALTER TABLE document ADD COLUMN IF NOT EXISTS file_asset_id uuid;

-- Platform branding (config-driven UI). One row (id=1) mirrors platform_settings.
ALTER TABLE platform_settings ADD COLUMN IF NOT EXISTS app_name     text;
ALTER TABLE platform_settings ADD COLUMN IF NOT EXISTS logo_asset   uuid;
ALTER TABLE platform_settings ADD COLUMN IF NOT EXISTS accent_color text;

-- file_asset may hold non-image documents too (raw bytes for re-processing).
-- No schema change needed (mime/bytes are already generic).
