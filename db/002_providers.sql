-- OmniShop AI — provider configuration (LLM / embedding / OCR).
-- Idempotent: run by initdb (fresh) AND at API startup (existing volumes).
--
-- Precedence: platform default -> tenant override (if policy allows) -> env/stub.
-- These tables are platform-managed (no RLS); access is gated in the app by role.

CREATE TABLE IF NOT EXISTS provider_config (
  scope       text PRIMARY KEY,   -- 'llm:platform' | 'llm:org:<uuid>' | 'embedding:platform' | 'ocr:platform' | 'ocr:org:<uuid>'
  provider    text NOT NULL,      -- anthropic | openai_compatible | gemini | local | tesseract | vlm | disabled | stub
  model       text,
  base_url    text,
  api_key_enc bytea,              -- encrypted at rest (never plaintext)
  extra       jsonb NOT NULL DEFAULT '{}',
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS platform_settings (
  id                int PRIMARY KEY DEFAULT 1,
  allow_tenant_llm  boolean NOT NULL DEFAULT true,
  allow_tenant_ocr  boolean NOT NULL DEFAULT true,
  CONSTRAINT platform_settings_singleton CHECK (id = 1)
);
INSERT INTO platform_settings (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- Grant the app role (safe to re-run).
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='omni_app') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON provider_config TO omni_app;
    GRANT SELECT, INSERT, UPDATE, DELETE ON platform_settings TO omni_app;
  END IF;
END$$;
