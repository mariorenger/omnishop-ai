-- 011: session revocation. Tokens issued before this instant are rejected.
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS tokens_valid_after timestamptz;
