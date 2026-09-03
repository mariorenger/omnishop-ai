-- 012: knowledge documents can be deactivated without deleting. Inactive
-- documents (and their chunks) are excluded from retrieval but stay stored.
ALTER TABLE document ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true;
