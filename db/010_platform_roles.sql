-- 010: platform-level roles (control plane, above tenant RBAC).
--   'admin'   -> full control of every platform config + tenants
--   'manager' -> read-only: stats, tenant list, report export; NO edits
--   NULL      -> ordinary tenant user
-- password_hash becomes optional so OAuth-only accounts (no local password) work.
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS platform_role text;
ALTER TABLE app_user ALTER COLUMN password_hash DROP NOT NULL;

-- Backfill from the legacy boolean and keep them consistent.
UPDATE app_user SET platform_role='admin' WHERE is_platform_admin AND platform_role IS NULL;
