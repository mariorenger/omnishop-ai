-- OmniShop AI — subscription renewal date. Paid plans get a 30-day term set on
-- activation; the tenant is reminded to renew as it approaches. Free plans leave
-- it NULL (never expire). Idempotent.

ALTER TABLE subscription ADD COLUMN IF NOT EXISTS current_period_end timestamptz;
