-- 009: admin-editable cost rates (USD per 1M tokens). NULL = fall back to env.
ALTER TABLE platform_settings ADD COLUMN IF NOT EXISTS cost_input_per_m     numeric;
ALTER TABLE platform_settings ADD COLUMN IF NOT EXISTS cost_output_per_m    numeric;
ALTER TABLE platform_settings ADD COLUMN IF NOT EXISTS cost_embedding_per_m numeric;
