-- 014: store a human-friendly customer name on a conversation when the channel
-- provides it (Telegram from-name, WhatsApp profile name, …). Falls back to the
-- channel-scoped id when unknown.
ALTER TABLE conversation ADD COLUMN IF NOT EXISTS customer_name text;
