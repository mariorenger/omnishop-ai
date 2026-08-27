-- 008: flexible pricing (BYOK / managed / pay-as-you-go) + per-end-user metering.
-- Idempotent (runs at startup).

-- Attribute each metered AI message to the end customer for per-user token views.
ALTER TABLE usage_event ADD COLUMN IF NOT EXISTS customer_ref text;

-- Plan entitlements gain:
--   llm_mode      : 'byok'    -> tenant brings their own OpenAI/Gemini/Claude key
--                   'managed' -> platform provides the model (price includes tokens)
--   billing_mode  : 'subscription' | 'payg'
--   ai_tokens_month : included AI tokens/month (managed subscription)
--   overage_per_1k  : USD per 1k tokens beyond the allowance (0 = hard cap)
--   payg_per_1k     : USD per 1k tokens (pay-as-you-go)
-- ai_messages_month stays as a simple/fair-use cap (0 = unlimited).

-- Free: bring-your-own key, small monthly message cap.
UPDATE plan SET name='Free', price_month=0, entitlements = entitlements ||
  '{"llm_mode":"byok","billing_mode":"subscription","ai_tokens_month":0,"overage_per_1k":0,"payg_per_1k":0,"ai_messages_month":200}'::jsonb
  WHERE code='free';

-- Thuê bot (BYOK): cheaper — pay for software/RAG/channels, use your own AI key.
UPDATE plan SET name='Thuê bot', price_month=19, entitlements = entitlements ||
  '{"llm_mode":"byok","billing_mode":"subscription","ai_tokens_month":0,"overage_per_1k":0,"payg_per_1k":0,"ai_messages_month":20000}'::jsonb
  WHERE code='starter';

-- Trọn gói AI (managed): higher — platform supplies the model, tokens included.
UPDATE plan SET name='Trọn gói AI', price_month=99, entitlements = entitlements ||
  '{"llm_mode":"managed","billing_mode":"subscription","ai_tokens_month":5000000,"overage_per_1k":0.02,"payg_per_1k":0,"ai_messages_month":0}'::jsonb
  WHERE code='growth';

-- Trả theo dùng (PAYG, managed): no monthly commit, billed per token used.
INSERT INTO plan (code, name, price_month, entitlements) VALUES
  ('payg', 'Trả theo dùng', 0,
   '{"ai_messages_month":0,"shops":10,"channels":20,"storage_mb":20000,"human_handoff":true,
     "channels_allowed":["website","messenger","instagram","telegram","zalo","whatsapp","tiktok","shopee"],
     "llm_mode":"managed","billing_mode":"payg","ai_tokens_month":0,"overage_per_1k":0,"payg_per_1k":0.03}'::jsonb)
ON CONFLICT (code) DO UPDATE SET name=EXCLUDED.name, price_month=EXCLUDED.price_month,
  entitlements=EXCLUDED.entitlements;
