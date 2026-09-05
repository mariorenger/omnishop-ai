-- OmniShop AI — SePay bank-transaction webhook log (platform-level, admin only).
-- SePay watches the merchant bank account and POSTs every incoming transfer to
-- /webhook/sepay-webhook; we record each one here (matched to an invoice or not)
-- so the admin can see incoming payments. No RLS: accessed via the admin pool.

CREATE TABLE IF NOT EXISTS sepay_transaction (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sepay_id         bigint,                       -- SePay's transaction id (for dedupe)
  gateway          text,                         -- bank name, e.g. Vietcombank
  account_no       text,
  amount           numeric(14,2) NOT NULL DEFAULT 0,
  content          text,                         -- transfer memo
  reference        text,                         -- bank reference code
  transfer_type    text,                         -- in | out
  matched_invoice  uuid,
  organization_id  uuid,
  status           text NOT NULL DEFAULT 'received',  -- received | activated | ignored
  created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS sepay_txn_sepay_id_uk ON sepay_transaction (sepay_id) WHERE sepay_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS sepay_txn_created_idx ON sepay_transaction (created_at DESC);
