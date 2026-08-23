-- OmniShop AI — billing: invoices & payments. Idempotent (initdb + startup).
-- Tenant-scoped with RLS like other tenant tables (ADR-008).

CREATE TABLE IF NOT EXISTS invoice (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  plan_code       text NOT NULL,
  amount          numeric(12,2) NOT NULL DEFAULT 0,
  currency        text NOT NULL DEFAULT 'USD',
  status          text NOT NULL DEFAULT 'pending',  -- pending | paid | void
  provider        text NOT NULL DEFAULT 'manual',   -- manual | stripe | vnpay | momo
  external_ref    text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  paid_at         timestamptz
);
CREATE INDEX IF NOT EXISTS invoice_org_idx ON invoice (organization_id, created_at);

CREATE TABLE IF NOT EXISTS payment (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
  invoice_id      uuid REFERENCES invoice(id) ON DELETE SET NULL,
  amount          numeric(12,2) NOT NULL DEFAULT 0,
  currency        text NOT NULL DEFAULT 'USD',
  provider        text NOT NULL DEFAULT 'manual',
  status          text NOT NULL DEFAULT 'succeeded',
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS payment_org_idx ON payment (organization_id, created_at);

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['invoice','payment'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY;', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY;', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I;', t);
    EXECUTE format($f$
      CREATE POLICY tenant_isolation ON %I
        USING (organization_id = current_setting('app.current_org', true)::uuid)
        WITH CHECK (organization_id = current_setting('app.current_org', true)::uuid);
    $f$, t);
  END LOOP;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='omni_app') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON invoice TO omni_app;
    GRANT SELECT, INSERT, UPDATE, DELETE ON payment TO omni_app;
  END IF;
END$$;
