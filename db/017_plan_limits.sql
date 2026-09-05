-- OmniShop AI — knowledge limits per plan: max number of documents and max size
-- per uploaded file. storage_mb already exists. Idempotent (merged into JSON).

UPDATE plan SET entitlements = entitlements || '{"knowledge_docs":20,"max_file_mb":5}'::jsonb   WHERE code='free';
UPDATE plan SET entitlements = entitlements || '{"knowledge_docs":200,"max_file_mb":25}'::jsonb  WHERE code='starter';
UPDATE plan SET entitlements = entitlements || '{"knowledge_docs":2000,"max_file_mb":50}'::jsonb WHERE code='growth';
UPDATE plan SET entitlements = entitlements || '{"knowledge_docs":2000,"max_file_mb":50}'::jsonb WHERE code='payg';

-- Any other/custom plans without the keys get safe defaults.
UPDATE plan SET entitlements = entitlements || '{"knowledge_docs":50,"max_file_mb":10}'::jsonb
  WHERE NOT (entitlements ? 'knowledge_docs');
