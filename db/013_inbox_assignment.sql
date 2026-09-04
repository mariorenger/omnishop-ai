-- 013: shared inbox — assign a conversation to a staff member ("who is handling")
-- and record which staff user sent each agent reply.
ALTER TABLE conversation ADD COLUMN IF NOT EXISTS assigned_user_id uuid REFERENCES app_user(id) ON DELETE SET NULL;
ALTER TABLE message      ADD COLUMN IF NOT EXISTS sender_user_id   uuid REFERENCES app_user(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS conversation_assigned_idx ON conversation (assigned_user_id);
