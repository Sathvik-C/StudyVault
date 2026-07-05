-- Migration: Add supabase_key column to attachments for cloud storage
-- Run once against your Neon DB
ALTER TABLE attachments ADD COLUMN IF NOT EXISTS supabase_key TEXT;
CREATE INDEX IF NOT EXISTS idx_attachments_supabase_key ON attachments(supabase_key);
