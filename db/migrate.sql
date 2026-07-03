-- ============================================================
-- Migration: Add chats table for incremental ingestion
-- Run once against your chatfiles DB
-- ============================================================

CREATE TABLE IF NOT EXISTS chats (
    id              SERIAL PRIMARY KEY,
    chat_name       TEXT NOT NULL UNIQUE,   -- e.g. "Chemistry Group"
    source_type     TEXT,                   -- 'telegram' | 'whatsapp'
    last_msg_date   TIMESTAMPTZ,            -- timestamp of last processed message
    total_messages  INTEGER DEFAULT 0,
    total_files     INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Link files table to chats
ALTER TABLE files ADD COLUMN IF NOT EXISTS chat_id INTEGER REFERENCES chats(id);
ALTER TABLE files ADD COLUMN IF NOT EXISTS chat_name TEXT;

-- Add timestamp to messages if not already there
ALTER TABLE messages ADD COLUMN IF NOT EXISTS msg_timestamp TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_chats_name ON chats(chat_name);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(msg_timestamp);
CREATE INDEX IF NOT EXISTS idx_files_chat_id ON files(chat_id);
