import logging
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)

BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id              SERIAL PRIMARY KEY,
    file_id         TEXT,
    filename        TEXT,
    size            BIGINT,
    uploaded_at     TIMESTAMPTZ DEFAULT NOW(),
    file_hash       TEXT UNIQUE,
    chat_id         INTEGER,
    chat_name       TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    file_id         INTEGER REFERENCES files(id),
    role            TEXT,
    content         TEXT,
    message_order   INTEGER,
    is_important    BOOLEAN DEFAULT FALSE,
    msg_timestamp   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS attachments (
    id                      SERIAL PRIMARY KEY,
    file_id                 INTEGER REFERENCES files(id),
    message_id              INTEGER REFERENCES messages(id),
    original_name           TEXT,
    storage_path            TEXT,
    category                TEXT,
    subject                 TEXT,
    subcategory             TEXT,
    classification_method   TEXT,
    file_hash               TEXT UNIQUE,
    size_bytes              BIGINT,
    note                    TEXT,
    tags                    TEXT[]
);

CREATE TABLE IF NOT EXISTS subjects (
    id          SERIAL PRIMARY KEY,
    file_id     INTEGER REFERENCES files(id),
    name        TEXT,
    file_count  INTEGER DEFAULT 0,
    UNIQUE (file_id, name)
);

CREATE TABLE IF NOT EXISTS important_messages (
    id                  SERIAL PRIMARY KEY,
    file_id             INTEGER REFERENCES files(id),
    message_id          INTEGER REFERENCES messages(id),
    sender              TEXT,
    content             TEXT,
    trigger_word        TEXT,
    detected_deadline   TEXT
);

CREATE TABLE IF NOT EXISTS chats (
    id              SERIAL PRIMARY KEY,
    chat_name       TEXT NOT NULL UNIQUE,
    source_type     TEXT,
    last_msg_date   TIMESTAMPTZ,
    total_messages  INTEGER DEFAULT 0,
    total_files     INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE files ADD COLUMN IF NOT EXISTS chat_id INTEGER REFERENCES chats(id);
ALTER TABLE files ADD COLUMN IF NOT EXISTS chat_name TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS msg_timestamp TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_chats_name ON chats(chat_name);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(msg_timestamp);
CREATE INDEX IF NOT EXISTS idx_files_chat_id ON files(chat_id);

CREATE TABLE IF NOT EXISTS document_chunks (
    id              SERIAL PRIMARY KEY,
    attachment_id   INTEGER REFERENCES attachments(id) ON DELETE CASCADE,
    file_id         INTEGER REFERENCES files(id),
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    embedding       FLOAT8[],
    page_number     INTEGER,
    source_name     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_attachment ON document_chunks(attachment_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON document_chunks(file_id);
"""

try:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for stmt in BASE_SCHEMA.split(";"):
            s = stmt.strip()
            if s:
                try:
                    conn.execute(text(s))
                    print(f"Executed: {s[:50]}...")
                except Exception as e:
                    print(f"Error on {s[:50]}: {e}")
    print("Database tables fixed successfully!")
except Exception as e:
    print(f"Failed to connect: {e}")
