-- ============================================================
-- RAG: Document chunks table for Q&A retrieval
-- ============================================================

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
