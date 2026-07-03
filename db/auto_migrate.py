"""
Auto-migration: ensures all required tables exist on startup.
Runs the SQL migration files idempotently (CREATE IF NOT EXISTS).
"""

import logging
from pathlib import Path
from sqlalchemy import text

logger = logging.getLogger(__name__)

MIGRATION_DIR = Path(__file__).resolve().parent


def _read_sql(filename: str) -> str:
    path = MIGRATION_DIR / filename
    if not path.exists():
        logger.warning("Migration file not found: %s", path)
        return ""
    return path.read_text(encoding="utf-8")


# Base schema — files, messages, attachments, subjects, important_messages
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
"""


def run_migrations(engine) -> None:
    """Run all migrations idempotently."""
    logger.info("Running database migrations...")

    with engine.begin() as conn:
        # 1. Base schema
        for statement in BASE_SCHEMA.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        logger.info("Base schema OK")

        # 2. Chats table + incremental ingestion columns (migrate.sql)
        migrate_sql = _read_sql("migrate.sql")
        if migrate_sql:
            for statement in migrate_sql.split(";"):
                stmt = statement.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        logger.debug("Migration statement skipped (likely already applied): %s", e)
            logger.info("Chats migration OK")

        # 3. RAG document_chunks table (rag_schema.sql)
        rag_sql = _read_sql("rag_schema.sql")
        if rag_sql:
            for statement in rag_sql.split(";"):
                stmt = statement.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        logger.debug("RAG migration statement skipped (likely already applied): %s", e)
            logger.info("RAG schema OK")

    logger.info("All migrations complete")
