"""
RAG Q&A API endpoints.

- POST /rag/{file_id}/ask      — Ask a question about uploaded documents
- GET  /rag/{file_id}/status    — Check indexing status
- POST /rag/{file_id}/index     — Manually trigger indexing for all PDFs
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from db.connection import get_engine
from services.rag import ask, get_index_status, index_attachment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG Q&A"])
engine = get_engine()


class AskRequest(BaseModel):
    question: str


@router.post("/{file_id}/ask")
def rag_ask(file_id: int, req: AskRequest):
    """Ask a question about the documents in this upload."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = ask(engine, req.question.strip(), file_id)
    return result


@router.get("/{file_id}/status")
def rag_status(file_id: int):
    """Check how many documents have been indexed for RAG."""
    return get_index_status(engine, file_id)


@router.post("/{file_id}/index")
def rag_index(file_id: int):
    """Manually trigger indexing for all PDF attachments in this upload."""
    with engine.begin() as conn:
        attachments = conn.execute(
            text("""
                SELECT a.id, a.original_name, a.storage_path, f.chat_id
                FROM attachments a
                JOIN files f ON a.file_id = f.id
                WHERE a.file_id = :fid
                  AND a.original_name ILIKE '%%.pdf'
            """),
            {"fid": file_id},
        ).fetchall()

    if not attachments:
        return {"message": "No PDF attachments found", "indexed": 0}
        
    import os
    if not os.getenv("HF_API_TOKEN"):
        raise HTTPException(status_code=400, detail="HF_API_TOKEN is missing in Render environment variables. Please get a free Hugging Face token and add it to Render to enable AI document indexing.")

    from models.config import STORAGE_DIR
    total_chunks = 0
    indexed_count = 0

    for att in attachments:
        chat_id = att.chat_id
        file_path = str(STORAGE_DIR / f"chat_{chat_id}" / att.storage_path)
        # index_attachment handles Supabase fallback if file not on disk
        try:
            chunks = index_attachment(
                engine, att.id, file_id, att.original_name, file_path
            )
            total_chunks += chunks
            if chunks > 0:
                indexed_count += 1
        except Exception as e:
            logger.warning("Failed to index attachment %d (%s): %s",
                           att.id, att.original_name, e)

    return {
        "message": f"Indexed {indexed_count} PDFs with {total_chunks} chunks",
        "indexed": indexed_count,
        "total_chunks": total_chunks,
    }


@router.post("/{file_id}/backfill-storage")
def backfill_storage(file_id: int):
    """
    Upload any existing attachments (missing supabase_key) to Supabase Storage.
    Run this once after enabling Supabase to backfill existing files.
    """
    from services.storage import is_available, upload_file as sb_upload
    from models.config import STORAGE_DIR

    if not is_available():
        raise HTTPException(status_code=400, detail="Supabase not configured")

    with engine.begin() as conn:
        attachments = conn.execute(
            text("""
                SELECT a.id, a.original_name, a.storage_path, f.chat_id
                FROM attachments a
                JOIN files f ON a.file_id = f.id
                WHERE a.file_id = :fid
                  AND a.supabase_key IS NULL
            """),
            {"fid": file_id},
        ).fetchall()

    if not attachments:
        return {"message": "All files already in Supabase", "uploaded": 0}

    uploaded = 0
    failed = 0

    for att in attachments:
        file_path = STORAGE_DIR / f"chat_{att.chat_id}" / att.storage_path
        if not file_path.is_file():
            failed += 1
            continue

        sb_key = f"chat_{att.chat_id}/{att.storage_path}".replace("\\", "/")
        try:
            ok = sb_upload(file_path, sb_key)
            if ok:
                with engine.begin() as conn:
                    conn.execute(
                        text("UPDATE attachments SET supabase_key = :key WHERE id = :aid"),
                        {"key": sb_key, "aid": att.id}
                    )
                uploaded += 1
            else:
                failed += 1
        except Exception as e:
            logger.warning("Backfill failed for %s: %s", att.original_name, e)
            failed += 1

    return {
        "message": f"Backfill complete: {uploaded} uploaded, {failed} failed/missing",
        "uploaded": uploaded,
        "failed": failed,
        "remaining": failed,
    }
