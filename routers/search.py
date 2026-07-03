from fastapi import APIRouter
from sqlalchemy import text
from db.connection import get_engine
from services.semantic_search import semantic_search, semantic_available, semantic_unavailable_reason

router = APIRouter(prefix="/search", tags=["Search"])
engine = get_engine()


@router.get("")
def search(q: str, file_id: int = None, semantic: bool = True, limit: int = 50):
    """
    Keyword search across messages and attachments.
    Phase 2: this will be replaced with pgvector semantic search.
    """
    limit = max(1, min(limit, 100))

    if semantic and semantic_available():
        result = semantic_search(engine=engine, q=q, file_id=file_id, limit=limit)
        if result is not None:
            return result

    like = f"%{q}%"
    params = {"q": like, "limit": limit}

    msg_filter = "AND file_id = :file_id" if file_id else ""
    if file_id:
        params["file_id"] = file_id

    with engine.begin() as conn:
        messages = conn.execute(
            text(f"""
                SELECT id, file_id, role, content, message_order, is_important, msg_timestamp
                FROM messages
                WHERE content ILIKE :q {msg_filter}
                ORDER BY file_id, message_order
                LIMIT :limit
            """),
            params,
        ).fetchall()

        attachments = conn.execute(
            text(f"""
                SELECT id, file_id, original_name, storage_path, category, subject
                FROM attachments
                WHERE original_name ILIKE :q {msg_filter}
                ORDER BY category, subject
                LIMIT :limit
            """),
            params,
        ).fetchall()

    fallback = {
        "query": q,
        "messages": [dict(r._mapping) for r in messages],
        "attachments": [dict(r._mapping) for r in attachments],
        "search_mode": "keyword",
    }
    if semantic:
        fallback["semantic_unavailable"] = True
        fallback["semantic_unavailable_reason"] = semantic_unavailable_reason()
    return fallback
