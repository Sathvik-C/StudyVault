from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sqlalchemy import text

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOAD_ERROR: str | None = None
_LAST_MODEL_ATTEMPT_AT = 0.0
_MODEL_RETRY_SECONDS = 30


def _get_model():
    """Lazy-load embedding model so app startup stays fast and resilient."""
    global _MODEL, _MODEL_LOAD_ERROR, _LAST_MODEL_ATTEMPT_AT
    if _MODEL is not None:
        return _MODEL
    now = time.time()
    if _MODEL_LOAD_ERROR is not None and (now - _LAST_MODEL_ATTEMPT_AT) < _MODEL_RETRY_SECONDS:
        return None

    try:
        _LAST_MODEL_ATTEMPT_AT = now
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _MODEL_LOAD_ERROR = None
        return _MODEL
    except Exception as exc:
        _MODEL_LOAD_ERROR = str(exc)
        logger.warning("Semantic model unavailable, falling back to keyword search: %s", exc)
        return None


def semantic_available() -> bool:
    return _get_model() is not None


def semantic_unavailable_reason() -> str | None:
    _get_model()
    return _MODEL_LOAD_ERROR


def _cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    denom = np.maximum(doc_norms * max(query_norm, 1e-12), 1e-12)
    return (doc_matrix @ query_vec) / denom


def _text_for_attachment(row: Any) -> str:
    parts = [
        row.original_name or "",
        row.category or "",
        row.subject or "",
        row.subcategory or "",
        row.note or "",
    ]
    tags = row.tags or []
    if isinstance(tags, list):
        parts.extend([str(t) for t in tags])
    return " ".join(p for p in parts if p).strip()


def semantic_search(engine, q: str, file_id: int | None = None, limit: int = 50) -> dict[str, Any] | None:
    model = _get_model()
    if model is None:
        return None

    limit = max(1, min(limit, 100))
    msg_pool_limit = max(500, limit * 40)
    file_pool_limit = max(500, limit * 40)

    params: dict[str, Any] = {
        "msg_pool_limit": msg_pool_limit,
        "file_pool_limit": file_pool_limit,
    }
    msg_filter = ""
    file_filter = ""
    if file_id:
        params["file_id"] = file_id
        msg_filter = "AND file_id = :file_id"
        file_filter = "AND file_id = :file_id"

    with engine.begin() as conn:
        message_rows = conn.execute(
            text(
                f"""
                SELECT id, file_id, role, content, message_order, is_important, msg_timestamp
                FROM messages
                WHERE content IS NOT NULL AND btrim(content) <> '' {msg_filter}
                ORDER BY id DESC
                LIMIT :msg_pool_limit
                """
            ),
            params,
        ).fetchall()

        attachment_rows = conn.execute(
            text(
                f"""
                SELECT id, file_id, original_name, storage_path, category, subject,
                       subcategory, note, tags
                FROM attachments
                WHERE original_name IS NOT NULL {file_filter}
                ORDER BY id DESC
                LIMIT :file_pool_limit
                """
            ),
            params,
        ).fetchall()

    msg_texts = [r.content for r in message_rows]
    att_texts = [_text_for_attachment(r) for r in attachment_rows]

    # If no candidate text exists, return empty semantic result shape.
    if not msg_texts and not att_texts:
        return {"query": q, "messages": [], "attachments": [], "search_mode": "semantic"}

    corpus = [q] + msg_texts + att_texts
    embeddings = model.encode(corpus, normalize_embeddings=False, show_progress_bar=False)
    emb = np.asarray(embeddings, dtype=np.float32)

    query_vec = emb[0]
    msg_mat = emb[1 : 1 + len(msg_texts)] if msg_texts else np.zeros((0, emb.shape[1]), dtype=np.float32)
    att_mat = emb[1 + len(msg_texts) :] if att_texts else np.zeros((0, emb.shape[1]), dtype=np.float32)

    scored_messages = []
    if len(message_rows):
        msg_scores = _cosine_similarity(query_vec, msg_mat)
        for row, score in zip(message_rows, msg_scores):
            d = dict(row._mapping)
            d["semantic_score"] = float(score)
            scored_messages.append(d)
        scored_messages.sort(key=lambda x: x["semantic_score"], reverse=True)
        scored_messages = scored_messages[:limit]

    scored_attachments = []
    if len(attachment_rows):
        att_scores = _cosine_similarity(query_vec, att_mat)
        for row, score in zip(attachment_rows, att_scores):
            d = dict(row._mapping)
            d["semantic_score"] = float(score)
            scored_attachments.append(d)
        scored_attachments.sort(key=lambda x: x["semantic_score"], reverse=True)
        scored_attachments = scored_attachments[:limit]

    return {
        "query": q,
        "messages": scored_messages,
        "attachments": scored_attachments,
        "search_mode": "semantic",
    }
