from __future__ import annotations
"""
Handles ZIP upload, extraction, parsing (Telegram or WhatsApp),
attachment movement, and batch AI classification.

Auto-detects export type:
  - result.json found → Telegram
  - _chat.txt found   → WhatsApp

Incremental ingestion:
  - Extracts chat name from export
  - On re-upload of same chat, skips messages already processed
  - Only classifies and moves new files
"""

import hashlib
import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from models.config import (
    UPLOAD_DIR,
    STORAGE_DIR,
    MAX_EXTRACTED_BYTES,
)
from services.classifier import batch_classify
from services.important import detect_important, write_important_messages_file
from services.whatsapp_parser import find_chat_txt, parse_whatsapp_chat
from services.file_extractor import extract_first_page_text

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif", ".tiff"}


# ── Helpers ──────────────────────────────────────────────────

def _flatten_telegram_text(text_value) -> str:
    if isinstance(text_value, str):
        return text_value
    if isinstance(text_value, list):
        chunks = []
        for item in text_value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(text_value) if text_value is not None else ""


def _is_real_file_path(value) -> bool:
    if not value:
        return False
    if isinstance(value, str) and value.strip().startswith("("):
        return False
    return True


def _find_result_json(extract_path: Path) -> Path | None:
    direct = extract_path / "result.json"
    if direct.is_file():
        return direct
    for candidate in extract_path.rglob("result.json"):
        if candidate.is_file():
            return candidate
    return None


def _safe_extract_zip(zip_path: Path, extract_path: Path) -> None:
    extract_root = extract_path.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        total = 0
        for member in zf.infolist():
            total += member.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise HTTPException(status_code=413, detail="Extracted content exceeds size limit")
            target = (extract_root / member.filename).resolve()
            if not str(target).startswith(str(extract_root)):
                raise HTTPException(status_code=400, detail="Invalid ZIP: path traversal detected")
        zf.extractall(extract_root)


def _resolve_telegram_source(extract_path: Path, json_dir: Path, relative: str) -> Path | None:
    extract_root = extract_path.resolve()
    for candidate in [(json_dir / relative).resolve(), (extract_root / relative).resolve()]:
        if str(candidate).startswith(str(extract_root)) and candidate.is_file():
            return candidate
    logger.warning("Telegram attachment not found: %s", relative)
    return None


def _unique_dest(dest_dir: Path, filename: str) -> Path:
    target = dest_dir / filename
    if not target.exists():
        return target
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for i in range(1, 9999):
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _get_context(messages: list[dict], index: int, window: int = 3) -> str:
    start = max(0, index - window)
    end = min(len(messages), index + window + 1)
    return " ".join(
        m.get("content", "") for m in messages[start:end] if m.get("content")
    )


def _guess_subject_from_context(context: str) -> str:
    """Guess the academic subject from surrounding chat context."""
    import re
    context_lower = context.lower() if context else ""

    subject_keywords = [
        (r"math|calculus|algebra|trigonometry|bmats", "Mathematics"),
        (r"physics|bphys|superconductor|laser|optic", "Physics"),
        (r"chemistry|chem", "Chemistry"),
        (r"biology|bio", "Biology"),
        (r"computer\s*science|cse|programming|python|java|c\+\+", "Computer Science"),
        (r"data\s*structure|dsa|bcsl305|bcs30", "Data Structures"),
        (r"operating\s*system", "Operating Systems"),
        (r"english|phonetics|professional\s*writing", "English"),
        (r"constitution|ico|dpsp", "Indian Constitution"),
        (r"data\s*analytics|dae|bcs358", "Data Analytics"),
        (r"kannada", "Kannada"),
        (r"exam|internal\s*test|ia\s*\d|cia", "Exams"),
        (r"attend|absent|present", "Attendance"),
        (r"lab|experiment", "Lab"),
    ]

    for pattern, subject in subject_keywords:
        if re.search(pattern, context_lower):
            return subject

    return "General"


def _parse_telegram_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _insert_attachment(conn, db_file_id, message_id, original_name,
                        storage_rel, category, subject, subcat, method,
                        file_hash, size, text_func) -> int | None:
    try:
        row = conn.execute(
            text_func("""
                INSERT INTO attachments
                    (file_id, message_id, original_name, storage_path,
                     category, subject, subcategory, classification_method,
                     file_hash, size_bytes)
                VALUES
                    (:file_id, :message_id, :original_name, :storage_path,
                     :category, :subject, :subcategory, :method,
                     :file_hash, :size_bytes)
                ON CONFLICT (file_hash) DO UPDATE SET
                    file_id = EXCLUDED.file_id,
                    message_id = EXCLUDED.message_id,
                    category = EXCLUDED.category,
                    subject = EXCLUDED.subject,
                    subcategory = EXCLUDED.subcategory,
                    storage_path = EXCLUDED.storage_path
                RETURNING id
            """),
            {
                "file_id": db_file_id,
                "message_id": message_id,
                "original_name": original_name,
                "storage_path": str(storage_rel),
                "category": category,
                "subject": subject,
                "subcategory": subcat,
                "method": method,
                "file_hash": file_hash,
                "size_bytes": size,
            },
        ).fetchone()
        return row.id if row else None
    except Exception as e:
        logger.error("DB Error: Failed to insert attachment %s: %s", original_name, e)
        return None


# ── Telegram parser ───────────────────────────────────────────

def _parse_telegram(json_path: Path, extract_path: Path) -> tuple[str, list[dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chat_name = data.get("name") or data.get("title") or "Unknown Chat"
    json_dir = json_path.parent
    messages = []

    for order, msg in enumerate(data.get("messages", [])):
        if msg.get("type") != "message":
            continue
        raw_file = msg.get("file") or msg.get("photo")
        file_val = raw_file if _is_real_file_path(raw_file) else None

        resolved = None
        if file_val:
            resolved = _resolve_telegram_source(extract_path, json_dir, str(file_val))

        messages.append({
            "role": msg.get("from", "unknown"),
            "content": _flatten_telegram_text(msg.get("text", "")),
            "order": order,
            "file": resolved,
            "date": _parse_telegram_date(msg.get("date")),
        })

    logger.info("Telegram: parsed %d messages from chat '%s'", len(messages), chat_name)
    return chat_name, messages


def _extract_whatsapp_chat_name(chat_txt_path: Path) -> str:
    name = chat_txt_path.stem
    for prefix in ["WhatsApp Chat with ", "WhatsApp Chat - "]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name or "Unknown WhatsApp Chat"


# ── Chat registry ─────────────────────────────────────────────

def _get_or_create_chat(conn, text_func, chat_name: str, source_type: str) -> tuple[int, datetime | None]:
    row = conn.execute(
        text_func("SELECT id, last_msg_date FROM chats  WHERE chat_name = :chat_name"),
        {"chat_name": chat_name}
    ).fetchone()

    if row:
        chat_id = row.id
        # Force ALL files with this name to use this chat_id
        conn.execute(
            text_func("UPDATE files SET chat_id = :chat_id WHERE chat_name = :chat_name"),
            {"chat_id": chat_id, "chat_name": chat_name}
        )
        # Use the actual last message timestamp from DB (ground truth)
        # so stale/pre-migration values in chats.last_msg_date don't block uploads
        actual_last_date = conn.execute(
            text_func("""
                SELECT MAX(m.msg_timestamp)
                FROM messages m
                JOIN files f ON m.file_id = f.id
                WHERE f.chat_id = :cid
            """),
            {"cid": chat_id}
        ).scalar()
        logger.info("Existing chat '%s', actual last msg in DB: %s", chat_name, actual_last_date)
        return chat_id, actual_last_date
    else:
        result = conn.execute(
            text_func("""
                INSERT INTO chats (chat_name, source_type)
                VALUES (:chat_name, :source_type)
                RETURNING id
            """),
            {"chat_name": chat_name, "source_type": source_type}
        )
        chat_id = result.scalar()
        logger.info("New chat '%s' created with id %d", chat_name, chat_id)
        return chat_id, None


def _update_chat(conn, text_func, chat_id: int, last_msg_date: datetime,
                 new_messages: int, new_files: int):
    conn.execute(
        text_func("""
            UPDATE chats
            SET last_msg_date = :last_date,
                total_messages = total_messages + :msgs,
                total_files = total_files + :files,
                updated_at = NOW()
            WHERE id = :chat_id
        """),
        {"chat_id": chat_id, "last_date": last_msg_date, "msgs": new_messages, "files": new_files}
    )


def _get_existing_structure(conn, text_func, chat_id: int) -> list[str]:
    rows = conn.execute(
        text_func("""
            SELECT DISTINCT a.category, a.subject, a.subcategory
            FROM attachments a
            JOIN files f ON a.file_id = f.id
            WHERE f.chat_id = :chat_id
        """),
        {"chat_id": chat_id}
    ).fetchall()

    structure = []
    for r in rows:
        parts = [r.category, r.subject]
        if r.subcategory:
            parts.append(r.subcategory)
        structure.append(" > ".join(parts))
    return structure


# ── Main ingestion ────────────────────────────────────────────

def ingest_zip(zip_path: Path, zip_uuid: str, filename: str,
               file_hash: str, file_size: int, engine) -> dict:
    from sqlalchemy import text

    extract_path = UPLOAD_DIR / zip_uuid

    # Always start with a clean extraction folder so that files moved to
    # storage during a previous run are re-extracted and available again.
    if extract_path.exists():
        shutil.rmtree(extract_path, ignore_errors=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    try:
        _safe_extract_zip(zip_path, extract_path)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Not a valid ZIP file")

    # ── Auto-detect export type ───────────────────────────────
    json_path = _find_result_json(extract_path)
    chat_txt_path = find_chat_txt(extract_path)

    if json_path:
        source_type = "telegram"
        chat_name, raw_messages = _parse_telegram(json_path, extract_path)
    elif chat_txt_path:
        source_type = "whatsapp"
        chat_name = _extract_whatsapp_chat_name(chat_txt_path)
        raw_messages = parse_whatsapp_chat(chat_txt_path, extract_path)
    else:
        shutil.rmtree(extract_path, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="Unrecognized export. Expected Telegram (result.json) or WhatsApp (_chat.txt)",
        )

    # ── Get or create chat, find cutoff date ─────────────────
    with engine.begin() as conn:
        chat_id, last_msg_date = _get_or_create_chat(conn, text, chat_name, source_type)

        # ── Find-or-create the files record for this chat ────
        existing_file = conn.execute(
            text("SELECT id FROM files WHERE chat_id = :chat_id ORDER BY uploaded_at ASC LIMIT 1"),
            {"chat_id": chat_id}
        ).fetchone()

        is_reupload = bool(existing_file)
        if existing_file:
            # Re-upload of same chat → reuse the original file record
            db_file_id = existing_file.id
            conn.execute(
                text("""
                    UPDATE files
                    SET file_hash = :file_hash, size = :size, uploaded_at = NOW()
                    WHERE id = :id
                """),
                {"file_hash": file_hash, "size": file_size, "id": db_file_id}
            )
            logger.info("Re-upload: merging into existing file_id=%d for chat '%s'", db_file_id, chat_name)
        else:
            # First upload for this chat → create new file record
            db_file_id = conn.execute(
                text("""
                    INSERT INTO files (file_id, filename, size, uploaded_at, file_hash, chat_id, chat_name)
                    VALUES (:file_id, :filename, :size, NOW(), :file_hash, :chat_id, :chat_name)
                    RETURNING id
                """),
                {
                    "file_id": zip_uuid,
                    "filename": filename,
                    "size": file_size,
                    "file_hash": file_hash,
                    "chat_id": chat_id,
                    "chat_name": chat_name,
                }
            ).scalar()
            logger.info("First upload: created file_id=%d for chat '%s'", db_file_id, chat_name)

        existing_structure = _get_existing_structure(conn, text, chat_id)

    # ── Filter to only new messages ───────────────────────────
    if last_msg_date:
        new_messages = [
            m for m in raw_messages
            if not m.get("date") or m["date"] > last_msg_date
        ]
        skipped = len(raw_messages) - len(new_messages)
        logger.info("Incremental: skipping %d old, processing %d new messages", skipped, len(new_messages))
    else:
        new_messages = raw_messages
        skipped = 0
        logger.info("First upload: processing all %d messages", len(new_messages))

    if not new_messages:
        shutil.rmtree(extract_path, ignore_errors=True)
        return {
            "db_file_id": db_file_id,
            "source_type": source_type,
            "chat_name": chat_name,
            "message_count": 0,
            "skipped_count": skipped,
            "important_count": 0,
            "attachments_moved": 0,
            "attachments_missing": 0,
            "subjects_detected": [],
            "new_attachment_ids": [],
            "is_reupload": is_reupload,
            "note": "No new messages since last upload",
        }

    # ── Insert new messages into DB ───────────────────────────
    message_db_ids = []
    with engine.begin() as conn:
        for msg in new_messages:
            row = conn.execute(
                text("""
                    INSERT INTO messages (file_id, role, content, message_order, msg_timestamp)
                    VALUES (:file_id, :role, :content, :message_order, :msg_timestamp)
                    RETURNING id
                """),
                {
                    "file_id": db_file_id,
                    "role": msg["role"],
                    "content": msg["content"],
                    "message_order": msg["order"],
                    "msg_timestamp": msg.get("date") if isinstance(msg.get("date"), datetime) else None,
                },
            )
            message_db_ids.append(row.scalar())

    # ── Background Processing for Heavy AI Tasks ──────────────
    import threading

    def _bg_process(messages_data, db_ids, extract_path_, chat_id_, chat_storage_):
        try:
            logger.info("Background processing started for file_id=%d", db_file_id)
            
            # ── Detect important messages ─────────────────────────────
            flagged = detect_important(messages_data)

            if flagged:
                flagged_orders = {m["order"] for m in flagged}
                with engine.begin() as conn:
                    for i, msg in enumerate(messages_data):
                        if msg["order"] in flagged_orders:
                            conn.execute(
                                text("UPDATE messages SET is_important = TRUE WHERE id = :id"),
                                {"id": db_ids[i]},
                            )
                    for msg in flagged:
                        idx = next(i for i, m in enumerate(messages_data) if m["order"] == msg["order"])
                        conn.execute(
                            text("""
                                INSERT INTO important_messages
                                    (file_id, message_id, sender, content, trigger_word, detected_deadline)
                                VALUES (:file_id, :message_id, :sender, :content, :trigger_word, :deadline)
                            """),
                            {
                                "file_id": db_file_id,
                                "message_id": db_ids[idx],
                                "sender": msg["role"],
                                "content": msg["content"],
                                "trigger_word": msg["trigger_word"],
                                "deadline": msg["detected_deadline"],
                            },
                        )

            # ── Separate images from documents ────────────────────────
            file_entries = []
            image_entries = []

            for i, msg in enumerate(messages_data):
                source = msg.get("file")
                if not source or not isinstance(source, Path) or not source.is_file():
                    continue
                ext = source.suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    image_entries.append({"msg_index": i, "source": source})
                else:
                    context = _get_context(messages_data, i)
                    from services.classifier import _apply_filename_rules
                    pre_cat = _apply_filename_rules(source.name, "")
                    if pre_cat:
                        first_page_text = ""
                    else:
                        first_page_text = extract_first_page_text(source)
                        if first_page_text:
                            logger.debug("Extracted %d chars from first page of %s",
                                         len(first_page_text), source.name)
                    file_entries.append({
                        "msg_index": i,
                        "source": source,
                        "filename": source.name,
                        "context": context,
                        "first_page_text": first_page_text,
                    })

            logger.info("Found %d documents to classify, %d images to move directly",
                        len(file_entries), len(image_entries))

            # ── Batch classify documents ──────────────────────────────
            classification_map = {}
            if file_entries:
                file_list = [
                    {
                        "filename": e["filename"],
                        "context": e["context"],
                        "first_page_text": e.get("first_page_text", ""),
                    }
                    for e in file_entries
                ]
                classification_map = batch_classify(file_list, existing_structure)

            # ── Setup storage ─────────────────────────────────────────
            chat_storage_.mkdir(parents=True, exist_ok=True)

            moved, missing = 0, 0
            subject_counts: dict[str, int] = {}
            new_attachment_ids: list[int] = []

            # ── Setup Supabase storage ────────────────────────────
            from services.storage import upload_file as sb_upload, is_available as sb_available
            use_supabase = sb_available()
            if use_supabase:
                logger.info("Supabase storage available — files will be uploaded to cloud")
            else:
                logger.warning("Supabase not configured — files stored on disk only")

            # ── Move images ─────────────────────────
            for entry in image_entries:
                source = entry["source"]
                if not source or not source.exists() or not source.is_file():
                    missing += 1
                    logger.warning("Image not found, skipping: %s", source)
                    continue

                i = entry["msg_index"]
                img_context = _get_context(messages_data, i)
                img_subject = _guess_subject_from_context(img_context)

                img_dir = chat_storage_ / "Images" / img_subject
                img_dir.mkdir(parents=True, exist_ok=True)

                dest = _unique_dest(img_dir, source.name)
                file_hash_val = _file_sha256(source)
                size = source.stat().st_size
                shutil.move(str(source), str(dest))
                moved += 1
                storage_rel = dest.relative_to(chat_storage_)

                # Upload to Supabase
                supabase_key = None
                if use_supabase:
                    sb_key = f"chat_{chat_id_}/Images/{img_subject}/{dest.name}"
                    if sb_upload(dest, sb_key):
                        supabase_key = sb_key

                with engine.begin() as conn:
                    aid = _insert_attachment(conn, db_file_id, db_ids[i],
                                       source.name, storage_rel,
                                       "Images", img_subject, None, "context",
                                       file_hash_val, size, text)
                    if aid:
                        new_attachment_ids.append(aid)
                        if supabase_key:
                            conn.execute(
                                text("UPDATE attachments SET supabase_key = :key WHERE id = :aid"),
                                {"key": supabase_key, "aid": aid}
                            )

            # ── Move documents ──────────────────
            for entry in file_entries:
                source = entry["source"]
                if not source or not source.exists() or not source.is_file():
                    missing += 1
                    logger.warning("File not found, skipping: %s", source)
                    continue

                classification = classification_map.get(entry["filename"], {
                    "category": "Other", "subject": "Unknown", "subcategory": None, "method": "fallback"
                })

                category = classification["category"]
                subject  = classification["subject"]
                subcat   = classification["subcategory"]
                method   = classification["method"]

                dest_dir = chat_storage_ / category / subject
                if subcat:
                    dest_dir = dest_dir / subcat
                dest_dir.mkdir(parents=True, exist_ok=True)

                dest = _unique_dest(dest_dir, source.name)
                file_hash_val = _file_sha256(source)
                size = source.stat().st_size
                shutil.move(str(source), str(dest))
                moved += 1
                subject_counts[subject] = subject_counts.get(subject, 0) + 1
                storage_rel = dest.relative_to(chat_storage_)
                i = entry["msg_index"]

                # Upload to Supabase
                supabase_key = None
                if use_supabase:
                    sb_path = str(storage_rel).replace("\\", "/")
                    sb_key = f"chat_{chat_id_}/{sb_path}"
                    if sb_upload(dest, sb_key):
                        supabase_key = sb_key

                with engine.begin() as conn:
                    aid = _insert_attachment(conn, db_file_id, db_ids[i],
                                       source.name, storage_rel,
                                       category, subject, subcat, method,
                                       file_hash_val, size, text)
                    if aid:
                        new_attachment_ids.append(aid)
                        if supabase_key:
                            conn.execute(
                                text("UPDATE attachments SET supabase_key = :key WHERE id = :aid"),
                                {"key": supabase_key, "aid": aid}
                            )

            # ── Update subjects registry ──────────────────────────────
            with engine.begin() as conn:
                for subject, count in subject_counts.items():
                    conn.execute(
                        text("""
                            INSERT INTO subjects (file_id, name, file_count)
                            VALUES (:file_id, :name, :count)
                            ON CONFLICT (file_id, name)
                            DO UPDATE SET file_count = subjects.file_count + EXCLUDED.file_count
                        """),
                        {"file_id": db_file_id, "name": subject, "count": count},
                    )

            # ── Update chat's last processed timestamp ────────────────
            dates_with_values = [m["date"] for m in messages_data if m.get("date")]
            if dates_with_values:
                latest_date = max(dates_with_values)
                with engine.begin() as conn:
                    _update_chat(conn, text, chat_id_, latest_date, len(messages_data), moved)

            # ── Write important_messages.txt ──────────────────────────
            write_important_messages_file(flagged, chat_storage_ / "important_messages.txt")

            # ── Cleanup ───────────────────────────────────────────────
            shutil.rmtree(extract_path_, ignore_errors=True)
            
            # ── Trigger RAG Indexing for PDFs ─────────────────────────
            try:
                from services.rag import index_attachment
                with engine.begin() as conn:
                    pdf_attachments = conn.execute(
                        text("""
                            SELECT a.id, a.original_name, a.storage_path
                            FROM attachments a
                            WHERE a.file_id = :fid
                              AND a.original_name ILIKE '%%.pdf'
                        """),
                        {"fid": db_file_id}
                    ).fetchall()

                logger.info("Starting background RAG indexing for %d PDFs in file %d", len(pdf_attachments), db_file_id)
                for pdf_row in pdf_attachments:
                    pdf_path = str(chat_storage_ / pdf_row.storage_path)
                    try:
                        index_attachment(engine, pdf_row.id, db_file_id, pdf_row.original_name, pdf_path)
                    except Exception as e:
                        logger.error("Failed to index PDF %s (attachment %d): %s", pdf_row.original_name, pdf_row.id, e)
                logger.info("Background RAG indexing complete for file %d", db_file_id)
            except Exception as e:
                logger.error("Error setting up RAG index: %s", e)
                
            logger.info("Background processing complete for file_id=%d", db_file_id)

        except Exception as e:
            logger.error("Background processing failed for file_id=%d: %s", db_file_id, e)

    # Start the background thread
    chat_storage = STORAGE_DIR / f"chat_{chat_id}"
    t = threading.Thread(
        target=_bg_process,
        args=(new_messages, message_db_ids, extract_path, chat_id, chat_storage)
    )
    t.daemon = True
    t.start()

    return {
        "db_file_id": db_file_id,
        "source_type": source_type,
        "chat_name": chat_name,
        "message_count": len(new_messages),
        "skipped_count": skipped,
        "important_count": "processing",
        "attachments_moved": "processing",
        "attachments_missing": 0,
        "subjects_detected": [],
        "new_attachment_ids": [],
        "is_reupload": is_reupload,
        "status": "Processing in background...",
    }
