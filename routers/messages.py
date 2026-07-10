from __future__ import annotations
from pathlib import Path
import hashlib
import re
import shutil
from collections import Counter
from datetime import datetime
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import text
from pydantic import BaseModel
from db.connection import get_engine
from models.config import STORAGE_DIR
from services.important import extract_deadline_from_text

router = APIRouter(prefix="/messages", tags=["Messages"])
engine = get_engine()

class UpdateRequest(BaseModel):
    filename: str | None = None
    category: str | None = None
    subject: str | None = None
    subcategory: str | None = None

class NoteRequest(BaseModel):
    note: str | None = None
    tags: list[str] | None = None

class FolderRequest(BaseModel):
    category: str
    subject: str | None = "General"
    subcategory: str | None = None

def _cleanup_empty_dirs(path: Path, root: Path):
    try:
        while path != root and path.is_dir() and not any(path.iterdir()):
            parent = path.parent
            path.rmdir()
            path = parent
    except Exception:
        pass

def _parse_deadline_date(value: str | None, reference_dt: datetime | None = None, context_text: str | None = None):
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    raw = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)

    default_year = reference_dt.year if isinstance(reference_dt, datetime) else datetime.now().year

    def infer_year_from_context() -> int | None:
        if not context_text:
            return None
        text = str(context_text)

        # Prefer a 4-digit year found close to the detected deadline token.
        token_idx = text.lower().find(value.lower()) if value else -1
        years: list[tuple[int, int]] = []
        for m_year in re.finditer(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text):
            yr = int(m_year.group(1))
            dist = abs(m_year.start() - token_idx) if token_idx >= 0 else 10_000
            years.append((dist, yr))
        if years:
            years.sort(key=lambda x: x[0])
            return years[0][1]
        return None

    inferred_year = infer_year_from_context()

    # Strict day-first numeric formats: DD/MM[/YY|YYYY] or DD-MM[/YY|YYYY]
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?$", raw)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year_part = m.group(3)
        year = int(year_part) if year_part else (inferred_year or default_year)
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day).date()
        except ValueError:
            return None

    # Month-name variants: "15 Nov 2026" / "Nov 15, 2026"
    for fmt in (
        "%d %b %Y", "%d %B %Y", "%d %b %y", "%d %B %y",
        "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y",
        "%d %b", "%d %B", "%b %d", "%B %d",
    ):
        try:
            parsed = datetime.strptime(raw, fmt)
            if "%Y" in fmt or "%y" in fmt:
                return parsed.date()
            return parsed.replace(year=(inferred_year or default_year)).date()
        except ValueError:
            continue

    # ISO formats
    for fmt in ("%Y-%m-%d",):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue

    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
            return datetime.fromisoformat(raw).date()
    except Exception:
        pass
    return None

@router.get("/{file_id}")
def get_messages(file_id: int, limit: int = 100, offset: int = 0):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, role, content, message_order, is_important
                FROM messages
                WHERE file_id = :file_id
                ORDER BY message_order
                LIMIT :limit OFFSET :offset
            """),
            {"file_id": file_id, "limit": limit, "offset": offset},
        ).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/{file_id}/important")
def get_important_messages(file_id: int):
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT im.id, im.message_id, im.sender, im.content, im.trigger_word,
                       im.detected_deadline, m.msg_timestamp
                FROM important_messages im
                LEFT JOIN messages m ON m.id = im.message_id
                WHERE im.file_id = :file_id
                ORDER BY im.id
            """),
            {"file_id": file_id},
        ).fetchall()

    items = []
    for r in rows:
        item = dict(r._mapping)
        best_deadline = extract_deadline_from_text(item.get("content")) or item.get("detected_deadline")
        item["detected_deadline"] = best_deadline
        parsed = _parse_deadline_date(best_deadline, item.get("msg_timestamp"), item.get("content"))
        item["deadline_iso"] = parsed.isoformat() if parsed else None
        item["deadline_display"] = parsed.strftime("%d/%m/%Y") if parsed else best_deadline
        items.append(item)
    return items

@router.get("/{file_id}/deadlines/upcoming")
def get_upcoming_deadlines(file_id: int, days: int = 7):
    if days < 1:
        days = 1
    if days > 90:
        days = 90

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, message_id, sender, content, detected_deadline
                       , (SELECT m.msg_timestamp FROM messages m WHERE m.id = important_messages.message_id) AS msg_timestamp
                FROM important_messages
                WHERE file_id = :fid
            """),
            {"fid": file_id},
        ).fetchall()

    today = datetime.now().date()
    cutoff = today.fromordinal(today.toordinal() + days)
    upcoming = []
    for r in rows:
        best_deadline = extract_deadline_from_text(r.content) or r.detected_deadline
        dd = _parse_deadline_date(best_deadline, r.msg_timestamp, r.content)
        if dd is None:
            continue
        if today <= dd <= cutoff:
            upcoming.append({
                "important_id": r.id,
                "message_id": r.message_id,
                "sender": r.sender,
                "deadline": dd.isoformat(),
                "deadline_text": best_deadline,
                "snippet": (r.content or "")[:120],
                "days_left": (dd - today).days,
            })

    upcoming.sort(key=lambda x: (x["days_left"], x["deadline"]))
    return {
        "range_days": days,
        "count": len(upcoming),
        "items": upcoming,
    }

@router.get("/{file_id}/message-context/{message_id}")
def get_message_context(file_id: int, message_id: int, window: int = 5):
    if window < 1:
        window = 1
    if window > 25:
        window = 25

    with engine.begin() as conn:
        center = conn.execute(
            text("""
                SELECT id, role, content, message_order, msg_timestamp
                FROM messages
                WHERE id = :mid AND file_id = :fid
            """),
            {"mid": message_id, "fid": file_id},
        ).fetchone()
        if not center:
            raise HTTPException(status_code=404, detail="Message not found")

        start_order = max(1, center.message_order - window)
        end_order = center.message_order + window
        rows = conn.execute(
            text("""
                SELECT id, role, content, message_order, msg_timestamp
                FROM messages
                WHERE file_id = :fid AND message_order BETWEEN :start_o AND :end_o
                ORDER BY message_order
            """),
            {"fid": file_id, "start_o": start_order, "end_o": end_order},
        ).fetchall()

    return {
        "center_message_id": int(center.id),
        "window": window,
        "messages": [dict(r._mapping) for r in rows],
    }

@router.get("/{file_id}/attachments")
def get_attachments(file_id: int, category: str = None, subject: str = None):
    with engine.begin() as conn:
        cid = conn.execute(text("SELECT chat_id FROM files WHERE id = :fid"), {"fid": file_id}).scalar()
    if not cid: return []

    query = """
        SELECT a.id, a.original_name, a.storage_path, a.category, a.subject,
               a.subcategory, a.classification_method, a.size_bytes, a.file_id,
               a.note, a.tags
        FROM attachments a
        JOIN files f ON a.file_id = f.id
        WHERE f.chat_id = :cid
    """
    params = {"cid": cid}
    if category:
        query += " AND a.category = :category"
        params["category"] = category
    if subject:
        query += " AND a.subject = :subject"
        params["subject"] = subject

    query += " ORDER BY a.category, a.subject, a.original_name"
    with engine.begin() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/{file_id}/subjects")
def get_subjects(file_id: int):
    with engine.begin() as conn:
        cid = conn.execute(text("SELECT chat_id FROM files WHERE id = :fid"), {"fid": file_id}).scalar()
        if not cid: return []
        rows = conn.execute(
            text("""
                SELECT name, SUM(file_count) as file_count
                FROM subjects
                WHERE file_id IN (SELECT id FROM files WHERE chat_id = :cid)
                GROUP BY name
                ORDER BY file_count DESC
            """),
            {"cid": cid},
        ).fetchall()
    return [dict(r._mapping) for r in rows]

@router.get("/{file_id}/dashboard")
def get_dashboard(file_id: int):
    with engine.begin() as conn:
        chat_id = conn.execute(text("SELECT chat_id FROM files WHERE id = :fid"), {"fid": file_id}).scalar()
        if not chat_id:
            raise HTTPException(status_code=404, detail="File not found")

        total_messages = conn.execute(
            text("SELECT COUNT(*) FROM messages WHERE file_id = :fid"),
            {"fid": file_id},
        ).scalar() or 0

        total_important = conn.execute(
            text("SELECT COUNT(*) FROM important_messages WHERE file_id = :fid"),
            {"fid": file_id},
        ).scalar() or 0

        attachment_rows = conn.execute(
            text("""
                SELECT original_name, subject
                FROM attachments
                WHERE file_id = :fid
            """),
            {"fid": file_id},
        ).fetchall()

        sender_rows = conn.execute(
            text("""
                SELECT role, COUNT(*) AS cnt
                FROM messages
                WHERE file_id = :fid
                GROUP BY role
                ORDER BY cnt DESC
                LIMIT 7
            """),
            {"fid": file_id},
        ).fetchall()

        # We use chat scope for subject vocabulary, then count mentions in the selected file.
        subject_rows = conn.execute(
            text("""
                SELECT DISTINCT a.subject
                FROM attachments a
                JOIN files f ON a.file_id = f.id
                WHERE f.chat_id = :cid AND a.subject IS NOT NULL AND a.subject <> ''
            """),
            {"cid": chat_id},
        ).fetchall()

        msg_rows = conn.execute(
            text("SELECT content FROM messages WHERE file_id = :fid"),
            {"fid": file_id},
        ).fetchall()

        deadline_rows = conn.execute(
            text("""
                SELECT detected_deadline, content
                       , (SELECT m.msg_timestamp FROM messages m WHERE m.id = important_messages.message_id) AS msg_timestamp
                FROM important_messages
                WHERE file_id = :fid
            """),
            {"fid": file_id},
        ).fetchall()

    ext_counter = Counter()
    for r in attachment_rows:
        name = (r.original_name or "").strip()
        ext = Path(name).suffix.lower().lstrip(".") if "." in name else "unknown"
        ext_counter[ext or "unknown"] += 1

    type_breakdown = [
        {"type": k, "count": v}
        for k, v in ext_counter.most_common(8)
    ]

    subjects = [str(r.subject).strip() for r in subject_rows if r.subject and str(r.subject).strip()]
    subject_counter = Counter()
    for row in msg_rows:
        content = (row.content or "").lower()
        if not content:
            continue
        for subject in subjects:
            if subject.lower() in content:
                subject_counter[subject] += 1

    messages_per_subject = [
        {"subject": k, "count": v}
        for k, v in subject_counter.most_common(8)
    ]

    if not messages_per_subject:
        # Fallback so dashboard always has useful subject data
        fallback_counter = Counter((r.subject or "Unknown") for r in attachment_rows)
        messages_per_subject = [
            {"subject": k, "count": v}
            for k, v in fallback_counter.most_common(8)
            if k
        ]

    deadline_points = []
    for r in deadline_rows:
        best_deadline = extract_deadline_from_text(r.content) or r.detected_deadline
        dt = _parse_deadline_date(best_deadline, r.msg_timestamp, r.content)
        if dt is None:
            continue
        deadline_points.append({
            "date": dt.isoformat(),
            "source": best_deadline,
            "snippet": (r.content or "")[:140],
        })

    grouped_deadlines = Counter(item["date"] for item in deadline_points)
    deadlines = [
        {"date": d, "count": c}
        for d, c in sorted(grouped_deadlines.items(), key=lambda x: x[0])
    ]

    return {
        "summary": {
            "messages": int(total_messages),
            "important": int(total_important),
            "attachments": int(len(attachment_rows)),
            "subjects": int(len({(r.subject or "").strip() for r in attachment_rows if (r.subject or "").strip()})),
            "upcoming_deadlines": int(len(deadlines)),
        },
        "messages_per_subject": messages_per_subject,
        "attachment_types": type_breakdown,
        "top_senders": [
            {"sender": r.role or "Unknown", "count": int(r.cnt or 0)} for r in sender_rows
        ],
        "deadlines": deadlines,
    }

@router.patch("/{file_id}/attachments/{attachment_id}")
def update_attachment(file_id: int, attachment_id: int, req: UpdateRequest):
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT a.storage_path, a.original_name, f.chat_id, a.category, a.subject, a.subcategory
                FROM attachments a
                JOIN files f ON a.file_id = f.id
                WHERE a.id = :aid
            """),
            {"aid": attachment_id},
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Attachment not found")

        res = row._mapping
        chat_id = res['chat_id']
        old_rel_path = res['storage_path']
        old_subject = res['subject']
        
        new_filename = req.filename or res['original_name']
        new_cat = req.category or res['category']
        new_sub = req.subject or res['subject']
        new_subcat = req.subcategory if req.subcategory is not None else res['subcategory']
        
        chat_root = STORAGE_DIR / f"chat_{chat_id}"
        new_dir = chat_root / new_cat / new_sub
        if new_subcat:
            new_dir = new_dir / new_subcat
        new_dir.mkdir(parents=True, exist_ok=True)
        
        from services.ingestion import _unique_dest
        dest_path = _unique_dest(new_dir, new_filename)
        new_rel_path = dest_path.relative_to(chat_root)
        
        old_full_path = chat_root / old_rel_path
        if old_full_path.exists():
            shutil.move(str(old_full_path), str(dest_path))
        
        conn.execute(
            text("""
                UPDATE attachments
                SET original_name = :name, category = :cat, subject = :sub, 
                    subcategory = :subcat, storage_path = :path
                WHERE id = :aid
            """),
            {
                "aid": attachment_id, "name": new_filename, "cat": new_cat,
                "sub": new_sub, "subcat": new_subcat, "path": str(new_rel_path)
            }
        )
        
        if new_sub != old_subject:
            conn.execute(text("UPDATE subjects SET file_count = GREATEST(0, file_count - 1) WHERE file_id = :fid AND name = :name"), {"fid": file_id, "name": old_subject})
            conn.execute(text("INSERT INTO subjects (file_id, name, file_count) VALUES (:fid, :name, 1) ON CONFLICT (file_id, name) DO UPDATE SET file_count = subjects.file_count + 1"), {"fid": file_id, "name": new_sub})
        
        _cleanup_empty_dirs(old_full_path.parent, chat_root)
    return {"status": "success", "new_path": str(new_rel_path)}

@router.patch("/{file_id}/attachments/{attachment_id}/note")
def update_attachment_note(file_id: int, attachment_id: int, req: NoteRequest):
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM attachments WHERE id = :aid"), {"aid": attachment_id}
        ).scalar()
        if not exists:
            raise HTTPException(status_code=404, detail="Attachment not found")
        conn.execute(
            text("UPDATE attachments SET note = :note, tags = :tags WHERE id = :aid"),
            {"aid": attachment_id, "note": req.note, "tags": req.tags or []},
        )
    return {"status": "saved"}

@router.delete("/{file_id}/attachments/{attachment_id}")
def delete_attachment(file_id: int, attachment_id: int):
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT a.storage_path, f.chat_id, a.subject
                FROM attachments a
                JOIN files f ON a.file_id = f.id
                WHERE a.id = :aid
            """),
            {"aid": attachment_id},
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Attachment not found")

        res = row._mapping
        chat_root = STORAGE_DIR / f"chat_{res['chat_id']}"
        full_path = chat_root / res['storage_path']
        if full_path.exists():
            full_path.unlink()
            _cleanup_empty_dirs(full_path.parent, chat_root)
            
        conn.execute(text("DELETE FROM attachments WHERE id = :aid"), {"aid": attachment_id})
        conn.execute(text("UPDATE subjects SET file_count = GREATEST(0, file_count - 1) WHERE file_id = :fid AND name = :name"), {"fid": file_id, "name": res['subject']})
    return {"status": "deleted"}

@router.post("/{file_id}/folders")
def create_folder(file_id: int, req: FolderRequest):
    try:
        with engine.begin() as conn:
            # 1. Resolve chat_id and get a STABLE file_id for this chat
            row = conn.execute(text("SELECT chat_id FROM files WHERE id = :fid"), {"fid": file_id}).fetchone()
            if not row:
                # Fallback: Find the latest file_id for this chat if the session is old
                row = conn.execute(text("SELECT id, chat_id FROM files WHERE chat_name = (SELECT chat_name FROM chats ORDER BY updated_at DESC LIMIT 1) ORDER BY id DESC")).fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Session expired. Please upload a ZIP or refresh.")
            
            fid, cid = (row[0], row[1]) if len(row) > 1 else (file_id, row[0])
            
            chat_root = STORAGE_DIR / f"chat_{cid}"
            path = chat_root / req.category / (req.subject or "General")
            if req.subcategory: path = path / req.subcategory
            path.mkdir(parents=True, exist_ok=True)
            
            import time
            keep_name = f".keep_{int(time.time())}"
            (path / keep_name).touch()
            
            rel_path = (path / keep_name).relative_to(chat_root)
            conn.execute(
                text("""
                    INSERT INTO attachments (file_id, original_name, storage_path, category, subject, subcategory, size_bytes, classification_method)
                    VALUES (:fid, :name, :path, :cat, :sub, :subcat, 0, 'manual')
                """),
                {"fid": fid, "name": keep_name, "path": str(rel_path), "cat": req.category, "sub": req.subject or "General", "subcat": req.subcategory}
            )
        return {"status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{file_id}/folders")
def delete_folder(file_id: int, category: str = Query(...), subject: str = Query(None), subcategory: str = Query(None)):
    try:
        with engine.begin() as conn:
            cid_row = conn.execute(text("SELECT chat_id FROM files WHERE id = :fid"), {"fid": file_id}).fetchone()
            if not cid_row: return {"status": "already deleted"}
            cid = cid_row[0]
            
            chat_root = STORAGE_DIR / f"chat_{cid}"
            filters = ["f.chat_id = :cid", "a.category = :cat"]
            params = {"cid": cid, "cat": category}
            dir_to_del = chat_root / category
            
            if subject:
                filters.append("a.subject = :sub")
                params["sub"] = subject
                dir_to_del = dir_to_del / subject
            if subcategory:
                filters.append("a.subcategory = :subcat")
                params["subcat"] = subcategory
                dir_to_del = dir_to_del / subcategory

            conn.execute(
                text(f"DELETE FROM attachments WHERE id IN (SELECT a.id FROM attachments a JOIN files f ON a.file_id = f.id WHERE {' AND '.join(filters)})"),
                params
            )
            if dir_to_del.exists() and dir_to_del.is_dir():
                # remove only the targeted folder tree; do not climb up and
                # delete empty parents because user explicitly asked for this
                # folder.  Keeping the parent (even if empty) avoids surprise
                # when deleting a sub‑folder.
                shutil.rmtree(str(dir_to_del))
                # note: no cleanup of parent directories here
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{file_id}/attachments/{attachment_id}")
def download_attachment(file_id: int, attachment_id: int, redirect: bool = True):
    import logging
    logger = logging.getLogger(__name__)
    try:
        # Try with supabase_key column
        try:
            with engine.begin() as conn:
                row = conn.execute(
                    text("""
                        SELECT a.storage_path, a.original_name, f.chat_id, a.supabase_key
                        FROM attachments a
                        JOIN files f ON a.file_id = f.id
                        WHERE a.id = :aid
                    """),
                    {"aid": attachment_id},
                ).fetchone()
        except Exception:
            # supabase_key column may not exist yet — fall back without it
            with engine.connect() as conn:
                row = conn.execute(
                    text("""
                        SELECT a.storage_path, a.original_name, f.chat_id
                        FROM attachments a
                        JOIN files f ON a.file_id = f.id
                        WHERE a.id = :aid
                    """),
                    {"aid": attachment_id},
                ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Attachment not found")

        storage_path = row[0]
        original_name = row[1]
        chat_id = row[2]
        supabase_key = row[3] if len(row) > 3 else None

        # ── Try Supabase first ────────────────────────────────
        if supabase_key:
            from services.storage import get_signed_url
            signed_url = get_signed_url(supabase_key, expires_in=3600)
            if signed_url:
                if not redirect:
                    # Frontend fetch mode — return URL as JSON so the frontend
                    # can window.open() it without popup-blocker issues
                    return {"url": signed_url, "filename": original_name}
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=signed_url)
            else:
                logger.warning("Signed URL generation failed for supabase_key=%s, falling back to disk", supabase_key)

        # ── Fall back to local disk ───────────────────────────
        file_path = STORAGE_DIR / f"chat_{chat_id}" / storage_path
        if not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="File not found — it may not have been uploaded to cloud storage yet. Please re-upload the chat export.",
            )
        import mimetypes
        mime, _ = mimetypes.guess_type(str(file_path))
        if mime is None:
            mime = "application/octet-stream"
        headers = {"Content-Disposition": f"inline; filename=\"{original_name}\""}
        return FileResponse(file_path, media_type=mime, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("download_attachment failed aid=%d: %s", attachment_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/{file_id}/custom-upload")
async def custom_upload(
    file_id: int,
    file: UploadFile = File(...),
    category: str = Form(...),
    subject: str = Form(...),
    subcategory: str = Form(""),
):
    """Upload any file directly to a chosen location in the vault structure."""
    from services.ingestion import _unique_dest
    import re

    # Sanitize path components - strip traversal characters
    def _safe(s: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', '_', s.strip()) or "Unknown"

    category = _safe(category)
    subject = _safe(subject)
    subcategory = _safe(subcategory) if subcategory.strip() else None

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, chat_id FROM files WHERE id = :fid"),
            {"fid": file_id},
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Upload session not found")
        db_file_id, chat_id = row.id, row.chat_id

    chat_root = STORAGE_DIR / f"chat_{chat_id}"
    dest_dir = chat_root / category / subject
    if subcategory:
        dest_dir = dest_dir / subcategory
    dest_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename).name or "upload"
    dest_path = _unique_dest(dest_dir, original_name)

    # Stream-save the file
    h = hashlib.sha256()
    size = 0
    with open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            h.update(chunk)
            out.write(chunk)

    file_hash = h.hexdigest()
    storage_rel = dest_path.relative_to(chat_root)

    with engine.begin() as conn:
        aid = conn.execute(
            text("""
                INSERT INTO attachments
                    (file_id, original_name, storage_path, category, subject,
                     subcategory, classification_method, file_hash, size_bytes)
                VALUES
                    (:fid, :name, :path, :cat, :sub, :subcat, 'manual', :fhash, :sz)
                ON CONFLICT (file_hash) DO UPDATE SET
                    file_id = EXCLUDED.file_id,
                    storage_path = EXCLUDED.storage_path,
                    category = EXCLUDED.category,
                    subject = EXCLUDED.subject,
                    subcategory = EXCLUDED.subcategory
                RETURNING id
            """),
            {
                "fid": db_file_id, "name": original_name,
                "path": str(storage_rel), "cat": category,
                "sub": subject, "subcat": subcategory,
                "fhash": file_hash, "sz": size,
            },
        ).scalar()

        # update subjects registry
        conn.execute(
            text("""
                INSERT INTO subjects (file_id, name, file_count)
                VALUES (:fid, :name, 1)
                ON CONFLICT (file_id, name)
                DO UPDATE SET file_count = subjects.file_count + 1
            """),
            {"fid": db_file_id, "name": subject},
        )

    return {"attachment_id": aid, "path": str(storage_rel)}
