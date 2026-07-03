import hashlib
import os
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import text

from db.connection import get_engine
from models.config import UPLOAD_DIR, MAX_UPLOAD_BYTES
from services.ingestion import ingest_zip

router = APIRouter(prefix="/upload", tags=["Upload"])
engine = get_engine()


@router.get("/list")
def list_uploads():
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id AS file_id, filename, chat_name, size, uploaded_at,
                   (SELECT COUNT(*) FROM messages WHERE file_id = files.id) as message_count,
                   (SELECT COUNT(*) FROM attachments WHERE file_id = files.id) as file_count,
                   COALESCE((SELECT source_type FROM chats WHERE id = files.chat_id), 'telegram') as source_type
            FROM files
            ORDER BY uploaded_at DESC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


@router.post("")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only Telegram export ZIP files are supported")

    file_uuid = str(uuid4())
    zip_path = UPLOAD_DIR / f"{file_uuid}.zip"
    hash_obj = hashlib.sha256()
    upload_size = 0

    # Stream-save and hash simultaneously
    with open(zip_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            upload_size += len(chunk)
            if upload_size > MAX_UPLOAD_BYTES:
                zip_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Upload exceeds size limit")
            hash_obj.update(chunk)
            f.write(chunk)

    file_hash = hash_obj.hexdigest()

    # Exact duplicate check — same bytes already processed
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM files WHERE file_hash = :h"), {"h": file_hash}
        ).scalar()
    if existing:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="This exact export has already been uploaded (no new content detected)")

    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is not a valid ZIP")

    # Run full ingestion pipeline — file record creation happens inside
    # so it can merge into the existing record for the same chat
    result = ingest_zip(
        zip_path=zip_path,
        zip_uuid=file_uuid,
        filename=file.filename,
        file_hash=file_hash,
        file_size=zip_path.stat().st_size,
        engine=engine,
    )

    return {
        "file_id": result.pop("db_file_id"),
        "status": "Upload successful",
        **result,
    }
