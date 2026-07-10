"""
backfill_supabase.py — Upload all local files to Supabase and update Neon DB.
Run once: python3 backfill_supabase.py
"""
import os
import sys
import time
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv()

from db.connection import get_engine
from services.storage import upload_file, is_available
from models.config import STORAGE_DIR
from sqlalchemy import text

if not is_available():
    print("ERROR: SUPABASE_URL or SUPABASE_KEY not set in .env")
    sys.exit(1)

engine = get_engine()

# Fetch all attachments without a supabase_key
with engine.begin() as conn:
    rows = conn.execute(text("""
        SELECT a.id, a.storage_path, f.chat_id
        FROM attachments a
        JOIN files f ON a.file_id = f.id
        WHERE a.supabase_key IS NULL
    """)).fetchall()

print(f"Found {len(rows)} files to upload to Supabase...")

def upload_one(row):
    fp = STORAGE_DIR / f"chat_{row.chat_id}" / row.storage_path
    if not fp.is_file():
        return row.id, None, "missing"
    sb_key = f"chat_{row.chat_id}/{str(row.storage_path).replace(chr(92), '/')}"
    try:
        ok = upload_file(fp, sb_key)
        return row.id, sb_key if ok else None, "ok" if ok else "failed"
    except Exception as e:
        return row.id, None, str(e)

uploaded = 0
failed = 0
missing = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(upload_one, row): row for row in rows}
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        aid, sb_key, status = future.result()
        if sb_key:
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE attachments SET supabase_key = :k WHERE id = :aid"),
                    {"k": sb_key, "aid": aid}
                )
            uploaded += 1
        elif status == "missing":
            missing += 1
        else:
            failed += 1
            print(f"  FAILED aid={aid}: {status}")

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(rows)} — {uploaded} uploaded, {missing} missing, {failed} failed")

print(f"\nDone! {uploaded} uploaded, {missing} missing on disk, {failed} failed")
