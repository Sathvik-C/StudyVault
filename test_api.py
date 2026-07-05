from sqlalchemy import create_engine, text
from pathlib import Path
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
with engine.begin() as conn:
    cid = conn.execute(text("SELECT chat_id FROM files WHERE id = 5")).scalar()
    print(f"cid: {cid}")
    rows = conn.execute(text("""
        SELECT a.id, a.original_name, a.storage_path, a.category, a.subject,
               a.subcategory, a.classification_method, a.size_bytes, a.file_id,
               a.note, a.tags
        FROM attachments a
        JOIN files f ON a.file_id = f.id
        WHERE f.chat_id = :cid
    """), {"cid": cid}).fetchall()
    print(f"attachments: {len(rows)}")
