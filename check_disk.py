from sqlalchemy import create_engine, text
from pathlib import Path
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
STORAGE_DIR = Path("data/storage")
with engine.connect() as conn:
    row = conn.execute(text("SELECT a.id, a.storage_path, f.chat_id FROM attachments a JOIN files f ON a.file_id = f.id ORDER BY a.id DESC LIMIT 1")).fetchone()
    if row:
        path = STORAGE_DIR / f"chat_{row[2]}" / row[1]
        print(f"File ID: {row[0]}, Path: {path}, Exists: {path.exists()}")
    else:
        print("No attachments found.")
