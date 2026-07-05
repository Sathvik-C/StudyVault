from sqlalchemy import create_engine, text
from pathlib import Path
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    print("Latest Files:")
    files = conn.execute(text("SELECT id, filename FROM files ORDER BY id DESC LIMIT 3")).fetchall()
    for f in files:
        count = conn.execute(text("SELECT COUNT(*) FROM attachments WHERE file_id = :fid"), {"fid": f.id}).scalar()
        print(f"File ID {f.id}: {f.filename} -> {count} attachments")
