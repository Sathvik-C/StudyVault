from sqlalchemy import create_engine, text
from pathlib import Path
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    row = conn.execute(text("SELECT id, chat_id, chat_name FROM files WHERE id = 5")).fetchone()
    print(f"File ID 5: {row}")
