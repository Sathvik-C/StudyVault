from sqlalchemy import create_engine, text
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    res = conn.execute(text("SELECT id, original_name, storage_path FROM attachments LIMIT 5")).fetchall()
    print("Attachments:", res)
