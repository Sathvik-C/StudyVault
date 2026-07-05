from sqlalchemy import create_engine, text
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    atts = conn.execute(text("SELECT COUNT(*) FROM attachments")).scalar()
    print("Attachments in DB:", atts)
    files = conn.execute(text("SELECT id, status FROM files ORDER BY id DESC LIMIT 5")).fetchall()
    print("Files in DB:", files)
