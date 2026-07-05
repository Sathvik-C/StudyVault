from sqlalchemy import create_engine, text
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    imp = conn.execute(text("SELECT COUNT(*) FROM important_messages WHERE file_id = 4")).scalar()
    print("Important messages:", imp)
