from sqlalchemy import create_engine, text
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    msgs = conn.execute(text("SELECT count(*) FROM messages WHERE file_id = 4")).scalar()
    print("Messages for file 4:", msgs)
