from sqlalchemy import create_engine
from services.rag import ask
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
res = ask(engine, "find notes for math", 4)
print(res)
