import logging
import sys
from sqlalchemy import create_engine
from db.auto_migrate import run_migrations

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
DATABASE_URL = "postgresql://neondb_owner:npg_TZN8YPXBq2yD@ep-super-smoke-ao9uc27u.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
try:
    run_migrations(engine)
except Exception as e:
    print(f"FAILED: {e}")
