import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

logger = logging.getLogger(__name__)

DATABASE_URL_ENV = os.getenv("DATABASE_URL")

if DATABASE_URL_ENV:
    # Render (and some providers) use "postgres://" but SQLAlchemy
    # requires "postgresql://". Fix it automatically.
    DATABASE_URL = DATABASE_URL_ENV
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        logger.info("Converted postgres:// → postgresql:// in DATABASE_URL")
else:
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "chatfiles")

    if db_password is None:
        raise RuntimeError("DB_PASSWORD is required when DATABASE_URL is not set")

    DATABASE_URL = URL.create(
        drivername="postgresql",
        username=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        database=db_name,
    )

# Small pool for free tier (limited connections)
engine = create_engine(
    DATABASE_URL,
    pool_size=3,
    max_overflow=2,
    pool_timeout=30,
    pool_recycle=300,
    pool_pre_ping=True,
)


def get_engine():
    return engine
