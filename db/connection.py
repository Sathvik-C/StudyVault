import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

DATABASE_URL_ENV = os.getenv("DATABASE_URL")

if DATABASE_URL_ENV:
    DATABASE_URL = DATABASE_URL_ENV
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

engine = create_engine(DATABASE_URL)


def get_engine():
    return engine
