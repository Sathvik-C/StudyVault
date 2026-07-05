from dotenv import load_dotenv
load_dotenv()

import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from routers import upload, search, messages, rag
from models.config import UPLOAD_DIR, STORAGE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="StudyVault", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(search.router)
app.include_router(messages.router)
app.include_router(rag.router)

# Serve static UI
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def on_startup():
    """Run database migrations on startup."""
    try:
        from db.connection import get_engine
        from db.auto_migrate import run_migrations
        run_migrations(get_engine())
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error("Migration failed: %s", e)
        # Don't crash the app — it might still work if tables already exist


@app.get("/health")
def health_check():
    """Health check endpoint for Render."""
    return {"status": "healthy", "service": "StudyVault"}


@app.get("/debug/storage")
def debug_storage():
    """Check Supabase storage configuration status."""
    from services.storage import is_available, _get_client, _get_supabase_url
    url = _get_supabase_url()
    available = is_available()
    client_ok = False
    error = None
    if available:
        try:
            client = _get_client()
            if client:
                buckets = client.storage.list_buckets()
                client_ok = True
        except Exception as e:
            error = str(e)
    return {
        "supabase_url_set": bool(url),
        "supabase_url_preview": url[:30] + "..." if url else None,
        "supabase_key_set": bool(os.getenv("SUPABASE_KEY")),
        "client_ok": client_ok,
        "error": error,
    }


@app.get("/")
def root():
    return FileResponse(static_dir / "landing.html")

@app.get("/app")
def app_ui():
    return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
