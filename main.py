from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from routers import upload, search, messages, rag
from models.config import UPLOAD_DIR, STORAGE_DIR

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Telegram Academic Organizer", version="2.0")

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

@app.get("/")
def root():
    return FileResponse(static_dir / "landing.html")

@app.get("/app")
def app_ui():
    return FileResponse(static_dir / "index.html")
