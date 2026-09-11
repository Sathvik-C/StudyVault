# 📚 StudyVault

**StudyVault** is a self-hosted academic assistant that turns raw Telegram or WhatsApp group exports into a searchable, AI-powered knowledge base. Upload a chat export ZIP and instantly get:

- AI-classified attachments (notes, assignments, question banks, etc.)
- Semantic search across messages and files
- RAG-powered Q&A over indexed PDFs
- Important message detection

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-source ingestion** | Auto-detects Telegram (`result.json`) and WhatsApp (`_chat.txt`) exports |
| **Incremental uploads** | Re-uploading the same chat only processes new messages/files |
| **AI classification** | Groq LLM assigns every attachment a category, subject, and subcategory |
| **OCR fallback** | Tesseract OCR kicks in for scanned/image-heavy PDFs |
| **Semantic search** | HuggingFace `all-MiniLM-L6-v2` embeddings over messages and files |
| **RAG Q&A** | Ask natural language questions; Qwen3-27b retrieves relevant PDF chunks and answers |
| **Supabase Storage** | Optional persistent cloud storage so attachments survive container restarts |
| **One-click deploy** | Dockerfile + `render.yaml` for zero-config Render deployment |

---

## 🏗️ Architecture

```
StudyVault/
├── main.py                  # FastAPI app entry point
├── routers/
│   ├── upload.py            # POST /upload — ZIP ingestion & delete
│   ├── search.py            # GET  /search — keyword + semantic search
│   ├── messages.py          # GET  /messages — paginated chat history
│   └── rag.py               # POST /rag/{file_id}/ask — RAG Q&A
├── services/
│   ├── ingestion.py         # ZIP extraction, parsing, dedup, scheduling
│   ├── classifier.py        # Groq batch AI classification (15 files/call)
│   ├── semantic_search.py   # HF Inference API embeddings + cosine search
│   ├── rag.py               # PDF chunking, embedding, retrieval, agent loop
│   ├── important.py         # Important message detection
│   ├── storage.py           # Supabase Storage client (optional)
│   ├── file_extractor.py    # First-page text extraction for classifiers
│   └── whatsapp_parser.py   # WhatsApp _chat.txt parser
├── db/
│   ├── connection.py        # SQLAlchemy engine factory
│   ├── auto_migrate.py      # Auto-runs SQL migrations on startup
│   ├── migrate.sql          # Core schema migrations
│   └── rag_schema.sql       # document_chunks table for RAG
├── models/
│   └── config.py            # Env-based config (paths, limits)
└── static/                  # Served frontend UI
```

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.11+
- PostgreSQL database
- Tesseract OCR (`brew install tesseract` on macOS)

### 1. Clone & install

```bash
git clone https://github.com/Sathvik-C/StudyVault.git
cd StudyVault
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# Required
DATABASE_URL=postgresql://user:password@localhost:5432/studyvault
GROQ_API_KEY=your_groq_api_key_here

# Required for semantic search & RAG indexing
HF_API_TOKEN=your_hf_token_here

# Optional — enables persistent file storage
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-service-role-key

# Optional — upload size limits (bytes)
MAX_UPLOAD_BYTES=52428800       # 50 MB
MAX_EXTRACTED_BYTES=1073741824  # 1 GB
```

### 3. Run

```bash
python main.py
```

The app runs on `http://localhost:8000`. Database migrations are applied automatically on startup.

---

## 🐳 Docker

```bash
docker build -t studyvault .
docker run -p 8000:8000 --env-file .env studyvault
```

---

## ☁️ Deploy to Render

The repo includes a `render.yaml` for one-click Docker deployment:

1. Connect your GitHub repo to [Render](https://render.com).
2. Add the required environment variables in the Render dashboard:
   - `DATABASE_URL`
   - `GROQ_API_KEY`
   - `HF_API_TOKEN`
   - `SUPABASE_URL` *(optional)*
   - `SUPABASE_KEY` *(optional)*
3. Render will build from the Dockerfile and deploy automatically.

> **Note:** The free Render plan uses ephemeral disk storage. Set up Supabase Storage to persist uploaded attachments across deploys.

---

## 📡 API Reference

### Upload

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a Telegram/WhatsApp ZIP export |
| `GET` | `/upload/list` | List all uploaded chat exports |
| `DELETE` | `/upload/{file_id}` | Delete a chat export and all its data |

### Search

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?q=...&file_id=...` | Semantic (or keyword fallback) search over messages and files |

### Messages

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/messages?file_id=...` | Paginated chat message history |

### RAG Q&A

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/rag/{file_id}/ask` | Ask a natural language question about the documents |
| `GET` | `/rag/{file_id}/status` | Check how many PDFs have been indexed |
| `POST` | `/rag/{file_id}/index` | Manually trigger PDF indexing |
| `POST` | `/rag/{file_id}/backfill-storage` | Backfill existing files to Supabase Storage |

### Utility

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (used by Render) |
| `GET` | `/debug/storage` | Supabase Storage configuration status |

---

## 🤖 How RAG Works

1. **Index** — PDFs are extracted page-by-page (with OCR fallback), split into 500-character chunks with 80-character overlap, and embedded via the HuggingFace `all-MiniLM-L6-v2` model (384 dimensions). Embeddings are stored in PostgreSQL.
2. **Retrieve** — On a question, the query is embedded and cosine similarity is computed against all stored chunks. The top-6 chunks are retrieved.
3. **Generate** — An agentic loop (up to 4 iterations) runs with `qwen/qwen3.6-27b` on Groq. The agent can call three tools: `search_files`, `search_document_contents`, and `list_categories` before producing a final plain-text answer.

---

## 🗂️ AI Classification Categories

Files are automatically assigned one of these categories by the Groq classifier:

`Notes` · `Assignments` · `Question Bank` · `Lab` · `Results` · `Attendance` · `Schedules` · `Exams` · `Notices` · `Reports` · `Certificates` · `Events` · `Posters` · `Contacts` · `Documents`

Each file also gets a **subject** (e.g., Mathematics, Data Structures, Python) and **subcategory** inferred from the filename, surrounding chat context, and first-page extracted text.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn / Gunicorn |
| Database | PostgreSQL + SQLAlchemy |
| AI — Classification | Groq (`llama-3.3-70b` / `qwen3`) |
| AI — Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| AI — Q&A | Groq (`qwen/qwen3.6-27b`) |
| PDF Extraction | PyMuPDF + pdfplumber |
| OCR | Tesseract + Pillow |
| Cloud Storage | Supabase Storage (optional) |
| Containerisation | Docker (multi-stage, python:3.11-slim) |
| Deployment | Render |

---

## 📄 License

MIT
