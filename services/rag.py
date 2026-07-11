"""
RAG (Retrieval-Augmented Generation) service for StudyVault.

Handles:
1. PDF text extraction and chunking
2. Embedding chunks with sentence-transformers
3. Storing/retrieving embeddings
4. Question answering via Groq LLM with retrieved context
"""

import logging
import os
from typing import Optional

import numpy as np
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks
TOP_K_DEFAULT = 6      # number of chunks to retrieve
EMBED_DIM = 384        # all-MiniLM-L6-v2 output dimension


# ── 1. PDF Text Extraction ────────────────────────────────────

def extract_pdf_text(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, page by page.
    Falls back to OCR (Tesseract) if native text extraction fails.
    Returns list of {page_number, text}.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.error("PyMuPDF, pytesseract, or Pillow not installed — cannot extract PDF text")
        return []

    pages = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text").strip()
            
            # Fallback to OCR if less than 20 characters were extracted
            if len(page_text) < 20:
                try:
                    # Convert PDF page to an image (2x zoom for better OCR resolution)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Run Tesseract OCR
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        page_text = ocr_text
                        logger.info("Used OCR for page %d of %s", page_num + 1, file_path)
                except Exception as ocr_err:
                    logger.warning("OCR failed on page %d of %s: %s", page_num + 1, file_path, ocr_err)

            # PostgreSQL does not allow NUL characters in TEXT fields
            page_text = page_text.replace("\x00", "")
            if page_text:
                pages.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                })
        doc.close()
        logger.info("Extracted text from %d pages of %s", len(pages), file_path)
    except Exception as e:
        logger.warning("Failed to extract PDF text from %s: %s", file_path, e)

    return pages


# ── 2. Chunking ───────────────────────────────────────────────

def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split page texts into overlapping chunks.
    Returns list of {content, page_number, chunk_index}.
    """
    chunks = []
    idx = 0

    for page in pages:
        text_content = page["text"]
        page_num = page["page_number"]
        start = 0

        while start < len(text_content):
            end = start + chunk_size
            chunk_text = text_content[start:end].strip()

            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "page_number": page_num,
                    "chunk_index": idx,
                })
                idx += 1

            start += chunk_size - overlap

    return chunks


# ── 3. Embedding (via HF Inference API) ──────────────────────

def embed_texts(texts: list[str]) -> Optional[np.ndarray]:
    """Embed a list of texts via HF API, returns (N, 384) array or None."""
    from services.semantic_search import _embed_via_hf_api
    embeddings = _embed_via_hf_api(texts)
    if embeddings is None:
        logger.warning("HF Inference API unavailable — skipping indexing")
        return None
    return np.asarray(embeddings, dtype=np.float64)


# ── 4. Storage ────────────────────────────────────────────────

def index_attachment(engine, attachment_id: int, file_id: int,
                     source_name: str, file_path: str) -> int:
    """
    Full pipeline: extract PDF → chunk → embed → store.
    If file is missing on disk but has a supabase_key, downloads it first.
    Returns number of chunks indexed.
    """
    # Check if already indexed
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE attachment_id = :aid"),
            {"aid": attachment_id},
        ).scalar()
        if existing > 0:
            logger.info("Attachment %d already indexed (%d chunks), skipping",
                        attachment_id, existing)
            return existing

    # Resolve file path — download from Supabase if not on disk
    resolved_path = file_path
    tmp_path = None

    if not os.path.isfile(file_path):
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT supabase_key FROM attachments WHERE id = :aid"),
                {"aid": attachment_id},
            ).fetchone()
        supabase_key = row.supabase_key if row else None

        if supabase_key:
            from services.storage import download_to_temp
            suffix = os.path.splitext(file_path)[1] or ".pdf"
            tmp_path = download_to_temp(supabase_key, suffix=suffix)
            if tmp_path:
                resolved_path = str(tmp_path)
                logger.info("Downloaded %s from Supabase for indexing", source_name)
            else:
                logger.warning("Could not download %s from Supabase — skipping", source_name)
                return 0
        else:
            logger.info("File not on disk and no supabase_key — skipping %s", source_name)
            return 0

    try:
        # Extract text
        pages = extract_pdf_text(resolved_path)
        if not pages:
            logger.info("No text extracted from %s — skipping indexing", source_name)
            return 0

        # Chunk
        chunks = chunk_pages(pages)
        if not chunks:
            return 0

        # Embed
        chunk_texts = [c["content"] for c in chunks]
        embeddings = embed_texts(chunk_texts)
        if embeddings is None:
            return 0

        # Store
        with engine.begin() as conn:
            for chunk, emb_vec in zip(chunks, embeddings):
                emb_list = emb_vec.tolist()
                conn.execute(
                    text("""
                        INSERT INTO document_chunks
                            (attachment_id, file_id, chunk_index, content,
                             embedding, page_number, source_name)
                        VALUES
                            (:aid, :fid, :cidx, :content,
                             :embedding, :page, :sname)
                    """),
                    {
                        "aid": attachment_id,
                        "fid": file_id,
                        "cidx": chunk["chunk_index"],
                        "content": chunk["content"],
                        "embedding": emb_list,
                        "page": chunk["page_number"],
                        "sname": source_name,
                    },
                )

        logger.info("Indexed %d chunks for attachment %d (%s)",
                    len(chunks), attachment_id, source_name)
        return len(chunks)
    finally:
        # Clean up temp file if we downloaded one
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_index_status(engine, file_id: int) -> dict:
    """Get indexing status for a file upload (scoped to the entire chat)."""
    with engine.begin() as conn:
        # Resolve chat_id to scope across all files in the same chat
        chat_id = conn.execute(
            text("SELECT chat_id FROM files WHERE id = :fid"),
            {"fid": file_id},
        ).scalar()

        if chat_id:
            # Count across ALL file_ids in this chat
            total_attachments = conn.execute(
                text("""
                    SELECT COUNT(*) FROM attachments a
                    JOIN files f ON a.file_id = f.id
                    WHERE f.chat_id = :cid
                      AND a.original_name ILIKE '%%.pdf'
                """),
                {"cid": chat_id},
            ).scalar() or 0

            indexed_attachments = conn.execute(
                text("""
                    SELECT COUNT(DISTINCT dc.attachment_id) FROM document_chunks dc
                    JOIN files f ON dc.file_id = f.id
                    WHERE f.chat_id = :cid
                """),
                {"cid": chat_id},
            ).scalar() or 0

            total_chunks = conn.execute(
                text("""
                    SELECT COUNT(*) FROM document_chunks dc
                    JOIN files f ON dc.file_id = f.id
                    WHERE f.chat_id = :cid
                """),
                {"cid": chat_id},
            ).scalar() or 0
        else:
            # Fallback to single file_id
            total_attachments = conn.execute(
                text("""
                    SELECT COUNT(*) FROM attachments
                    WHERE file_id = :fid
                      AND original_name ILIKE '%%.pdf'
                """),
                {"fid": file_id},
            ).scalar() or 0

            indexed_attachments = conn.execute(
                text("""
                    SELECT COUNT(DISTINCT attachment_id) FROM document_chunks
                    WHERE file_id = :fid
                """),
                {"fid": file_id},
            ).scalar() or 0

            total_chunks = conn.execute(
                text("SELECT COUNT(*) FROM document_chunks WHERE file_id = :fid"),
                {"fid": file_id},
            ).scalar() or 0

    return {
        "total_pdf_attachments": total_attachments,
        "indexed_attachments": indexed_attachments,
        "total_chunks": total_chunks,
        "fully_indexed": total_attachments > 0 and total_attachments == indexed_attachments,
    }


# ── 5. Retrieval ──────────────────────────────────────────────

def _cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of doc vectors."""
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    denom = np.maximum(doc_norms * max(query_norm, 1e-12), 1e-12)
    return (doc_matrix @ query_vec) / denom


def retrieve_chunks(engine, question: str, file_id: int,
                    top_k: int = TOP_K_DEFAULT) -> list[dict]:
    """
    Retrieve the most relevant chunks for a question.
    Returns list of {content, page_number, source_name, score}.
    """
    # Embed the question via HF API
    from services.semantic_search import _embed_via_hf_api
    q_emb = _embed_via_hf_api([question])
    if q_emb is None:
        return []
    query_vec = np.asarray(q_emb[0], dtype=np.float64)

    # Fetch all chunks for this chat (across all file_ids)
    with engine.begin() as conn:
        chat_id = conn.execute(
            text("SELECT chat_id FROM files WHERE id = :fid"),
            {"fid": file_id},
        ).scalar()

        if chat_id:
            rows = conn.execute(
                text("""
                    SELECT dc.id, dc.content, dc.embedding, dc.page_number,
                           dc.source_name, dc.attachment_id
                    FROM document_chunks dc
                    JOIN files f ON dc.file_id = f.id
                    WHERE f.chat_id = :cid
                """),
                {"cid": chat_id},
            ).fetchall()
        else:
            rows = conn.execute(
                text("""
                    SELECT id, content, embedding, page_number, source_name,
                           attachment_id
                    FROM document_chunks
                    WHERE file_id = :fid
                """),
                {"fid": file_id},
            ).fetchall()

    if not rows:
        return []

    # Build embedding matrix
    doc_vecs = []
    valid_rows = []
    for row in rows:
        emb = row.embedding
        if emb and len(emb) == EMBED_DIM:
            doc_vecs.append(emb)
            valid_rows.append(row)

    if not doc_vecs:
        return []

    doc_matrix = np.array(doc_vecs, dtype=np.float64)
    scores = _cosine_similarity(query_vec, doc_matrix)

    # Get top-K
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = valid_rows[idx]
        results.append({
            "content": row.content,
            "page_number": row.page_number,
            "source_name": row.source_name,
            "attachment_id": row.attachment_id,
            "score": float(scores[idx]),
        })

    return results


import json

# ── 6. Agent Tools ───────────────────────────────────────────

def search_files(engine, file_id: int, category: str = None,
                 subject: str = None, keyword: str = None) -> list[dict]:
    """Search for files matching the given criteria with fuzzy matching."""
    conditions = ["file_id = :fid"]
    params = {"fid": file_id}

    if category:
        conditions.append(
            "(category ILIKE :cat OR subcategory ILIKE :cat OR storage_path ILIKE :cat)"
        )
        params["cat"] = f"%{category}%"
    if subject:
        conditions.append(
            "(subject ILIKE :sub OR storage_path ILIKE :sub OR original_name ILIKE :sub)"
        )
        params["sub"] = f"%{subject}%"

    query = "SELECT id, original_name, category, subject, subcategory, storage_path, size_bytes FROM attachments WHERE " + " AND ".join(conditions)

    with engine.begin() as conn:
        rows = conn.execute(text(query), params).fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r.id,
            "filename": r.original_name,
            "category": r.category,
            "subject": r.subject,
            "subcategory": r.subcategory,
            "storage_path": r.storage_path,
            "size_bytes": r.size_bytes
        })
        
    if keyword:
        import difflib
        import re
        
        # Remove common file extension words from the keyword for better fuzzy matching
        kw_clean = re.sub(r'\b(pdf|doc|docx|file|document)\b', '', keyword.lower()).strip()
        if not kw_clean:
            kw_clean = keyword.lower()
            
        kw_words = kw_clean.split()
        
        scored_results = []
        for res in results:
            text_to_search = f"{res['filename']} {res['category']} {res['subject']} {res['storage_path']}".lower()
            
            score = 0.0
            # 1. Exact substring match
            if kw_clean in text_to_search:
                score = 1.0
            # 2. All words present in any order
            elif kw_words and all(w in text_to_search for w in kw_words):
                score = 0.9
            # 3. Fuzzy typo match using difflib
            else:
                score_name = difflib.SequenceMatcher(None, kw_clean, res['filename'].lower()).ratio()
                score_subj = difflib.SequenceMatcher(None, kw_clean, (res['subject'] or "").lower()).ratio()
                score_cat = difflib.SequenceMatcher(None, kw_clean, (res['category'] or "").lower()).ratio()
                score = max(score_name, score_subj, score_cat)
            
            if score >= 0.4:  # Threshold for fuzzy match
                scored_results.append((score, res))
                
        # Sort by best match score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        results = [x[1] for x in scored_results]

    return results[:30]


def list_categories(engine, file_id: int) -> dict:
    """List all available categories and subjects in the database."""
    with engine.begin() as conn:
        cats = conn.execute(text(
            "SELECT DISTINCT category FROM attachments WHERE file_id = :fid AND category IS NOT NULL ORDER BY category"
        ), {"fid": file_id}).fetchall()
        subs = conn.execute(text(
            "SELECT DISTINCT subject FROM attachments WHERE file_id = :fid AND subject IS NOT NULL ORDER BY subject"
        ), {"fid": file_id}).fetchall()

    return {
        "categories": [r.category for r in cats],
        "subjects": [r.subject for r in subs],
    }


def _build_system_prompt(engine, file_id: int) -> str:
    """Build a rich system prompt with available categories and subjects."""
    meta = list_categories(engine, file_id)
    cats_str = ", ".join(meta["categories"]) if meta["categories"] else "none"
    subs_str = ", ".join(meta["subjects"]) if meta["subjects"] else "none"

    return f"""You are StudyVault AI — a smart academic assistant that helps students find and understand their documents.

You have access to these tools:
1. **search_files** — Search for files by category, subject, or keyword. Use this when the user asks to FIND, LIST, SHOW, RETURN, or GET files/documents/notes/pdfs.
2. **search_document_contents** — Semantically search INSIDE PDF text to answer factual questions. Use this when the user asks ABOUT the content of documents (e.g. "what does module 3 cover?").
3. **list_categories** — List all available categories and subjects. Use this if you are unsure what category or subject to search for.

AVAILABLE CATEGORIES: {cats_str}
AVAILABLE SUBJECTS: {subs_str}

ROUTING RULES:
- CRITICAL: DO NOT use a `category` or `subject` argument unless it EXACTLY matches one of the items in the AVAILABLE lists above.
- If the user asks for something (like "assignments" or "attendance") but it is NOT in the AVAILABLE lists, you MUST use the `keyword` argument instead (e.g., `keyword="assignments"`).
- Map partial queries to the closest AVAILABLE item (e.g., if user asks for "data analysis" and "Data Analytics with Excel" is available, use `subject="Data Analytics with Excel"`).
- IGNORE generic words like "pdf", "file", or "document" when formulating your search keyword.
- Example: "What is covered in Module 3?" → search_document_contents(query="Module 3 topics")
- Example: "What is the formula for..." → search_document_contents(query="formula for...")

RESPONSE FORMATTING:
- When listing files, format them as a numbered list with the filename and category/subject.
- When answering factual questions, cite which document and page the answer came from.
- Be concise and helpful. Use bullet points for lists.
- If no results are found, suggest alternative search terms."""


# ── 7. Agent Loop ────────────────────────────────────────────

def _parse_action(text_content: str) -> Optional[dict]:
    """Try to extract a JSON action block from the LLM response."""
    import re
    # Look for ```json ... ``` or ```action ... ``` blocks
    patterns = [
        r'```(?:json|action)\s*\n(.*?)\n```',
        r'```\s*\n(\{.*?\})\n```',
        r'(\{"action"\s*:.*?\})',
    ]
    for pat in patterns:
        m = re.search(pat, text_content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


def ask(engine, question: str, file_id: int,
        top_k: int = TOP_K_DEFAULT) -> dict:
    """
    Agentic Chatbot: Uses manual JSON parsing for tool routing
    (avoids Groq native tool-calling bugs).
    """
    if not GROQ_API_KEY:
        return {
            "answer": "AI answering is unavailable — GROQ_API_KEY not configured.",
            "sources": [],
            "chunks_used": 0,
        }

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        return {"answer": f"Error loading Groq client: {e}", "sources": [], "chunks_used": 0}

    system_prompt = _build_system_prompt(engine, file_id)

    # Replace the tool-calling instructions with manual JSON action format
    action_instructions = """
/nothink

IMPORTANT — HOW TO USE TOOLS:
When you need to use a tool, output ONLY a JSON block like this (no other text):

To search for files:
```json
{"action": "search_files", "category": "Notes", "subject": "Mathematics", "keyword": ""}
```

To search inside document contents:
```json
{"action": "search_document_contents", "query": "your search query here"}
```

To list available categories:
```json
{"action": "list_categories"}
```

RULES:
- Output ONLY the JSON block when calling a tool. No extra text before or after.
- After receiving tool results, write your final answer in plain text (no JSON).
- category, subject, and keyword are all optional for search_files — use only the ones that are relevant.
"""

    messages = [
        {"role": "system", "content": system_prompt + action_instructions},
        {"role": "user", "content": question}
    ]

    final_answer = ""
    sources_to_return = []
    chunks_used = 0

    try:
        for iteration in range(4):
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )

            reply = response.choices[0].message.content or ""
            # Strip Qwen3 chain-of-thought think blocks if present
            import re as _re
            reply = _re.sub(r'<think>.*?</think>', '', reply, flags=_re.DOTALL).strip()
            messages.append({"role": "assistant", "content": reply})

            # Try to parse a tool action from the reply
            action = _parse_action(reply)

            if action and "action" in action:
                action_name = action["action"]

                if action_name == "search_files":
                    found_files = search_files(
                        engine, file_id,
                        category=action.get("category"),
                        subject=action.get("subject"),
                        keyword=action.get("keyword")
                    )
                    if found_files:
                        tool_result = f"Found {len(found_files)} files:\n"
                        for i, f in enumerate(found_files, 1):
                            tool_result += f"{i}. {f['filename']} (Category: {f['category']}, Subject: {f['subject']})\n"
                            sources_to_return.append({
                                "filename": f["filename"],
                                "page": "N/A",
                                "attachment_id": f["id"],
                                "snippet": f"Category: {f['category']}, Subject: {f['subject']}"
                            })
                    else:
                        tool_result = "No files found matching the criteria. Try broadening your search."

                elif action_name == "search_document_contents":
                    query = action.get("query", question)
                    chunks_list = retrieve_chunks(engine, query, file_id, top_k)
                    chunks_used += len(chunks_list)

                    context_parts = []
                    for i, c in enumerate(chunks_list, 1):
                        source_label = f"{c['source_name']} (page {c['page_number']})"
                        context_parts.append(f"[Source {i}: {source_label}]\n{c['content']}")
                        sources_to_return.append({
                            "filename": c["source_name"],
                            "page": c["page_number"],
                            "attachment_id": c.get("attachment_id"),
                            "snippet": c["content"][:120] + "..." if len(c["content"]) > 120 else c["content"],
                        })
                    tool_result = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found in documents."

                elif action_name == "list_categories":
                    meta = list_categories(engine, file_id)
                    tool_result = json.dumps(meta, indent=2)

                else:
                    tool_result = "Unknown action."

                # Feed tool results back to the LLM
                if iteration >= 2:
                    instruction = "This is your LAST step. Do NOT output JSON or call tools. You MUST write a helpful plain-text response summarizing the results or lack thereof."
                else:
                    instruction = "Write your final answer to the user based on these results (plain text, NO JSON), OR you may call another tool if you still need more information."
                    
                messages.append({
                    "role": "user",
                    "content": f"TOOL RESULTS:\n{tool_result}\n\n{instruction}"
                })
            else:
                # No action block found — this is the final answer
                final_answer = reply
                break
        
        # If we exhausted iterations and still have no final answer, use the last reply
        if not final_answer and reply:
            final_answer = reply

    except Exception as e:
        logger.error("Groq Agent failed: %s", e)
        final_answer = f"Sorry, I encountered an error generating the answer: {str(e)}"

    # Deduplicate sources
    unique_sources = []
    seen = set()
    for s in sources_to_return:
        key = (s["filename"], s.get("page", ""))
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "answer": final_answer or "I couldn't generate an answer.",
        "sources": unique_sources,
        "chunks_used": chunks_used,
    }


