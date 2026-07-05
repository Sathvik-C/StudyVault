"""
services/storage.py — Supabase Storage wrapper for StudyVault.

Handles:
- Uploading files to Supabase Storage
- Generating signed download URLs
- Downloading files from Supabase to a local temp path (for RAG indexing)

Falls back gracefully if SUPABASE_URL / SUPABASE_KEY are not set
(local dev without Supabase configured).
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
BUCKET = "studyvault-files"

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _client
    except Exception as e:
        logger.error("Failed to init Supabase client: %s", e)
        return None


def is_available() -> bool:
    """Returns True if Supabase storage is configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


def upload_file(local_path: Path, storage_key: str) -> bool:
    """
    Upload a file to Supabase Storage.
    storage_key is the path within the bucket e.g. "chat_1/Notes/General/file.pdf"
    Returns True on success, False on failure.
    """
    client = _get_client()
    if not client:
        logger.warning("Supabase not configured — skipping upload of %s", local_path)
        return False

    try:
        import mimetypes
        mime, _ = mimetypes.guess_type(str(local_path))
        mime = mime or "application/octet-stream"

        with open(local_path, "rb") as f:
            data = f.read()

        # upsert=True overwrites if key already exists (re-uploads)
        client.storage.from_(BUCKET).upload(
            path=storage_key,
            file=data,
            file_options={"content-type": mime, "upsert": "true"},
        )
        logger.debug("Uploaded %s → supabase://%s/%s", local_path.name, BUCKET, storage_key)
        return True
    except Exception as e:
        logger.error("Supabase upload failed for %s: %s", storage_key, e)
        return False


def get_signed_url(storage_key: str, expires_in: int = 3600) -> Optional[str]:
    """
    Generate a signed URL for a file in Supabase Storage.
    expires_in: seconds until expiry (default 1 hour).
    Returns the URL string or None on failure.
    """
    client = _get_client()
    if not client:
        return None

    try:
        res = client.storage.from_(BUCKET).create_signed_url(
            path=storage_key,
            expires_in=expires_in,
        )
        return res.get("signedURL") or res.get("signed_url")
    except Exception as e:
        logger.error("Failed to create signed URL for %s: %s", storage_key, e)
        return None


def download_to_temp(storage_key: str, suffix: str = "") -> Optional[Path]:
    """
    Download a file from Supabase Storage to a local temp file.
    Returns the Path to the temp file, or None on failure.
    Caller is responsible for deleting the temp file after use.
    """
    client = _get_client()
    if not client:
        return None

    try:
        data = client.storage.from_(BUCKET).download(storage_key)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        logger.debug("Downloaded supabase://%s/%s → %s", BUCKET, storage_key, tmp.name)
        return Path(tmp.name)
    except Exception as e:
        logger.error("Supabase download failed for %s: %s", storage_key, e)
        return None


def delete_file(storage_key: str) -> bool:
    """Delete a file from Supabase Storage."""
    client = _get_client()
    if not client:
        return False
    try:
        client.storage.from_(BUCKET).remove([storage_key])
        return True
    except Exception as e:
        logger.error("Supabase delete failed for %s: %s", storage_key, e)
        return False
