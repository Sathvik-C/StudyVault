from __future__ import annotations
"""
WhatsApp Android export parser.

Parses _chat.txt and matches media files from the ZIP.

WhatsApp Android format:
    18/02/2026, 10:23 am - John: Hello
    18/02/2026, 10:24 am - John: <Media omitted>
    18/02/2026, 10:24 am - John: notes.pdf (file attached)

Media files sit alongside _chat.txt in the ZIP.
"""

import re
import logging
from pathlib import Path
from datetime import datetime, timezone

def _parse_whatsapp_date(date_str: str):
    """Parse WhatsApp date strings into datetime objects."""
    formats = [
        "%d/%m/%Y %I:%M %p",    # 31/12/2024 7:53 pm
        "%d/%m/%Y %I:%M%p",     # 31/12/2024 7:53pm
        "%d/%m/%y %I:%M %p",    # 31/12/24 7:53 pm
        "%d/%m/%y %I:%M%p",     # 31/12/24 7:53pm
        "%d/%m/%Y %H:%M",       # 31/12/2024 19:53
        "%d/%m/%y %H:%M",       # 31/12/24 19:53
        "%m/%d/%Y %I:%M %p",    # 12/31/2024 7:53 pm
        "%m/%d/%Y %I:%M%p",     # 12/31/2024 7:53pm
        "%m/%d/%y %I:%M %p",    # 12/31/24 7:53 pm
        "%m/%d/%y %I:%M%p",     # 12/31/24 7:53pm
        "%m/%d/%Y %H:%M",       # 12/31/2024 19:53
        "%m/%d/%y %H:%M",       # 12/31/24 19:53
    ]
    # normalize narrow no-break space to regular space
    date_str = date_str.replace("\u202f", " ").replace("\xa0", " ").strip()
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.warning("Could not parse WhatsApp date: %s", date_str)
    return None

logger = logging.getLogger(__name__)

# Matches: "18/02/2026, 10:23 am - Sender: message"
# Also handles 24hr: "18/02/2026, 10:23 - Sender: message"
MESSAGE_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[ap]m)?)\s-\s([^:]+):\s(.*)$",
    re.IGNORECASE,
)

# Matches filenames at end of message like "notes.pdf (file attached)"
FILE_ATTACHMENT_PATTERN = re.compile(
    r"^(.+\.\w{2,5})\s*\(file attached\)$",
    re.IGNORECASE,
)

# WhatsApp puts this when media isn't exported
OMITTED_PATTERN = re.compile(r"<media omitted>", re.IGNORECASE)


def parse_whatsapp_chat(chat_txt_path: Path, extract_path: Path) -> list[dict]:
    """
    Parse _chat.txt and return list of message dicts compatible
    with the rest of the ingestion pipeline.

    Each dict has: role, content, order, file (Path or None), date
    """
    messages = []
    order = 0
    current_msg = None

    with open(chat_txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\n")
        match = MESSAGE_PATTERN.match(line)

        if match:
            # Save previous message
            if current_msg is not None:
                messages.append(_finalize(current_msg, extract_path, order))
                order += 1

            date_str, time_str, sender, content = match.groups()
            current_msg = {
                "role": sender.strip(),
                "content": content.strip(),
                "date": f"{date_str} {time_str}",
                "file_hint": None,
            }
        else:
            # Continuation line — append to current message
            if current_msg is not None:
                current_msg["content"] += "\n" + line.strip()

    # Don't forget the last message
    if current_msg is not None:
        messages.append(_finalize(current_msg, extract_path, order))

    logger.info("WhatsApp: parsed %d messages", len(messages))
    return messages


def _finalize(msg: dict, extract_path: Path, order: int) -> dict:
    """
    Resolve file attachment from content if present.
    Looks for actual file in extract_path directory.
    """
    content = msg["content"]
    resolved_file = None

    # Check for "(file attached)" pattern
    file_match = FILE_ATTACHMENT_PATTERN.match(content)
    if file_match:
        filename = file_match.group(1).strip()
        candidate = _find_file(extract_path, filename)
        if candidate:
            resolved_file = candidate
            content = ""  # clear content since it's just a filename
        else:
            logger.warning("WhatsApp attachment not found in ZIP: %s", filename)

    # Check for <Media omitted>
    elif OMITTED_PATTERN.search(content):
        content = ""  # media wasn't exported, nothing to do
        logger.debug("Media omitted in message from %s", msg["role"])

    return {
        "role": msg["role"],
        "content": content,
        "order": order,
        "file": resolved_file,  # absolute Path or None
        "date": _parse_whatsapp_date(msg["date"]) if msg.get("date") else None,
    }


def _find_file(extract_path: Path, filename: str) -> Path | None:
    """Search for a file by name anywhere inside extract_path."""
    # Direct match first
    direct = extract_path / filename
    if direct.is_file():
        return direct

    # Recursive search (handles nested folders)
    for candidate in extract_path.rglob(filename):
        if candidate.is_file():
            return candidate

    return None


def find_chat_txt(extract_path: Path) -> Path | None:
    """
    Locate the WhatsApp export text file inside an extracted directory.

    New Android/desktop exports are named “WhatsApp Chat … .txt”
    (the chat/group name is appended).  Older/third‑party tools produced
    `_chat.txt`.  Return the first matching file, or None if nothing is
    found.
    """
    # legacy filename still supported
    direct = extract_path / "_chat.txt"
    if direct.is_file():
        return direct

    # search every .txt for one whose name starts with “WhatsApp Chat”
    for candidate in extract_path.rglob("*.txt"):
        if candidate.name.lower().startswith("whatsapp chat"):
            return candidate

    return None
