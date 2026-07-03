from __future__ import annotations
"""
Scans parsed messages and flags important ones based on trigger keywords.
Also extracts any deadline/date mentions from the message text.
"""

import re
import logging
from models.config import IMPORTANT_TRIGGERS

logger = logging.getLogger(__name__)

# Date pattern with ordinal-day support: "9th February", "Nov 15", "15/11", "15-11-2024" etc.
DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*(?:\s*,?\s*\d{2,4})?|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{2,4})?)\b",
    re.IGNORECASE,
)

DEADLINE_HINTS = (
    "deadline", "due", "submit", "submission", "last date", "before",
    "on or before", "by", "exam on", "test on", "registration", "complete by"
)


def _looks_like_chat_timestamp_context(context: str) -> bool:
    # WhatsApp export lines often contain fragments like "2/4/26, 9:21AM - ~ Name added ..."
    if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", context, re.IGNORECASE):
        return True
    if " added " in context.lower() or " - ~" in context:
        return True
    return False


def _extract_deadline_candidate(content: str) -> str | None:
    matches = list(DATE_PATTERN.finditer(content))
    if not matches:
        return None

    ranked: list[tuple[int, int, str]] = []
    content_lower = content.lower()

    for m in matches:
        candidate = m.group(0)
        start, end = m.span()
        window_start = max(0, start - 36)
        window_end = min(len(content), end + 36)
        context = content[window_start:window_end]
        context_lower = content_lower[window_start:window_end]

        score = 0
        if any(ch.isalpha() for ch in candidate):
            score += 20  # Prefer month-name dates over numeric-only dates.
        if re.search(r"\d{2,4}", candidate):
            score += 8
        if any(h in context_lower for h in DEADLINE_HINTS):
            score += 28
        if _looks_like_chat_timestamp_context(context):
            score -= 35

        ranked.append((score, start, candidate))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    best = ranked[0][2] if ranked else None
    return best


def extract_deadline_from_text(content: str | None) -> str | None:
    if not content:
        return None
    return _extract_deadline_candidate(content)


def detect_important(messages: list[dict]) -> list[dict]:
    """
    Given a list of parsed message dicts, return those that are important.

    Each returned dict has:
        - role, content, order (original fields)
        - trigger_word: which keyword caused the flag
        - detected_deadline: any date found in the message, or None
    """
    flagged = []

    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue

        content_lower = content.lower()
        triggered_by = None

        for trigger in IMPORTANT_TRIGGERS:
            if trigger in content_lower:
                triggered_by = trigger
                break

        if triggered_by is None:
            continue

        # Extract a deadline candidate using context-aware scoring.
        detected_deadline = _extract_deadline_candidate(content)

        flagged.append({
            **msg,
            "trigger_word": triggered_by,
            "detected_deadline": detected_deadline,
        })

        logger.info(
            "Flagged important message from '%s' (trigger: '%s', deadline: %s)",
            msg.get("role"), triggered_by, detected_deadline
        )

    return flagged


def write_important_messages_file(
    flagged_messages: list[dict],
    output_path,  # Path object
) -> None:
    """
    Write all important messages to a human-readable text file
    at storage/{file_id}/important_messages.txt
    """
    if not flagged_messages:
        return

    lines = [
        "=" * 60,
        "  IMPORTANT MESSAGES",
        "=" * 60,
        "",
    ]

    for i, msg in enumerate(flagged_messages, start=1):
        lines.append(f"[{i}] From: {msg.get('role', 'Unknown')}")
        if msg.get("detected_deadline"):
            lines.append(f"    📅 Deadline detected: {msg['detected_deadline']}")
        lines.append(f"    🔖 Trigger: \"{msg.get('trigger_word')}\"")
        lines.append(f"    💬 {msg.get('content', '').strip()}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %d important messages to %s", len(flagged_messages), output_path)
