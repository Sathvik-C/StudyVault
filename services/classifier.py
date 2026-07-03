"""
Classifier service — AI batch classification via Groq with strict categories.

All files are sent in a single prompt with their surrounding context
AND the extracted text from the first page of each document.
Groq returns a JSON array with category, subject, and subcategory for each.
Falls back to Documents/Unknown if API call fails.

Batches of 15 files max per call for higher accuracy.
"""

import json
import logging
import os
import re
import time
from typing import Optional
from groq import Groq

logger = logging.getLogger(__name__)


class _RateLimitExceeded(Exception):
    """Raised by _call_groq when Groq returns 429. Signals batch_classify to
    stop sending API requests and classify remaining files locally."""


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

BATCH_SIZE = 15           # files per Groq call
INTER_BATCH_DELAY = 2.0   # seconds between calls to stay under TPM limits


# ── Extended filename rules with subject hints ────────────────
# (pattern, forced_category, subject)  — subject=None means Groq decides
FILENAME_RULES_FULL = [
    (r"(?i)(absent|attendance|present)",              "Attendance",    "General"),
    (r"(?i)(marks|result|cia|vtu.*marks|final.*marks|grade)", "Results", None),
    (r"(?i)(time\s*table|timetable|schedule)",        "Schedules",     "General"),
    (r"(?i)(seating\s*arrangement)",                  "Exams",         "General"),
    (r"(?i)(question\s*bank|q\s*bank)",               "Question Bank", None),
    (r"(?i)(assignment|homework)",                    "Assignments",   None),
    (r"(?i)(lab\s*\d|lab\s*internal|lab\s*ia)",       "Lab",           None),
    (r"(?i)(circular|notice|holiday)",                "Notices",       "General"),
    (r"(?i)(certificate|coursera|udemy|nptel)",       "Certificates",  "General"),
    (r"(?i)(poster)",                                 "Posters",       "General"),
    (r"(?i)(\.vcf$)",                                 "Contacts",      "General"),
    (r"(?i)(project|report|synopsis)",                "Reports",       None),
    (r"(?i)(event|fest|hackathon|workshop|algojam|paramcon)", "Events", "General"),
]

# Subject keyword rules — applied when category is already determined
SUBJECT_RULES = [
    (r"(?i)(math|maths|calculus|bmats|algebra)",      "Mathematics"),
    (r"(?i)(physics|bphys|phy\b)",                    "Physics"),
    (r"(?i)(chemistry|chem|bches)",                   "Chemistry"),
    (r"(?i)(dsa|data.struct|bcsl305|bcs30[12])",      "Data Structures"),
    (r"(?i)(python|bplck)",                           "Python"),
    (r"(?i)(data.analytic|bcs358|dae\b)",             "Data Analytics"),
    (r"(?i)(operating.system|bcs503\b)",              "Operating Systems"),
    (r"(?i)(computer.network|cn\b|bcs502)",           "Computer Networks"),
    (r"(?i)(constitution|ico\b|dpsp)",                "Indian Constitution"),
    (r"(?i)(english|phonetic|bengk)",                 "English"),
    (r"(?i)(kannada|bkskk|balake)",                   "Kannada"),
    (r"(?i)(ada\b|bcs401|algorithm)",                 "Algorithms"),
    (r"(?i)(nosql|mongodb|bds)",                      "NoSQL"),
    (r"(?i)(sepm|bcs501)",                            "Software Engineering"),
    (r"(?i)(aiml|bds602|machine.learn)",              "Machine Learning"),
    (r"(?i)(dvlab|dv.lab|bail504)",                   "Data Visualization"),
    (r"(?i)(uhv\b)",                                  "Universal Human Values"),
    (r"(?i)(evs\b|environment)",                      "Environmental Studies"),
]


# ── Standard Category Vocabulary ─────────────────────────────

VALID_CATEGORIES = [
    "Notes",
    "Assignments",
    "Attendance",
    "Exams",
    "Schedules",
    "Results",
    "Resources",
    "Lab",
    "Reports",
    "Notices",
    "Events",
    "Certificates",
    "Images",
    "Videos",
    "Contacts",
    "Documents",
    "Posters",
    "Question Bank",
]

# Map common AI-invented categories back to standard ones
CATEGORY_ALIASES = {
    "calendar": "Schedules",
    "time table": "Schedules",
    "timetable": "Schedules",
    "grades": "Results",
    "marks": "Results",
    "contact": "Contacts",
    "syllabus": "Resources",
    "scans": "Documents",
    "notifications": "Notices",
    "programs": "Events",
    "lists": "Attendance",
    "disciplinary": "Notices",
    "experiments": "Lab",
    "other": "Documents",
    "unknown": "Documents",
}

# Keyword rules: if a filename contains these, override the category
FILENAME_RULES = [
    # (pattern, forced_category)
    (r"(?i)(absent|attendance|present)", "Attendance"),
    (r"(?i)(marks|result|cia|vtu.*marks|final.*marks|grade)", "Results"),
    (r"(?i)(time\s*table|timetable|schedule)", "Schedules"),
    (r"(?i)(seating\s*arrangement)", "Exams"),
    (r"(?i)(question\s*bank|q\s*bank)", "Question Bank"),
    (r"(?i)(assignment|homework)", "Assignments"),
    (r"(?i)(lab\s*\d|lab\s*internal|lab\s*ia)", "Lab"),
    (r"(?i)(circular|notice|holiday)", "Notices"),
    (r"(?i)(certificate|coursera|udemy)", "Certificates"),
    (r"(?i)(poster)", "Posters"),
    (r"(?i)(\.vcf$)", "Contacts"),
]


def _build_prompt(files: list, existing_structure: list = None) -> str:
    file_lines = []
    for i, f in enumerate(files, 1):
        file_lines.append(f"{i}. Filename: {f['filename']}")
        first_page = f.get("first_page_text", "").strip()
        if first_page:
            file_lines.append(f"   First page content: {first_page[:2000]}")
        if f.get("context", "").strip():
            file_lines.append(f"   Chat context: {f['context'][:200]}")

    files_text = "\n".join(file_lines)

    existing_text = ""
    if existing_structure:
        existing_text = "\nExisting folders (prefer these for consistency):\n- " + "\n- ".join(existing_structure[:30])

    categories_list = ", ".join(VALID_CATEGORIES)

    return f"""You are an academic file organizer. Classify each file from a student study group chat.
{existing_text}

ALLOWED CATEGORIES (you MUST pick from this list):
{categories_list}

Category definitions:
- Notes: Lecture notes, study materials, module content, solved problems
- Assignments: Homework, assignments, lab answer sheets, project work
- Attendance: Absent lists, attendance sheets, student lists, room allotment
- Exams: Seating arrangements, exam instructions, internal test info, appraisals
- Schedules: Time tables, timetables, schedules, academic calendars, extra classes
- Results: Marks sheets, grades, CIA marks, VTU marks, final results, IA marks
- Resources: Textbooks, handbooks, reference material, training circulars, invitations
- Lab: Lab manuals, lab experiments, lab data sheets, lab programs
- Reports: Project reports, DAE reports
- Notices: Holidays, circulars, notifications, disciplinary notices
- Events: Event posters, fest info, hackathons, induction programs
- Certificates: Course certificates, achievement certificates
- Images: Photos (JPG/PNG) — only if no better category fits
- Videos: Video files (MP4/MOV) — only if no better category fits
- Contacts: Contact cards (VCF files)
- Documents: Miscellaneous documents that don't fit elsewhere
- Posters: Event posters, academic posters
- Question Bank: Question banks, model question papers, solved question papers

CRITICAL RULES:
- Files with "marks", "results", "CIA", "VTU", "grades" in filename → ALWAYS category "Results"
- Files with "absent", "attendance" → ALWAYS category "Attendance"
- Files with "time table", "timetable", "schedule" → ALWAYS category "Schedules"
- Files with "seating arrangement" → ALWAYS category "Exams"
- PDF files should NEVER go in "Videos" or "Images"
- .vcf files should ALWAYS go in "Contacts"
- First page content is the PRIMARY signal for classification — prioritize it over filename
- Use chat context as a secondary hint when document text is ambiguous

EXAMPLES:
Input:
1. Filename: scan123.jpg
   First page content: "Dear students, tomorrow's CS lab is cancelled"
Output:
{{"files": [{{"index": 1, "reasoning": "Mentions a lab cancellation for CS", "category": "Notices", "subject": "Computer Science", "subcategory": null}}]}}

Input:
2. Filename: img_999.pdf
   First page content: "Module 1: Introduction to Data Structures. Trees and Graphs..."
Output:
{{"files": [{{"index": 2, "reasoning": "Contains notes for Data Structures", "category": "Notes", "subject": "Data Structures", "subcategory": "Module 1"}}]}}

For each file, determine:
- REASONING: Explain your thought process in 1 sentence.
- CATEGORY: One from the allowed list above
- SUBJECT: Academic subject (e.g., Mathematics, Computer Science, Physics, English, General)
- SUBCATEGORY: Specific topic/module if identifiable, or null

Files to classify:
{files_text}

Respond ONLY with a valid JSON object containing a "files" array. Do not use markdown fences.
"""


def _call_groq(files: list, existing_structure: list = None) -> list:
    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY — all files will go to Documents/Unknown")
        return _fallback_results(files)

    prompt = _build_prompt(files, existing_structure)

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=4000,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise academic file classifier. Always respond with a valid JSON object containing a 'files' array. Never invent new categories."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        text = response.choices[0].message.content.strip()
        parsed = json.loads(text)
        results = parsed.get("files", [])
        logger.info("Groq classified %d files in one batch", len(results))
        return results

    except json.JSONDecodeError as e:
        logger.error("Groq returned invalid JSON: %s", e)
        return _fallback_results(files)

    except Exception as e:
        err_str = str(e)
        is_rate_limit = (
            "429" in err_str
            or "rate_limit" in err_str.lower()
            or "rate limit" in err_str.lower()
        )
        if is_rate_limit:
            # Fail fast — don't block the upload waiting for the limit to reset.
            # Signal callers to stop sending more batches this session.
            logger.warning(
                "Groq rate limit hit — switching remaining files to filename-rule "
                "classification so the upload completes immediately. (%s)", err_str[:120]
            )
            raise _RateLimitExceeded() from None

        logger.error("Groq batch classification failed: %s", e)
        return _fallback_results(files)


def _fallback_results(files: list) -> list:
    return [
        {"index": i + 1, "category": "Documents", "subject": "Unknown", "subcategory": None}
        for i in range(len(files))
    ]


def _normalize_category(category: str) -> str:
    """Normalize a category name to one of the standard categories."""
    if not category:
        return "Documents"

    # Exact match (case-insensitive)
    for valid in VALID_CATEGORIES:
        if category.lower() == valid.lower():
            return valid

    # Check aliases
    cat_lower = category.lower().strip()
    if cat_lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cat_lower]

    # Partial match
    for alias, target in CATEGORY_ALIASES.items():
        if alias in cat_lower:
            return target

    # No match → Documents
    logger.debug("Unknown category '%s' normalized to 'Documents'", category)
    return "Documents"


def _apply_filename_rules(filename: str, category: str) -> str:
    """Override category based on filename keyword rules."""
    for pattern, forced_category in FILENAME_RULES:
        if re.search(pattern, filename):
            if category != forced_category:
                logger.debug("Filename rule overrode '%s' → '%s' for %s",
                             category, forced_category, filename)
            return forced_category
    return category


def _normalize_subject(subject: str) -> str:
    """Normalize common subject variations."""
    if not subject:
        return "Unknown"

    aliases = {
        "maths": "Mathematics",
        "math": "Mathematics",
        "cs": "Computer Science",
        "cse": "Computer Science",
        "phy": "Physics",
        "chem": "Chemistry",
        "bio": "Biology",
        "dsa": "Data Structures",
        "dae": "Data Analytics with Excel",
        "ico": "Indian Constitution",
        "os": "Operating Systems",
        "cn": "Computer Networks",
        "ca": "Computer Architecture",
        "atc": "Automata Theory",
        "iks": "Indian Knowledge Systems",
    }

    subj_lower = subject.lower().strip()
    if subj_lower in aliases:
        return aliases[subj_lower]

    return subject.strip()


def _guess_subject_from_filename(filename: str) -> Optional[str]:
    """Return a subject if the filename clearly indicates one, else None."""
    for pattern, subject in SUBJECT_RULES:
        if re.search(pattern, filename):
            return subject
    return None


def _classify_by_filename(filename: str) -> Optional[dict]:
    """
    Try to fully classify a file using filename rules alone.
    Returns a result dict if confident, or None if Groq is needed.
    """
    for pattern, category, hint_subject in FILENAME_RULES_FULL:
        if re.search(pattern, filename):
            subject = hint_subject or _guess_subject_from_filename(filename) or "General"
            return {
                "category": category,
                "subject": subject,
                "subcategory": None,
                "method": "filename_rule",
            }
    # Also check by extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "vcf":
        return {"category": "Contacts", "subject": "General", "subcategory": None, "method": "filename_rule"}
    return None


def batch_classify(file_list: list, existing_structure: list = None) -> dict:
    """
    Classify all files. Files with clear filename patterns are classified
    locally (no Groq call). Only ambiguous files go to Groq, in batches
    with throttling and retry-on-rate-limit.

    Args:
        file_list: list of {filename, context, first_page_text (optional)}
        existing_structure: list of "Category > Subject > Subcategory" strings

    Returns:
        dict mapping filename → {category, subject, subcategory, method}
    """
    if not file_list:
        return {}

    results_map = {}
    needs_ai = []   # files that couldn't be determined from filename alone

    # ── Pass 1: classify by filename rules (zero API calls) ──────────
    for f in file_list:
        fname = f["filename"]
        result = _classify_by_filename(fname)
        if result:
            results_map[fname] = result
        else:
            needs_ai.append(f)

    logger.info(
        "Filename rules classified %d/%d files. Sending %d to Groq.",
        len(results_map), len(file_list), len(needs_ai)
    )

    # ── Pass 2: AI classify only truly ambiguous files ────────────
    for batch_idx, batch_start in enumerate(range(0, len(needs_ai), BATCH_SIZE)):
        batch = needs_ai[batch_start: batch_start + BATCH_SIZE]
        logger.info(
            "AI batch %d: classifying files %d–%d of %d",
            batch_idx + 1, batch_start + 1, batch_start + len(batch), len(needs_ai)
        )

        # Throttle between calls to avoid TPM limit
        if batch_idx > 0:
            time.sleep(INTER_BATCH_DELAY)

        try:
            raw_results = _call_groq(batch, existing_structure)
        except _RateLimitExceeded:
            logger.warning(
                "Rate limit hit on batch %d — skipping remaining %d AI batches "
                "and classifying by filename rules instead.",
                batch_idx + 1,
                len(needs_ai) - batch_start,
            )
            break  # jump straight to fill-missing (filename fallback)

        for item in raw_results:
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(batch):
                filename = batch[idx]["filename"]
                category = item.get("category", "Documents")
                subject = item.get("subject", "Unknown")
                subcategory = item.get("subcategory")

                # ── Post-processing pipeline ──
                category  = _normalize_category(category)
                category  = _apply_filename_rules(filename, category)
                subject   = _normalize_subject(subject)

                if isinstance(subcategory, str) and subcategory.lower() in ("none", "null", ""):
                    subcategory = None

                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext in ("pdf", "docx", "pptx", "doc", "xlsx") and category in ("Videos", "Images"):
                    category = "Documents"

                results_map[filename] = {
                    "category": category,
                    "subject": subject,
                    "subcategory": subcategory,
                    "method": "ai" if GROQ_API_KEY else "fallback",
                }

    # ── Fill anything still missing ────────────────────────────
    for f in file_list:
        if f["filename"] not in results_map:
            fname = f["filename"]
            category = _apply_filename_rules(fname, "Documents")
            subject  = _guess_subject_from_filename(fname) or "Unknown"
            results_map[fname] = {
                "category": category,
                "subject": subject,
                "subcategory": None,
                "method": "fallback",
            }

    return results_map
