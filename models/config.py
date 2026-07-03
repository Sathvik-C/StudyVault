import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STORAGE_DIR = BASE_DIR / "storage"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))   # 200MB
MAX_EXTRACTED_BYTES = int(os.getenv("MAX_EXTRACTED_BYTES", str(1024 * 1024 * 1024)))  # 1GB

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Category keyword rules ───────────────────────────────────
# Order matters: first match wins
CATEGORY_RULES = [
    {
        "category": "Attendance",
        "keywords": ["attendance", "roll call", "present", "absent", "rollsheet"],
    },
    {
        "category": "Exams",
        "keywords": ["exam", "test", "quiz", "midterm", "final", "paper", "question paper", "qp"],
    },
    {
        "category": "Assignments",
        "keywords": ["assignment", "homework", "hw", "submit", "submission", "task", "project"],
    },
    {
        "category": "Notes",
        "keywords": ["note", "notes", "module", "chapter", "unit", "lecture", "lec", "summary", "handout"],
    },
    {
        "category": "Resources",
        "keywords": ["resource", "reference", "book", "textbook", "pdf", "material", "syllabus", "curriculum"],
    },
]

# ── Subject keyword rules ─────────────────────────────────────
SUBJECT_RULES = [
    {"subject": "Mathematics",  "keywords": ["math", "maths", "algebra", "calculus", "statistics", "stat", "trig"]},
    {"subject": "Physics",      "keywords": ["physics", "phy", "mechanics", "optics", "thermodynamics"]},
    {"subject": "Chemistry",    "keywords": ["chemistry", "chem", "organic", "inorganic", "periodic"]},
    {"subject": "Biology",      "keywords": ["biology", "bio", "botany", "zoology", "genetics", "anatomy"]},
    {"subject": "Computer Science", "keywords": ["cs", "computer", "programming", "python", "java", "code", "algorithm", "data structure", "dbms", "os", "networking"]},
    {"subject": "English",      "keywords": ["english", "grammar", "essay", "writing", "literature"]},
    {"subject": "History",      "keywords": ["history", "hist", "medieval", "modern", "ancient"]},
    {"subject": "Geography",    "keywords": ["geography", "geo", "map", "climate", "physical"]},
    {"subject": "Economics",    "keywords": ["economics", "eco", "micro", "macro", "gdp", "market"]},
]

# ── Important message trigger words ──────────────────────────
IMPORTANT_TRIGGERS = [
    "important", "note:", "notes:", "remember", "don't forget", "do not forget",
    "deadline", "due date", "due on", "submit by", "last date",
    "exam on", "exam date", "test on", "quiz on",
    "announcement", "notice", "attention", "urgent", "asap",
    "marks", "result", "results", "grade",
]

# ── Subcategory extraction patterns ──────────────────────────
from typing import Optional
import re

SUBCATEGORY_PATTERNS = [
    r"module\s*(\d+)",
    r"chapter\s*(\d+)",
    r"unit\s*(\d+)",
    r"lecture\s*(\d+)",
    r"lec\s*(\d+)",
    r"week\s*(\d+)",
    r"part\s*(\d+)",
]


def extract_subcategory(text: str) -> Optional[str]:
    text_lower = text.lower()
    for pattern in SUBCATEGORY_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            label = re.split(r"\\s", pattern)[0].replace("\\", "").capitalize()
            return f"{label} {match.group(1)}"
    return None
