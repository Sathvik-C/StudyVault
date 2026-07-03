"""
generate_test_chat.py
─────────────────────
Generates a realistic fake WhatsApp academic group export ZIP
for testing the StudyVault ingestion pipeline.

Creates:
  • _chat.txt            — 200+ realistic messages spanning 6 months
  • 80+ dummy attachments (PDFs, images, DOCX, XLSX, PPTX, ZIP)
ff
Usage:
    python3 generate_test_chat.py
    → produces:  test_chat_export.zip

Then upload that ZIP through the StudyVault UI.
"""

import io
import os
import random
import zipfile
from datetime import datetime, timedelta

# ── Configurable ──────────────────────────────────────────────
OUTPUT_FILE = "test_chat_export.zip"
CHAT_NAME   = "WhatsApp Chat with 4th Sem CS-B Study Group"
START_DATE  = datetime(2025, 7, 15, 8, 30)

# ── Members ───────────────────────────────────────────────────
MEMBERS = [
    "Rohan Sharma", "Priya Nair", "Arjun Patel", "Sneha Reddy",
    "Kiran Mehta", "Divya Krishnan", "Rahul Gupta", "Ananya Singh",
    "Vikram Joshi", "Meera Iyer",
]

# ── File library ──────────────────────────────────────────────
# (filename, extension, category_hint)
FILE_LIBRARY = [
    # Results / Marks
    ("DSA_VTU_IA_Marks.pdf",               "pdf",  "result"),
    ("OS_Final_CIE_Marks.pdf",             "pdf",  "result"),
    ("CN_VTU_Marks.pdf",                   "pdf",  "result"),
    ("DBMS_IA_Marks.pdf",                  "pdf",  "result"),
    ("TOC_Final_Marks.pdf",                "pdf",  "result"),
    ("Math_IA2_Result.pdf",                "pdf",  "result"),

    # Notes
    ("DSA_Module3_Notes.pdf",              "pdf",  "notes"),
    ("OS_Module1_Notes.pdf",               "pdf",  "notes"),
    ("CN_Module2_Notes.pdf",               "pdf",  "notes"),
    ("DBMS_Module4_Notes.pdf",             "pdf",  "notes"),
    ("TOC_Module2_Notes.pdf",              "pdf",  "notes"),
    ("Math_Module5.pdf",                   "pdf",  "notes"),
    ("OS_Solved_Questions.pdf",            "pdf",  "notes"),
    ("CN_Important_Topics.pdf",            "pdf",  "notes"),
    ("DSA_Programs.pdf",                   "pdf",  "notes"),
    ("DBMS_SQL_Notes.pdf",                 "pdf",  "notes"),
    ("OS_Process_Scheduling.pptx",         "pptx", "notes"),
    ("CN_Transport_Layer.pptx",            "pptx", "notes"),

    # Assignments
    ("Assignment1_DSA.pdf",                "pdf",  "assignment"),
    ("Assignment2_OS.pdf",                 "pdf",  "assignment"),
    ("DBMS_Assignment1.pdf",               "pdf",  "assignment"),
    ("Math_Assignment3.pdf",               "pdf",  "assignment"),
    ("CN_Assignment2.docx",               "docx", "assignment"),
    ("TOC_Homework1.pdf",                  "pdf",  "assignment"),
    ("Lab_IA_Answer_Sheet.pdf",            "pdf",  "assignment"),

    # Lab
    ("DSA_Lab_Manual.pdf",                 "pdf",  "lab"),
    ("OS_Lab_Programs.pdf",                "pdf",  "lab"),
    ("DBMS_Lab_Manual.pdf",                "pdf",  "lab"),
    ("CN_Lab_Experiments.pdf",             "pdf",  "lab"),
    ("DSA_Lab_IA_Sheet.pdf",               "pdf",  "lab"),

    # Schedules / Timetable
    ("4th_Sem_Timetable.pdf",              "pdf",  "schedule"),
    ("IA_Test_Schedule.pdf",               "pdf",  "schedule"),
    ("Lab_Timetable.pdf",                  "pdf",  "schedule"),
    ("Exam_Schedule_VTU.pdf",              "pdf",  "schedule"),

    # Question Banks
    ("DSA_Question_Bank.pdf",              "pdf",  "qbank"),
    ("OS_Question_Bank.pdf",               "pdf",  "qbank"),
    ("DBMS_PYQ_2024.pdf",                  "pdf",  "qbank"),
    ("CN_Model_QP.pdf",                    "pdf",  "qbank"),
    ("TOC_Question_Bank.pdf",              "pdf",  "qbank"),
    ("Math_QB_Module4.pdf",                "pdf",  "qbank"),

    # Notices / Circulars
    ("Holiday_Notice_Nov.pdf",             "pdf",  "notice"),
    ("Exam_Circular_VTU.pdf",              "pdf",  "notice"),
    ("Remedial_Class_Notice.pdf",          "pdf",  "notice"),
    ("Fee_Circular.pdf",                   "pdf",  "notice"),

    # Resources
    ("VTU_Syllabus_4thSem.pdf",            "pdf",  "resource"),
    ("NPTEL_Registration.pdf",             "pdf",  "resource"),
    ("Reference_Book_DSA.pdf",             "pdf",  "resource"),
    ("CS_Handbook_2024.pdf",               "pdf",  "resource"),

    # Events
    ("Hackathon_Poster.pdf",               "pdf",  "event"),
    ("Tech_Fest_Schedule.pdf",             "pdf",  "event"),
    ("Workshop_Registration.pdf",          "pdf",  "event"),

    # Certificates
    ("NPTEL_DSA_Certificate.pdf",          "pdf",  "cert"),
    ("Coursera_Python_Certificate.pdf",    "pdf",  "cert"),
    ("Hackathon_Participation.pdf",        "pdf",  "cert"),

    # Images (jpg)
    ("IMG-20250901-WA0001.jpg",            "jpg",  "image"),
    ("IMG-20250912-WA0003.jpg",            "jpg",  "image"),
    ("IMG-20251005-WA0007.jpg",            "jpg",  "image"),
    ("IMG-20251103-WA0002.jpg",            "jpg",  "image"),
    ("IMG-20251115-WA0005.jpg",            "jpg",  "image"),
    ("IMG-20251204-WA0008.jpg",            "jpg",  "image"),
    ("IMG-20260102-WA0004.jpg",            "jpg",  "image"),
    ("IMG-20260210-WA0006.jpg",            "jpg",  "image"),
    ("IMG-20260315-WA0001.jpg",            "jpg",  "image"),
    ("IMG-20260401-WA0009.jpg",            "jpg",  "image"),

    # XLSX
    ("Student_List_4B.xlsx",               "xlsx", "attendance"),
    ("Marks_Entry_Sheet.xlsx",             "xlsx", "result"),
    ("Attendance_Oct.xlsx",                "xlsx", "attendance"),

    # DOCX
    ("Project_Report_Draft.docx",         "docx", "report"),
    ("Mini_Project_Synopsis.docx",        "docx", "report"),
    ("Lab_Record_Format.docx",            "docx", "lab"),

    # Misc
    ("VTU_Admit_Card.pdf",                 "pdf",  "resource"),
    ("Hostel_Circular.pdf",               "pdf",  "notice"),
    ("Activity_Points.pdf",               "pdf",  "resource"),
    ("IEEE_Membership.pdf",               "pdf",  "cert"),
]

# ── Message templates ─────────────────────────────────────────
MESSAGES_WITHOUT_FILES = [
    "Anyone done the DSA assignment?",
    "What's the syllabus for the IA test?",
    "Did anyone attend today's OS class?",
    "The DBMS lab is shifted to Thursday",
    "Can someone share the CN notes?",
    "Module 3 is very important for the exam",
    "Which chapters are covered in IA2?",
    "Prof said to focus on dynamic programming",
    "The lab viva is next week, be prepared",
    "Attendance is below 75% for many students, be careful",
    "Anyone know when results will be out?",
    "Extra class for TOC tomorrow at 4pm",
    "Study group meeting at 6pm in library?",
    "Don't forget to submit the assignment by tomorrow",
    "The question bank has most of the important questions",
    "OS internals and scheduling algorithms are must-study",
    "CN transport layer MCQs are tough, practice them",
    "DBMS normalization is confirmed for exam",
    "Bro did you check the time table? Lab changed",
    "Thanks for sharing the notes 🙏",
    "Can someone explain Banker's algorithm?",
    "VTU results are out! Check your results",
    "Congrats to everyone who cleared the backlogs",
    "Free period at 11am today",
    "Prof is absent, self-study for CN",
    "Anyone else finding TOC difficult?",
    "The hackathon registrations are open!",
    "Last date for NPTEL enrollment is Friday",
    "Practical exam next month, start preparing",
    "Check the notice board for room allotment",
    "IA test postponed to next week",
    "Reminder: submit activity points by end of month",
    "Good luck everyone for the exam tomorrow 👍",
    "Share your notes if you have module 4 done",
    "Does anyone have the previous year question paper?",
    "Group project submission is on 15th",
    "Attendance sheet was circulated, sign it",
]

DEADLINE_MESSAGES = [
    "Assignment submission deadline is {date}, don't miss it!",
    "IA test on {date}, cover all modules",
    "Project submission due on {date}",
    "Last date to submit lab record: {date}",
    "VTU practical exam scheduled on {date}",
    "Fee payment deadline: {date}",
    "NPTEL quiz deadline {date}",
    "Internal exam on {date} - be prepared",
]

FILE_SHARE_TEMPLATES = [
    "Sharing the {name}",
    "Here's the {name}",
    "{name} (file attached)",
    "Check this out - {name}",
    "As requested - {name}",
    "Found this useful - {name}",
    "Important! {name}",
    "Please go through this {name}",
    "{name}",
]


def make_dummy_pdf_bytes(title: str) -> bytes:
    """Create a minimal valid-ish PDF with some readable text."""
    content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 200 >>
stream
BT
/F1 16 Tf
72 720 Td
({title}) Tj
0 -30 Td
/F1 11 Tf
(This is a sample academic document for testing StudyVault.) Tj
0 -20 Td
(Subject: Computer Science | Semester: 4) Tj
0 -20 Td
(Generated for testing purposes.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000525 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
610
%%EOF"""
    return content.encode("latin-1", errors="replace")


def make_dummy_bytes(ext: str, name: str) -> bytes:
    """Return minimal dummy bytes for any file extension."""
    if ext == "pdf":
        return make_dummy_pdf_bytes(name)
    elif ext == "jpg":
        # Minimal valid JPEG header
        return bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0xFF, 0xDB, 0x00, 0x43, 0x00,
        ] + [16] * 64 + [
            0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
            0xFF, 0xD9,
        ])
    elif ext == "docx":
        # docx is a ZIP — return a tiny valid one
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("[Content_Types].xml",
                '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>")
            z.writestr("word/document.xml",
                f'<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f'<w:body><w:p><w:r><w:t>{name}</w:t></w:r></w:p></w:body></w:document>')
        return buf.getvalue()
    elif ext in ("xlsx", "pptx"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("placeholder.txt", f"Dummy {ext} file: {name}")
        return buf.getvalue()
    else:
        return f"Dummy file: {name}\n".encode()


def format_date(dt: datetime) -> str:
    """WhatsApp date format: DD/MM/YYYY, H:MM am/pm"""
    return dt.strftime("%-d/%-m/%Y, %-I:%M %p").lower()


def generate_chat_and_files():
    random.seed(42)
    chat_lines = []
    attachments = {}  # filename → bytes

    current_dt = START_DATE
    files_to_share = list(FILE_LIBRARY)
    random.shuffle(files_to_share)
    file_idx = 0

    def advance(min_min=5, max_min=120):
        nonlocal current_dt
        current_dt += timedelta(minutes=random.randint(min_min, max_min))

    # Opening burst
    for _ in range(3):
        advance(1, 30)
        sender = random.choice(MEMBERS)
        msg = random.choice(MESSAGES_WITHOUT_FILES)
        chat_lines.append(f"{format_date(current_dt)} - {sender}: {msg}")

    # Main body — mix of text messages, file shares, deadline messages
    total_messages = 220
    for i in range(total_messages):
        advance(10, 480)
        sender = random.choice(MEMBERS)

        roll = random.random()

        if roll < 0.35 and file_idx < len(files_to_share):
            # Share a file
            fname, ext, _ = files_to_share[file_idx]
            file_idx += 1
            template = random.choice(FILE_SHARE_TEMPLATES)
            display = template.format(name=fname)
            if "(file attached)" not in display:
                display = f"{fname} (file attached)"
            chat_lines.append(f"{format_date(current_dt)} - {sender}: {display}")
            attachments[fname] = make_dummy_bytes(ext, fname)

        elif roll < 0.45:
            # Deadline message
            future = current_dt + timedelta(days=random.randint(3, 21))
            day = future.strftime("%-d %B")
            tmpl = random.choice(DEADLINE_MESSAGES)
            chat_lines.append(f"{format_date(current_dt)} - {sender}: {tmpl.format(date=day)}")

        else:
            # Regular message
            msg = random.choice(MESSAGES_WITHOUT_FILES)
            chat_lines.append(f"{format_date(current_dt)} - {sender}: {msg}")

    # Share any remaining files
    while file_idx < len(files_to_share):
        advance(30, 300)
        fname, ext, _ = files_to_share[file_idx]
        file_idx += 1
        sender = random.choice(MEMBERS)
        chat_lines.append(f"{format_date(current_dt)} - {sender}: {fname} (file attached)")
        attachments[fname] = make_dummy_bytes(ext, fname)

    chat_txt = "\n".join(chat_lines)
    return chat_txt, attachments


def build_zip(output_path: str):
    print(f"Generating test chat export...")
    chat_txt, attachments = generate_chat_and_files()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"WhatsApp Chat with 4th Sem CS-B Study Group.txt", chat_txt)
        for fname, data in attachments.items():
            zf.writestr(fname, data)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ Created: {output_path}")
    print(f"   Messages : {chat_txt.count(chr(10)) + 1}")
    print(f"   Files    : {len(attachments)}")
    print(f"   ZIP size : {size_kb:.1f} KB")
    print(f"\nUpload '{output_path}' through the StudyVault UI to test.")


if __name__ == "__main__":
    build_zip(OUTPUT_FILE)
