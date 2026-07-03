import re

with open('services/classifier.py', 'r') as f:
    content = f.read()

# 1. Update Batch Size
content = content.replace("BATCH_SIZE = 30", "BATCH_SIZE = 15")

# 2. Increase text truncation from 500 to 2000
content = content.replace("first_page[:500]", "first_page[:2000]")

# 3. Update prompt to include JSON Object requirement, Examples, and Reasoning CoT
old_prompt_end = """For each file, determine:
- CATEGORY: One from the allowed list above
- SUBJECT: Academic subject (e.g., Mathematics, Computer Science, Physics, English, General)
- SUBCATEGORY: Specific topic/module if identifiable, or null

Files to classify:
{files_text}

Respond ONLY with a valid JSON array, no explanation, no markdown:
[
  {{"index": 1, "category": "Notes", "subject": "Chemistry", "subcategory": "Module 1"}},
  {{"index": 2, "category": "Attendance", "subject": "General", "subcategory": null}},
  ...
]\"\"\""""

new_prompt_end = """EXAMPLES:
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
\"\"\""""

content = content.replace(old_prompt_end, new_prompt_end)

# 4. Update the Groq call to use response_format and parse the JSON object
old_groq = """        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise academic file classifier. Always respond with valid JSON only. Never invent new categories outside the allowed list."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        results = json.loads(text)"""

new_groq = """        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        results = parsed.get("files", [])"""

content = content.replace(old_groq, new_groq)

with open('services/classifier.py', 'w') as f:
    f.write(content)

print("Classifier updated!")
