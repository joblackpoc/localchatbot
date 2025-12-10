import docx
import pdfplumber
from bs4 import BeautifulSoup

def extract_text_from_file(path):
    # PDF
    if path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    # DOCX
    if path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    # HTML
    if path.endswith(".html") or path.endswith(".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator="\n")

    # PY or TXT
    if path.endswith(".py") or path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    return ""
