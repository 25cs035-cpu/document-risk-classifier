"""Text extraction (PDF / DOCX / TXT) and cleaning."""
import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"page\s+\d+(\s+of\s+\d+)?", " ", text, flags=re.I)  # page numbers
    text = re.sub(r"\s+", " ", text)  # collapse whitespace / newlines
    return text.strip().lower()


def extract_text(source, filename=None) -> str:
    """source: a file path (str) or a file-like object (e.g. Streamlit upload)."""
    name = (filename or (source if isinstance(source, str) else "")).lower()

    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(source)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        from docx import Document

        doc = Document(source)
        return "\n".join(p.text for p in doc.paragraphs)

    if isinstance(source, str):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    data = source.read()
    return data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)
