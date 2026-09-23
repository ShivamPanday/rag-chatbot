"""
document_loader.py
--------------------
Extracts plain text from uploaded documents, regardless of format.
"""

from io import BytesIO
from pathlib import Path

import docx
import pdfplumber


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from a document given its bytes and filename."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_bytes)
    elif suffix == ".docx":
        return _extract_docx(file_bytes)
    elif suffix == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(file_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _extract_docx(file_bytes: bytes) -> str:
    document = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
