"""Functions for reading text out of PDF files."""

import re

import fitz  # this is PyMuPDF


def clean_text(text: str) -> str:
    """Tidy up messy PDF text: remove extra spaces and blank lines."""
    text = re.sub(r"[ \t]+", " ", text)      # many spaces -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)   # many blank lines -> one blank line
    return text.strip()


def extract_text_from_pdf(pdf_bytes: bytes):
    """
    Read a PDF and return (text, number_of_pages).

    pdf_bytes: the raw content of the uploaded PDF file.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())

    page_count = len(doc)
    doc.close()

    full_text = clean_text("\n".join(pages_text))
    return full_text, page_count