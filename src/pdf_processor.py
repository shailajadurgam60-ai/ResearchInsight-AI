import pymupdf as fitz
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts while preserving
    meaningful document content.
    """

    # Fix words broken across lines by hyphenation
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace line breaks with spaces
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract and clean text from each PDF page.

    Returns:
        List of dictionaries containing:
        - text
        - page_number
        - source
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        raw_text = page.get_text("text")

        text = clean_text(raw_text)

        if text:
            pages.append(
                {
                    "text": text,
                    "page_number": page_number,
                    "source": pdf_path.name,
                }
            )

    document.close()

    return pages