from typing import List, Dict


def create_chunks(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Split page-level text into overlapping word-based chunks.

    Each chunk retains its source document and page number.

    Args:
        pages: Output from extract_text_from_pdf()
        chunk_size: Maximum number of words per chunk
        overlap: Number of words shared between consecutive chunks

    Returns:
        List of chunk dictionaries.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)

            chunks.append(
                {
                    "text": chunk_text,
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "chunk_id": len(chunks)
                }
            )

            start += chunk_size - overlap

    return chunks