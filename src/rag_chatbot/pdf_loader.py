"""Load and chunk PDF files for RAG ingestion."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def load_pdf(
    path: str, chunk_size: int = 500
) -> tuple[list[str], list[str]]:
    """Load and chunk a PDF file.

    Args:
        path: Path to the PDF file.
        chunk_size: Maximum characters per chunk.

    Returns:
        Tuple of (chunks, ids) where ids are "{filename}_{offset}".
    """
    chunks: list[str] = []
    ids: list[str] = []
    file_name = Path(path).name

    doc = pymupdf.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i : i + chunk_size]
        chunks.append(chunk)
        ids.append(f"{file_name}_{i}")

    return chunks, ids
