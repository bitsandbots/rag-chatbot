"""Load and chunk text files for RAG ingestion."""

from __future__ import annotations

from pathlib import Path


def load_text_files(
    directory: str, chunk_size: int = 500
) -> tuple[list[str], list[str]]:
    """Load and chunk text files from a directory.

    Args:
        directory: Path to directory containing .txt files.
        chunk_size: Maximum characters per chunk.

    Returns:
        Tuple of (chunks, ids) where ids are "{filename}_{offset}".
    """
    chunks: list[str] = []
    ids: list[str] = []

    for file_path in sorted(Path(directory).glob("*.txt")):
        text = file_path.read_text()
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)
            ids.append(f"{file_path.name}_{i}")

    return chunks, ids
