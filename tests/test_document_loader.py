from __future__ import annotations

from pathlib import Path

from rag_chatbot.document_loader import load_text_files


def test_load_text_files(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("Hello world this is a test file.")
    chunks, ids = load_text_files(str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0] == "Hello world this is a test file."
    assert ids[0] == "hello.txt_0"


def test_chunking(tmp_path: Path) -> None:
    text = "A" * 1200  # Should produce 3 chunks at chunk_size=500
    (tmp_path / "big.txt").write_text(text)
    chunks, ids = load_text_files(str(tmp_path), chunk_size=500)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500
    assert len(chunks[2]) == 200
    assert ids[0] == "big.txt_0"
    assert ids[1] == "big.txt_500"
    assert ids[2] == "big.txt_1000"


def test_empty_directory(tmp_path: Path) -> None:
    chunks, ids = load_text_files(str(tmp_path))
    assert chunks == []
    assert ids == []
