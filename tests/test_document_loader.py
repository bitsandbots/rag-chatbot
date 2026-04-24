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


def test_multiple_files_sorted(tmp_path: Path) -> None:
    """Files should be processed in sorted order regardless of creation order."""
    (tmp_path / "zebra.txt").write_text("Z content")
    (tmp_path / "alpha.txt").write_text("A content")
    chunks, ids = load_text_files(str(tmp_path))
    assert len(chunks) == 2
    assert chunks[0] == "A content"
    assert chunks[1] == "Z content"
    assert ids[0] == "alpha.txt_0"
    assert ids[1] == "zebra.txt_0"


def test_non_txt_files_ignored(tmp_path: Path) -> None:
    """Only .txt files should be loaded; others ignored."""
    (tmp_path / "data.txt").write_text("included")
    (tmp_path / "data.csv").write_text("excluded")
    (tmp_path / "data.json").write_text("excluded")
    (tmp_path / "readme.md").write_text("excluded")
    chunks, ids = load_text_files(str(tmp_path))
    assert len(chunks) == 1
    assert chunks[0] == "included"


def test_exact_chunk_boundary(tmp_path: Path) -> None:
    """Text exactly divisible by chunk_size should produce no partial chunk."""
    text = "X" * 500
    (tmp_path / "exact.txt").write_text(text)
    chunks, ids = load_text_files(str(tmp_path), chunk_size=500)
    assert len(chunks) == 1
    assert len(chunks[0]) == 500


def test_chunk_size_one(tmp_path: Path) -> None:
    """Edge case: chunk_size=1 should produce one chunk per character."""
    (tmp_path / "tiny.txt").write_text("abc")
    chunks, ids = load_text_files(str(tmp_path), chunk_size=1)
    assert chunks == ["a", "b", "c"]
    assert ids == ["tiny.txt_0", "tiny.txt_1", "tiny.txt_2"]


def test_multiple_files_with_chunking(tmp_path: Path) -> None:
    """Chunks from multiple files should be interleaved by file order."""
    (tmp_path / "a.txt").write_text("A" * 600)
    (tmp_path / "b.txt").write_text("B" * 300)
    chunks, ids = load_text_files(str(tmp_path), chunk_size=500)
    # a.txt: 2 chunks (500 + 100), b.txt: 1 chunk (300)
    assert len(chunks) == 3
    assert ids[0] == "a.txt_0"
    assert ids[1] == "a.txt_500"
    assert ids[2] == "b.txt_0"
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 300
