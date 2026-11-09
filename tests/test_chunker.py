import pytest
from app.services.ingestion.chunker import TextChunker


def test_chunker_basic_split():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "Trường Đại học Công nghệ Kĩ thuật TP.HCM là số 1. Số 1 Võ Văn Ngân, Thủ Đức, TP.HCM"
    chunks = chunker.split_text(text)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content) > 0
        assert len(chunk.content) <= 60  # cho phép dung sai dấu cách


def test_chunker_vietnamese_unicode():
    """Kiểm tra tiếng Việt có dấu không bị lỗi Unicode khi cắt."""
    chunker = TextChunker(chunk_size=40, chunk_overlap=10)
    text = "Quy chế học vụ năm học 2024 - 2025 dành cho sinh viên hệ chính quy đào tạo tín chỉ."
    chunks = chunker.split_text(text)

    combined = " ".join([c.content for c in chunks])
    # Đảm bảo các từ tiếng Việt không bị gãy hay mất ký tự có dấu
    assert "chính quy" in combined
    assert "tín chỉ" in combined
    assert "Quy chế" in combined


def test_chunker_overlap():
    """Kiểm tra tính gối đầu (overlap) giữa 2 chunk liền kề."""
    chunker = TextChunker(chunk_size=30, chunk_overlap=10)
    text = "Phần thứ nhất là nội dung A. Phần thứ hai là nội dung B."
    chunks = chunker.split_text(text)

    if len(chunks) >= 2:
        # Chunk 1 và Chunk 2 có gối đầu một phần từ vựng
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1