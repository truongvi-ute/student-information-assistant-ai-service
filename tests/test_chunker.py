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
    assert "chính quy" in combined
    assert "tín chỉ" in combined
    assert "Quy chế" in combined


def test_chunker_overlap():
    """Kiểm tra tính gối đầu (overlap) giữa 2 chunk liền kề."""
    chunker = TextChunker(chunk_size=30, chunk_overlap=10)
    text = "Phần thứ nhất là nội dung A. Phần thứ hai là nội dung B."
    chunks = chunker.split_text(text)

    if len(chunks) >= 2:
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1


def test_chunker_keeps_markdown_table_intact():
    """Bảng Markdown nằm gọn trong chunk_size không được bị cắt ngang hoặc mất xuống dòng."""
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    text = (
        "# Bảng học phí\n\n"
        "Đoạn giới thiệu rất dài nhằm kiểm tra việc tách khối trước bảng.\n\n"
        "| Ngành | Học phí |\n"
        "| --- | --- |\n"
        "| CNTT | 650.000 |\n"
        "| KHDL | 680.000 |"
    )
    chunks = chunker.split_text(text)

    table_chunks = [c for c in chunks if "| CNTT | 650.000 |" in c.content]
    assert len(table_chunks) == 1
    table_content = table_chunks[0].content
    assert (
        "| Ngành | Học phí |\n| --- | --- |\n| CNTT | 650.000 |\n| KHDL | 680.000 |"
        in table_content
    )


def test_chunker_repeats_header_when_table_exceeds_chunk_size():
    """Nếu bảng quá lớn so với chunk_size, tách theo dòng và lặp lại header bảng."""
    chunker = TextChunker(chunk_size=90, chunk_overlap=10)
    text = (
        "| Môn học | Số tín chỉ |\n"
        "| --- | --- |\n"
        "| Giải tích 1 | 3 |\n"
        "| Đại số tuyến tính | 3 |\n"
        "| Cấu trúc dữ liệu | 4 |"
    )
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    for c in chunks:
        assert "| Môn học | Số tín chỉ |" in c.content
