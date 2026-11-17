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


def test_chunker_tracks_exact_char_offsets():
    """Kiểm tra char_start và char_end ánh xạ đúng vị trí trên văn bản Markdown chuẩn hóa."""
    chunker = TextChunker(chunk_size=120, chunk_overlap=20)
    text = (
        "# Chương 1: Quy định chung\n\n"
        "Sinh viên bậc đại học chính quy đăng ký tối thiểu 14 tín chỉ trong mỗi học kỳ chính.\n\n"
        "Sinh viên diện cảnh báo học vụ chỉ được đăng ký tối đa 14 tín chỉ."
    )
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.char_start is not None
        assert chunk.char_end is not None
        assert 0 <= chunk.char_start < chunk.char_end <= len(text)
        sliced = text[chunk.char_start : chunk.char_end]
        assert sliced.strip() != ""
        assert sliced in text


def test_chunker_oversized_prose_overlap_offsets():
    """Kiểm tra đoạn văn dài bị tách có overlap vẫn giữ char_start/char_end hợp lệ."""
    chunker = TextChunker(chunk_size=60, chunk_overlap=15)
    text = (
        "Quy chế đào tạo theo hệ thống tín chỉ áp dụng cho toàn bộ sinh viên chính quy. "
        "Thời gian học tập tối đa được quy định chi tiết theo từng ngành đào tạo."
    )
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    assert chunks[0].char_start == 0
    for chunk in chunks:
        assert chunk.char_start is not None and chunk.char_end is not None
        assert 0 <= chunk.char_start < chunk.char_end <= len(text)
    # Chunk sau gối đầu hoặc nối tiếp chunk trước
    assert chunks[1].char_start <= chunks[0].char_end


def test_chunker_oversized_table_offsets():
    """Kiểm tra bảng bị tách vẫn có char_start/char_end nằm đúng trong bảng gốc."""
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
    for chunk in chunks:
        assert chunk.char_start is not None and chunk.char_end is not None
        assert 0 <= chunk.char_start < chunk.char_end <= len(text)
    assert "| Giải tích 1 | 3 |" in text[chunks[0].char_start : chunks[0].char_end]
    assert "| Cấu trúc dữ liệu | 4 |" in text[chunks[-1].char_start : chunks[-1].char_end]
