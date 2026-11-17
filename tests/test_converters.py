import io

import docx
import pymupdf
import pytest

from app.core.exceptions import DocumentLoadError
from app.services.ingestion.converters import converter_registry


def _build_sample_docx_bytes() -> bytes:
    """Tạo file .docx mẫu trong RAM gồm tiêu đề, đoạn văn và bảng biểu tiếng Việt."""
    document = docx.Document()
    document.add_heading("Quy định học phí năm học 2025", level=1)
    document.add_paragraph("Sinh viên đóng học phí theo số tín chỉ đăng ký.")

    table = document.add_table(rows=3, cols=2)
    rows = [
        ("Ngành đào tạo", "Đơn giá (VNĐ/tín chỉ)"),
        ("Công nghệ thông tin", "650.000"),
        ("Kỹ thuật dữ liệu", "680.000"),
    ]
    for r_idx, (col1, col2) in enumerate(rows):
        table.rows[r_idx].cells[0].text = col1
        table.rows[r_idx].cells[1].text = col2

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _build_two_page_pdf_bytes() -> bytes:
    """Tạo file .pdf 2 trang trong RAM để kiểm tra gộp luồng liên trang."""
    pdf = pymupdf.open()
    page1 = pdf.new_page()
    page1.insert_text((72, 72), "Trang 1: Sinh vien dang ky hoc phan")
    page2 = pdf.new_page()
    page2.insert_text((72, 72), "Trang 2: toi thieu 14 tin chi moi ky.")
    return pdf.tobytes()


def test_convert_docx_preserves_heading_and_markdown_table():
    docx_bytes = _build_sample_docx_bytes()
    md = converter_registry.convert_to_markdown(docx_bytes, "hoc_phi.docx")

    assert "# Quy định học phí năm học 2025" in md
    assert "| Ngành đào tạo | Đơn giá (VNĐ/tín chỉ) |" in md
    assert "| Công nghệ thông tin | 650.000 |" in md
    assert "| Kỹ thuật dữ liệu | 680.000 |" in md


def test_convert_pdf_merges_pages_into_single_markdown_stream():
    pdf_bytes = _build_two_page_pdf_bytes()
    md = converter_registry.convert_to_markdown(pdf_bytes, "quy_che.pdf")

    assert "Sinh vien dang ky hoc phan" in md
    assert "toi thieu 14 tin chi moi ky." in md


def test_convert_unsupported_extension_raises_error():
    with pytest.raises(DocumentLoadError):
        converter_registry.convert_to_markdown(b"binary", "bang_diem.xlsx")


def test_convert_docx_strips_embedded_base64_images():
    """Ảnh nhúng trong .docx phải bị bỏ qua, không sinh chuỗi base64 khổng lồ vào Markdown."""
    # Tạo 1 PNG 1x1 tối giản và nhúng vào .docx
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00"
        b"\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    document = docx.Document()
    document.add_heading("Chương trình đào tạo", level=1)
    document.add_picture(io.BytesIO(png_1x1))
    document.add_paragraph("Tổng số tín chỉ: 150")
    buf = io.BytesIO()
    document.save(buf)

    md = converter_registry.convert_to_markdown(buf.getvalue(), "ctdt.docx")

    assert "data:image" not in md
    assert "base64" not in md
    assert "# Chương trình đào tạo" in md
    assert "Tổng số tín chỉ: 150" in md
