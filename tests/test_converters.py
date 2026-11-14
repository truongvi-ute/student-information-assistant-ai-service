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
