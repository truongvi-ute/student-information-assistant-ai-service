import io
import re
from abc import ABC, abstractmethod
from pathlib import Path

import mammoth
import pymupdf
import pymupdf4llm
from markdownify import markdownify as html_to_markdown

from app.core.exceptions import DocumentLoadError


def normalize_markdown(text: str) -> str:
    """Chuẩn hóa khoảng trắng và xuống dòng thừa mà không phá vỡ cấu trúc Markdown."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Gộp từ 3 dòng trống liên tiếp trở lên thành 2 dòng trống
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class BaseDocumentConverter(ABC):
    """Interface chung cho mọi bộ chuyển đổi tài liệu sang Markdown."""

    @abstractmethod
    def convert(self, file_bytes: bytes, filename: str) -> str:
        """Nhận nội dung nhị phân của file và trả về chuỗi Markdown chuẩn hóa."""


class TextMarkdownConverter(BaseDocumentConverter):
    """Chuyển đổi file văn bản thuần (.txt) và Markdown (.md) đọc theo UTF-8."""

    def convert(self, file_bytes: bytes, filename: str) -> str:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise DocumentLoadError(
                f"File '{filename}' không đúng bảng mã UTF-8: {e}"
            ) from e
        return normalize_markdown(text)


class PdfMarkdownConverter(BaseDocumentConverter):
    """Chuyển đổi toàn bộ PDF thành một luồng Markdown liên tục, giữ bảng biểu."""

    def convert(self, file_bytes: bytes, filename: str) -> str:
        try:
            with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf_doc:
                if pdf_doc.page_count == 0:
                    return ""
                markdown_text = pymupdf4llm.to_markdown(
                    pdf_doc,
                    show_progress=False,
                    use_ocr=False,
                )
            return normalize_markdown(markdown_text)
        except Exception as e:
            raise DocumentLoadError(
                f"Không thể chuyển đổi PDF '{filename}' sang Markdown: {e}"
            ) from e


class DocxMarkdownConverter(BaseDocumentConverter):
    """Chuyển đổi file Word (.docx) sang Markdown thông qua HTML trung gian."""

    def convert(self, file_bytes: bytes, filename: str) -> str:
        try:
            with io.BytesIO(file_bytes) as stream:
                result = mammoth.convert_to_html(stream)
                html_content = result.value

            if not html_content or not html_content.strip():
                return ""

            markdown_text = html_to_markdown(
                html_content,
                heading_style="ATX",
                bullets="-",
            )
            return normalize_markdown(markdown_text)
        except Exception as e:
            raise DocumentLoadError(
                f"Không thể chuyển đổi DOCX '{filename}' sang Markdown: {e}"
            ) from e


class DocumentConverterRegistry:
    """Ánh xạ đuôi file sang Converter tương ứng (Strategy Registry)."""

    def __init__(self) -> None:
        text_converter = TextMarkdownConverter()
        self._converters: dict[str, BaseDocumentConverter] = {
            ".txt": text_converter,
            ".md": text_converter,
            ".pdf": PdfMarkdownConverter(),
            ".docx": DocxMarkdownConverter(),
        }

    @property
    def supported_extensions(self) -> set[str]:
        return set(self._converters.keys())

    def get_converter(self, filename: str) -> BaseDocumentConverter:
        ext = Path(filename).suffix.lower()
        converter = self._converters.get(ext)
        if converter is None:
            raise DocumentLoadError(
                f"Định dạng '{ext}' chưa được hỗ trợ. Chỉ nhận: {sorted(self.supported_extensions)}"
            )
        return converter

    def convert_to_markdown(self, file_bytes: bytes, filename: str) -> str:
        if not file_bytes:
            return ""
        converter = self.get_converter(filename)
        return converter.convert(file_bytes, filename)


converter_registry = DocumentConverterRegistry()
