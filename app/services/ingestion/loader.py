from pathlib import Path
from pypdf import PdfReader
from app.core.exceptions import DocumentLoadError
from app.core.logging import logger
from app.schemas.document import ExtractedDocument


class DocumentLoader:
    """Đọc và trích xuất nội dung từ một file tài liệu (.txt, .md, .pdf)."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def load_file(self, file_path: str | Path) -> list[ExtractedDocument]:
        """Đọc một file tài liệu cụ thể và trả về danh sách ExtractedDocument."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentLoadError(f"File không tồn tại: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(f"Định dạng '{ext}' chưa được hỗ trợ. Chỉ nhận: {self.SUPPORTED_EXTENSIONS}")

        try:
            if ext in {".txt", ".md"}:
                return self._load_text(path)
            elif ext == ".pdf":
                return self._load_pdf(path)
        except Exception as e:
            logger.error("Lỗi khi đọc file %s: %s", path.name, str(e))
            raise DocumentLoadError(f"Không thể đọc file {path.name}: {e}") from e

        return []

    def _load_text(self, path: Path) -> list[ExtractedDocument]:
        """Trích xuất nội dung từ file text/markdown với UTF-8."""
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("File văn bản rỗng: %s", path.name)
            return []

        metadata = {
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": path.suffix.lower(),
        }
        return [ExtractedDocument(content=text, metadata=metadata)]

    def _load_pdf(self, path: Path) -> list[ExtractedDocument]:
        """Trích xuất nội dung từng trang của file PDF kèm thông tin số trang."""
        reader = PdfReader(str(path))
        extracted_docs: list[ExtractedDocument] = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue

            metadata = {
                "source": str(path.resolve()),
                "filename": path.name,
                "extension": ".pdf",
                "page_number": page_idx + 1,
                "total_pages": len(reader.pages),
            }
            extracted_docs.append(ExtractedDocument(content=page_text, metadata=metadata))

        if not extracted_docs:
            logger.warning("File PDF không trích xuất được chữ (hoặc file rỗng): %s", path.name)

        return extracted_docs


# Instance mặc định dùng cho toàn ứng dụng
document_loader = DocumentLoader()