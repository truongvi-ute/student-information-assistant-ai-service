from pathlib import Path
from typing import Any, ClassVar

from app.core.exceptions import DocumentLoadError
from app.core.logging import logger
from app.schemas.document import ExtractedDocument
from app.services.ingestion.converters import (
    DocumentConverterRegistry,
    converter_registry,
)


class DocumentLoader:
    """Đọc và chuẩn hóa tài liệu (.txt, .md, .pdf, .docx) về định dạng Markdown."""

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".txt", ".md", ".pdf", ".docx"}

    def __init__(self, registry: DocumentConverterRegistry | None = None) -> None:
        self.registry = registry or converter_registry

    def load_file(self, file_path: str | Path) -> list[ExtractedDocument]:
        """Đọc một file từ đường dẫn cục bộ và chuyển thành ExtractedDocument Markdown."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentLoadError(f"File không tồn tại: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Định dạng '{ext}' chưa được hỗ trợ. Chỉ nhận: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        file_bytes = path.read_bytes()
        extra_metadata = {
            "source": str(path.resolve()),
        }
        return self.load_bytes(
            file_bytes=file_bytes,
            filename=path.name,
            extra_metadata=extra_metadata,
        )

    def load_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ExtractedDocument]:
        """Chuyển đổi nội dung bytes của file thành ExtractedDocument Markdown liên tục."""
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Định dạng '{ext}' chưa được hỗ trợ. Chỉ nhận: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if not file_bytes:
            logger.warning("File rỗng (0 bytes): %s", filename)
            return []

        try:
            markdown_text = self.registry.convert_to_markdown(file_bytes, filename)
        except DocumentLoadError:
            raise
        except Exception as e:
            logger.error("Lỗi khi trích xuất file %s: %s", filename, str(e))
            raise DocumentLoadError(f"Không thể đọc file {filename}: {e}") from e

        if not markdown_text:
            logger.warning("File không trích xuất được nội dung văn bản: %s", filename)
            return []

        metadata: dict[str, Any] = {
            "source": filename,
            "filename": filename,
            "extension": ext,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return [ExtractedDocument(content=markdown_text, metadata=metadata)]


# Instance mặc định dùng cho toàn ứng dụng
document_loader = DocumentLoader()
