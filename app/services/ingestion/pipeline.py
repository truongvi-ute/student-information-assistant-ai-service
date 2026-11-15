import uuid
from pathlib import Path
from typing import Any

from qdrant_client.http.models import PointStruct

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.db.qdrant import get_qdrant_client, init_collection
from app.schemas.document import ExtractedDocument, IngestionResult
from app.services.embedding import EmbeddingService, embedding_service
from app.services.ingestion.chunker import TextChunker, text_chunker
from app.services.ingestion.loader import DocumentLoader, document_loader


class IngestionPipeline:
    """Điều phối toàn bộ dây chuyền nạp tài liệu vào Qdrant Vector DB."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embed_service: EmbeddingService | None = None,
    ):
        self.document_loader = loader or document_loader
        self.text_chunker = chunker or text_chunker
        self.embedding_service = embed_service or embedding_service
        self.qdrant_client = get_qdrant_client()

    @staticmethod
    def generate_point_id(source: str, chunk_index: int) -> str:
        """Sinh Point ID xác định (UUIDv5) từ nguồn và số thứ tự chunk để đảm bảo tính idempotent."""
        unique_key = f"{source}::{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_key))

    def _index_documents(
        self,
        extracted_documents: list[ExtractedDocument],
        total_files: int,
        failed_files: list[str],
    ) -> IngestionResult:
        """Thực thi phần chung: cắt chunk -> tạo embedding -> upsert vào Qdrant."""
        if not extracted_documents:
            return IngestionResult(
                total_files=total_files,
                failed_files=failed_files,
                status="empty",
            )

        chunks = self.text_chunker.split_documents(extracted_documents)
        if not chunks:
            logger.warning("Không tạo được chunk nào từ các tài liệu.")
            return IngestionResult(
                total_files=total_files,
                failed_files=failed_files,
                status="no_chunks",
            )

        logger.info("Đang tạo embedding cho %d chunks...", len(chunks))
        texts = [c.content for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            doc_id = chunk.metadata.get("document_id")
            id_source = (
                f"doc:{doc_id}"
                if doc_id
                else chunk.metadata.get("source", "unknown")
            )
            point_id = self.generate_point_id(str(id_source), chunk.chunk_index)
            payload = {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        logger.info(
            "Đang lưu %d points vào Qdrant collection '%s'...",
            len(points),
            settings.QDRANT_COLLECTION,
        )
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=batch,
            )

        logger.info("Nạp thành công %d chunks vào Qdrant.", len(points))
        return IngestionResult(
            total_files=total_files,
            total_chunks=len(chunks),
            successful_chunks=len(points),
            failed_files=failed_files,
            status="success",
        )

    def ingest_path(
        self, target_path: str | Path, recreate_collection: bool = False
    ) -> IngestionResult:
        """Nạp tài liệu từ một file cục bộ hoặc toàn bộ thư mục vào Qdrant."""
        path = Path(target_path)
        if not path.exists():
            raise IngestionError(f"Đường dẫn không tồn tại: {path}")

        init_collection(self.qdrant_client, recreate=recreate_collection)

        extracted_documents: list[ExtractedDocument] = []
        failed_files: list[str] = []

        if path.is_file():
            files_to_process = [path]
        else:
            files_to_process = [
                f
                for f in path.rglob("*")
                if f.is_file()
                and f.suffix.lower() in self.document_loader.SUPPORTED_EXTENSIONS
            ]

        for file_item in files_to_process:
            try:
                loaded_docs = self.document_loader.load_file(file_item)
                extracted_documents.extend(loaded_docs)
            except Exception as e:
                logger.warning("Bỏ qua file lỗi %s: %s", file_item.name, str(e))
                failed_files.append(file_item.name)

        if not extracted_documents:
            logger.warning("Không có tài liệu hợp lệ nào để nạp từ %s", path.name)

        return self._index_documents(
            extracted_documents=extracted_documents,
            total_files=len(files_to_process),
            failed_files=failed_files,
        )

    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
        recreate_collection: bool = False,
    ) -> IngestionResult:
        """Nạp trực tiếp dữ liệu nhị phân (từ UploadFile hoặc Storage) vào Qdrant."""
        if not filename or not filename.strip():
            raise IngestionError("Thiếu tên file (filename) khi nạp tài liệu.")

        init_collection(self.qdrant_client, recreate=recreate_collection)

        try:
            loaded_docs = self.document_loader.load_bytes(
                file_bytes=file_bytes,
                filename=filename,
                extra_metadata=metadata,
            )
        except Exception as e:
            logger.error("Lỗi khi nạp file từ bytes '%s': %s", filename, str(e))
            raise IngestionError(f"Không thể nạp file '{filename}': {e}") from e

        return self._index_documents(
            extracted_documents=loaded_docs,
            total_files=1,
            failed_files=[],
        )


# Instance mặc định dùng trong toàn bộ ứng dụng
ingestion_pipeline = IngestionPipeline()
