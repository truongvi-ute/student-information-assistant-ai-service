import uuid
from pathlib import Path
from typing import Any

from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.db.qdrant import get_qdrant_client, init_collection
from app.schemas.document import ExtractedDocument, IngestionResult
from app.services.embedding import EmbeddingService, embedding_service
from app.services.ingestion.chunker import TextChunker, text_chunker
from app.services.ingestion.loader import DocumentLoader, document_loader
from app.services.storage import StorageAdapter, storage_adapter


class IngestionPipeline:
    """Điều phối toàn bộ dây chuyền nạp tài liệu vào Qdrant Vector DB."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embed_service: EmbeddingService | None = None,
        storage: StorageAdapter | None = None,
    ):
        self.document_loader = loader or document_loader
        self.text_chunker = chunker or text_chunker
        self.embedding_service = embed_service or embedding_service
        self.storage_adapter = storage or storage_adapter
        self.qdrant_client = get_qdrant_client()

    @staticmethod
    def generate_point_id(source: str, chunk_index: int) -> str:
        """Sinh Point ID xác định (UUIDv5) từ nguồn và số thứ tự chunk để đảm bảo tính idempotent."""
        unique_key = f"{source}::{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_key))

    @staticmethod
    def build_extracted_markdown_key(document_id: str) -> str:
        """Quy ước object key của bản Markdown chuẩn hóa trên RustFS/S3."""
        return f"extracted/{document_id}.md"

    def _delete_existing_document_points(self, document_id: str) -> None:
        """Xóa toàn bộ points cũ của document_id trong Qdrant trước khi upsert bản mới."""
        self.qdrant_client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def _index_documents(
        self,
        extracted_documents: list[ExtractedDocument],
        total_files: int,
        failed_files: list[str],
        upload_extracted_markdown: bool = False,
    ) -> IngestionResult:
        """Thực thi phần chung: cắt chunk -> tạo embedding -> lưu Markdown (nếu cần) -> upsert vào Qdrant."""
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

        doc_contents: dict[str, str] = {}
        for doc in extracted_documents:
            raw_doc_id = doc.metadata.get("document_id")
            if raw_doc_id:
                doc_contents[str(raw_doc_id)] = doc.content

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
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        for doc_id in doc_contents:
            self._delete_existing_document_points(doc_id)

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

        if upload_extracted_markdown:
            for doc_id, markdown_content in doc_contents.items():
                extracted_key = self.build_extracted_markdown_key(doc_id)
                self.storage_adapter.upload_text(extracted_key, markdown_content)
                logger.info("Đã lưu bản Markdown chuẩn hóa lên Storage: %s", extracted_key)

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
        upload_extracted_markdown: bool = False,
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
            upload_extracted_markdown=upload_extracted_markdown,
        )

    def ingest_from_storage(
        self,
        document_id: str,
        file_key: str,
        filename: str,
        metadata: dict[str, Any] | None = None,
        recreate_collection: bool = False,
    ) -> IngestionResult:
        """Kéo file từ RustFS/S3 theo file_key, lưu bản Markdown chuẩn hóa và nạp vào Qdrant."""
        if not document_id or not document_id.strip():
            raise IngestionError("Thiếu document_id khi nạp tài liệu từ Storage.")

        cleaned_doc_id = document_id.strip()
        file_bytes = self.storage_adapter.download_file(file_key)
        merged_metadata: dict[str, Any] = {
            **(metadata or {}),
            "document_id": cleaned_doc_id,
            "storage_path": file_key,
        }
        return self.ingest_bytes(
            file_bytes=file_bytes,
            filename=filename,
            metadata=merged_metadata,
            recreate_collection=recreate_collection,
            upload_extracted_markdown=True,
        )


# Instance mặc định dùng trong toàn bộ ứng dụng
ingestion_pipeline = IngestionPipeline()
