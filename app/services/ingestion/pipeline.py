import uuid
from pathlib import Path
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

    def ingest_path(self, target_path: str | Path, recreate_collection: bool = False) -> IngestionResult:
        """Nạp tài liệu từ một file cụ thể hoặc toàn bộ thư mục vào Qdrant."""
        path = Path(target_path)
        if not path.exists():
            raise IngestionError(f"Đường dẫn không tồn tại: {path}")

        # 1. Khởi tạo collection nếu chưa có
        init_collection(self.qdrant_client, recreate=recreate_collection)

        # 2. Thu thập danh sách tài liệu từ file hoặc thư mục
        extracted_documents: list[ExtractedDocument] = []
        failed_files: list[str] = []
        files_to_process: list[Path] = []

        if path.is_file():
            files_to_process = [path]
        else:
            files_to_process = [
                f for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in self.document_loader.SUPPORTED_EXTENSIONS
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
            return IngestionResult(total_files=len(files_to_process), failed_files=failed_files, status="empty")

        # 3. Cắt nhỏ tài liệu thành chunks
        chunks = self.text_chunker.split_documents(extracted_documents)
        if not chunks:
            logger.warning("Không tạo được chunk nào từ các tài liệu.")
            return IngestionResult(total_files=len(files_to_process), failed_files=failed_files, status="no_chunks")

        # 4. Tạo toạ độ vector qua Embedding Service
        logger.info("Đang tạo embedding cho %d chunks...", len(chunks))
        texts = [c.content for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # 5. Đóng gói PointStruct kèm ID xác định (UUIDv5)
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            src = chunk.metadata.get("source", "unknown")
            point_id = self.generate_point_id(src, chunk.chunk_index)
            payload = {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # 6. Đẩy vào Qdrant (Upsert theo từng mẻ 100 điểm)
        logger.info("Đang lưu %d points vào Qdrant collection '%s'...", len(points), settings.QDRANT_COLLECTION)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=batch,
            )

        logger.info("Nạp thành công %d chunks vào Qdrant.", len(points))
        return IngestionResult(
            total_files=len(files_to_process),
            total_chunks=len(chunks),
            successful_chunks=len(points),
            failed_files=failed_files,
            status="success",
        )


# Instance mặc định dùng trong toàn bộ ứng dụng
ingestion_pipeline = IngestionPipeline()
