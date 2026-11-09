import uuid
from pathlib import Path
from qdrant_client.http.models import PointStruct
from app.core.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.db.qdrant import get_qdrant_client, init_collection
from app.schemas.document import IngestionResult
from app.services.embedding import EmbeddingService, embedding_service
from app.services.ingestion.chunker import TextChunker, chunker
from app.services.ingestion.loader import DocumentLoader, loader


class IngestionPipeline:
    """Điều phối toàn bộ dây chuyền nạp tài liệu vào Qdrant Vector DB."""

    def __init__(
        self,
        doc_loader: DocumentLoader | None = None,
        text_chunker: TextChunker | None = None,
        embed_service: EmbeddingService | None = None,
    ):
        self.loader = doc_loader or loader
        self.chunker = text_chunker or chunker
        self.embedding = embed_service or embedding_service
        self.client = get_qdrant_client()

    @staticmethod
    def generate_point_id(source: str, chunk_index: int) -> str:
        """Sinh Point ID xác định (UUIDv5) từ nguồn và số thứ tự chunk để đảm bảo tính idempotent."""
        unique_key = f"{source}::{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_key))

    def ingest_file(self, file_path: str | Path, recreate_collection: bool = False) -> IngestionResult:
        """Nạp một file tài liệu cụ thể vào Qdrant."""
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"File không tồn tại: {path}")

        # 1. Đảm bảo collection trên Qdrant đã sẵn sàng
        init_collection(self.client, recreate=recreate_collection)

        # 2. Bước 1: Đọc tài liệu qua Loader
        try:
            documents = self.loader.load_file(path)
        except Exception as e:
            logger.error("Lỗi khi đọc file %s: %s", path.name, str(e))
            return IngestionResult(total_files=1, failed_files=[path.name], status="failed")

        if not documents:
            logger.warning("File rỗng hoặc không có nội dung chữ: %s", path.name)
            return IngestionResult(total_files=1, failed_files=[path.name], status="empty")

        # 3. Bước 2: Cắt nhỏ tài liệu qua Chunker
        chunks = self.chunker.split_documents(documents)
        if not chunks:
            logger.warning("Không tạo được chunk nào từ file: %s", path.name)
            return IngestionResult(total_files=1, failed_files=[path.name], status="no_chunks")

        # 4. Bước 3: Tạo toạ độ vector qua Embedding Service
        logger.info("Đang tạo embedding cho %d chunks của file %s...", len(chunks), path.name)
        texts = [c.content for c in chunks]
        embeddings = self.embedding.embed_texts(texts)

        # 5. Bước 4: Đóng gói PointStruct kèm ID xác định
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            src = chunk.metadata.get("source", str(path.resolve()))
            point_id = self.generate_point_id(src, chunk.chunk_index)
            payload = {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # 6. Bước 5: Đẩy vào Qdrant (Upsert theo từng mẻ 100 điểm)
        logger.info("Đang lưu %d points vào Qdrant collection '%s'...", len(points), settings.QDRANT_COLLECTION)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=batch,
            )

        logger.info("Nạp thành công file %s (%d chunks).", path.name, len(points))
        return IngestionResult(
            total_files=1,
            total_chunks=len(chunks),
            successful_chunks=len(points),
            status="success",
        )


# Instance mặc định dùng trong toàn bộ ứng dụng
pipeline = IngestionPipeline()