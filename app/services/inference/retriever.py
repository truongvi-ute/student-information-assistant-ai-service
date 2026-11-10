from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import logger
from app.db.qdrant import get_qdrant_client
from app.schemas.response import RetrievedChunk
from app.services.embedding import EmbeddingService, embedding_service


class VectorRetriever:
    """Truy xuất các đoạn văn bản liên quan nhất từ Qdrant Vector DB."""

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embed_service: EmbeddingService | None = None,
    ):
        self._custom_client = qdrant_client
        self.embedding_service = embed_service or embedding_service
        self.collection_name = settings.QDRANT_COLLECTION

    @property
    def client(self) -> QdrantClient:
        if self._custom_client is not None:
            return self._custom_client
        return get_qdrant_client()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filename_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Tìm top_k đoạn văn bản có ngữ nghĩa gần nhất với câu truy vấn."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        try:
            # 1. Kiểm tra collection có tồn tại không (nếu chưa có thì trả danh sách rỗng, không lỗi 500)
            if not self.client.collection_exists(self.collection_name):
                logger.warning("Collection '%s' chưa tồn tại khi truy xuất.", self.collection_name)
                return []

            # 2. Chuyển câu hỏi thành vector 768 chiều
            query_vector = self.embedding_service.embed_query(cleaned_query)

            # 3. Xây dựng bộ lọc metadata nếu người dùng truyền filename_filter (lọc theo filename)
            qdrant_filter: Filter | None = None
            if filename_filter:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename_filter),
                        )
                    ]
                )

            # 4. Truy vấn Qdrant bằng phương thức query_points
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # 5. Chuyển đổi kết quả ScoredPoint sang schema RetrievedChunk
            chunks: list[RetrievedChunk] = []
            for point in response.points:
                payload = point.payload or {}
                content = payload.get("content", "")
                if not content:
                    continue

                chunks.append(
                    RetrievedChunk(
                        id=str(point.id),
                        content=content,
                        score=float(point.score),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        filename=payload.get("filename"),
                        page_number=payload.get("page_number"),
                        metadata=payload,
                    )
                )

            logger.info("Truy xuất được %d chunks cho câu hỏi: '%s'", len(chunks), cleaned_query[:50])
            return chunks

        except Exception as e:
            logger.error("Lỗi khi truy xuất Qdrant: %s", str(e))
            raise RetrievalError(f"Không thể truy xuất dữ liệu từ Qdrant: {e}") from e


# Instance mặc định dùng cho toàn ứng dụng
vector_retriever = VectorRetriever()
