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
            effective_threshold = (
                settings.RETRIEVAL_SCORE_THRESHOLD
                if score_threshold is None
                else score_threshold
            )
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=effective_threshold,
                with_payload=True,
            )

            # 5. Chuyển đổi kết quả ScoredPoint sang schema RetrievedChunk
            chunks: list[RetrievedChunk] = []
            for point in response.points:
                payload = point.payload or {}
                content = payload.get("content", "")
                if not content:
                    continue

                raw_doc_id = payload.get("document_id")
                raw_char_start = payload.get("char_start")
                raw_char_end = payload.get("char_end")

                chunks.append(
                    RetrievedChunk(
                        id=str(point.id),
                        content=content,
                        score=float(point.score),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        document_id=str(raw_doc_id) if raw_doc_id is not None else None,
                        filename=payload.get("filename"),
                        page_number=payload.get("page_number"),
                        char_start=int(raw_char_start) if raw_char_start is not None else None,
                        char_end=int(raw_char_end) if raw_char_end is not None else None,
                        metadata=payload,
                    )
                )

            logger.info("Truy xuất được %d chunks cho câu hỏi: '%s'", len(chunks), cleaned_query[:50])
            return chunks

        except Exception as e:
            logger.error("Lỗi khi truy xuất Qdrant: %s", str(e))
            raise RetrievalError(f"Không thể truy xuất dữ liệu từ Qdrant: {e}") from e

    def retrieve_multi_queries(
        self,
        queries: list[str],
        top_k_per_query: int = 4,
        max_total_chunks: int = 8,
        filename_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Truy xuất Qdrant cho danh sách các câu truy vấn con, gộp cân bằng theo vòng (round-robin) và khử trùng lặp."""
        cleaned_queries = [q.strip() for q in queries if q and q.strip()]
        if not cleaned_queries:
            return []

        if len(cleaned_queries) == 1:
            return self.retrieve(
                query=cleaned_queries[0],
                top_k=top_k_per_query,
                filename_filter=filename_filter,
                score_threshold=score_threshold,
            )

        per_query_results: list[list[RetrievedChunk]] = []
        for sub_q in cleaned_queries:
            sub_chunks = self.retrieve(
                query=sub_q,
                top_k=top_k_per_query,
                filename_filter=filename_filter,
                score_threshold=score_threshold,
            )
            per_query_results.append(sub_chunks)

        # Gộp theo kiểu xen kẽ (round-robin) để đảm bảo mọi nhánh câu hỏi đều có đại diện
        merged_by_id: dict[str, RetrievedChunk] = {}
        max_len = max((len(lst) for lst in per_query_results), default=0)

        for idx in range(max_len):
            for chunk_list in per_query_results:
                if idx >= len(chunk_list):
                    continue
                candidate = chunk_list[idx]
                existing = merged_by_id.get(candidate.id)
                if existing is None:
                    if len(merged_by_id) < max_total_chunks:
                        merged_by_id[candidate.id] = candidate
                elif candidate.score > existing.score:
                    merged_by_id[candidate.id] = candidate

        return list(merged_by_id.values())


# Instance mặc định dùng cho toàn ứng dụng
vector_retriever = VectorRetriever()
