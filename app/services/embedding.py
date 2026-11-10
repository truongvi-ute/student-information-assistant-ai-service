import google.generativeai as genai
from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import logger


class EmbeddingService:
    """Quản lý việc chuyển đổi văn bản sang vector embedding qua Google Gemini API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.expected_dim = settings.QDRANT_VECTOR_SIZE

        self._configured = False

    def _ensure_configured(self) -> None:
        """Cấu hình Gemini API key theo cơ chế lazy loading."""
        if not self._configured:
            genai.configure(api_key=self.api_key)
            self._configured = True

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Tạo embedding cho danh sách đoạn văn bản theo từng mẻ (batch)."""
        if not texts:
            return []

        self._ensure_configured()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=batch,
                    task_type="retrieval_document",
                    output_dimensionality=self.expected_dim,
                )
                # Gemini trả về dict {"embedding": [...]} khi input là list
                batch_embeddings: list[list[float]] = result["embedding"]

                # Kiểm tra số chiều vector để đảm bảo khớp với Qdrant
                for emb in batch_embeddings:
                    if len(emb) != self.expected_dim:
                        raise EmbeddingError(
                            f"Kích thước vector ({len(emb)}) không khớp với Qdrant ({self.expected_dim})"
                        )

                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Lỗi khi gọi Gemini Embedding API: %s", str(e))
                raise EmbeddingError(f"Lỗi tạo embedding: {e}") from e

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Tạo embedding cho một câu truy vấn tìm kiếm."""
        self._ensure_configured()
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                # task_type khác khi embed câu hỏi (retrieval_query) so với nạp tài liệu
                task_type="retrieval_query",
                output_dimensionality=self.expected_dim,
            )
            vector: list[float] = result["embedding"]
            if len(vector) != self.expected_dim:
                raise EmbeddingError(
                    f"Kích thước vector ({len(vector)}) không khớp với Qdrant ({self.expected_dim})"
                )
            return vector
        except Exception as e:
            logger.error("Lỗi khi gọi Gemini Embedding API: %s", str(e))
            raise EmbeddingError(f"Lỗi tạo embedding query: {e}") from e


# Instance mặc định dùng cho toàn ứng dụng
embedding_service = EmbeddingService()
