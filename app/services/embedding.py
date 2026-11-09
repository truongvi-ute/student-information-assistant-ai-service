from openai import OpenAI
from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import logger


class EmbeddingService:
    """Quản lý việc chuyển đổi văn bản sang vector embedding qua OpenAI API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model = model or settings.EMBEDDING_MODEL
        self.expected_dim = settings.QDRANT_VECTOR_SIZE

        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Khởi tạo OpenAI client theo cơ chế lazy loading."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key or "mock-key",
                base_url=self.base_url,
            )
        return self._client

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Tạo embedding cho danh sách đoạn văn bản theo từng mẻ (batch)."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                )
                batch_embeddings = [item.embedding for item in response.data]

                # Kiểm tra số chiều vector để đảm bảo khớp với Qdrant
                for emb in batch_embeddings:
                    if len(emb) != self.expected_dim:
                        raise EmbeddingError(
                            f"Kích thước vector ({len(emb)}) không khớp với Qdrant ({self.expected_dim})"
                        )

                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Lỗi khi gọi Embedding API: %s", str(e))
                raise EmbeddingError(f"Lỗi tạo embedding: {e}") from e

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Tạo embedding cho một câu truy vấn tìm kiếm."""
        embeddings = self.embed_texts([query])
        return embeddings[0]


# Đối tượng singleton dùng chung cho Ingestion và Retrieval
embedding_service = EmbeddingService()