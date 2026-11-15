import time

import google.generativeai as genai
from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import logger


class EmbeddingService:
    """Quản lý việc chuyển đổi văn bản sang vector embedding qua Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rpm_limit: int | None = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.expected_dim = settings.QDRANT_VECTOR_SIZE
        self.rpm_limit = settings.EMBEDDING_RPM_LIMIT if rpm_limit is None else rpm_limit

        self._configured = False
        self._window_start: float | None = None
        self._window_count = 0

    def _ensure_configured(self) -> None:
        """Cấu hình Gemini API key theo cơ chế lazy loading."""
        if not self._configured:
            genai.configure(api_key=self.api_key)
            self._configured = True

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.expected_dim:
            raise EmbeddingError(
                f"Kích thước vector ({len(vector)}) không khớp với Qdrant ({self.expected_dim})"
            )

    def _wait_for_rate_window(self, count: int) -> None:
        """Chủ động chờ sang cửa sổ 60s mới nếu sắp vượt ngưỡng rpm_limit."""
        if self.rpm_limit <= 0:
            return

        now = time.monotonic()
        if self._window_start is None or now - self._window_start >= 60.0:
            self._window_start = now
            self._window_count = 0

        if self._window_count + count > self.rpm_limit:
            # Cộng thêm 1s đệm (61s) để chắc chắn cửa sổ 1 phút của Gemini đã reset
            wait_seconds = max(0.0, 61.0 - (now - self._window_start))
            if wait_seconds > 0:
                logger.info(
                    "Đã gửi %d/%d chunks trong phút hiện tại. Chờ %.1fs trước khi gửi tiếp %d chunks...",
                    self._window_count,
                    self.rpm_limit,
                    wait_seconds,
                    count,
                )
                time.sleep(wait_seconds)
            self._window_start = time.monotonic()
            self._window_count = 0

        self._window_count += count

    def _call_embed(
        self,
        content: str | list[str],
        task_type: str,
        error_prefix: str,
    ) -> list[float] | list[list[float]]:
        """Gọi Gemini Embedding API và chuẩn hóa lỗi trả về."""
        self._ensure_configured()
        try:
            result = genai.embed_content(
                model=self.model,
                content=content,
                task_type=task_type,
                output_dimensionality=self.expected_dim,
            )
            return result["embedding"]
        except Exception as e:
            logger.error("Lỗi khi gọi Gemini Embedding API: %s", str(e))
            raise EmbeddingError(f"{error_prefix}: {e}") from e

    def embed_texts(self, texts: list[str], batch_size: int = 40) -> list[list[float]]:
        """Tạo embedding cho danh sách đoạn văn bản theo từng mẻ (batch)."""
        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0")

        effective_batch_size = (
            min(batch_size, self.rpm_limit) if self.rpm_limit > 0 else batch_size
        )
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i : i + effective_batch_size]
            self._wait_for_rate_window(len(batch))
            batch_embeddings: list[list[float]] = self._call_embed(
                content=batch,
                task_type="retrieval_document",
                error_prefix="Lỗi tạo embedding",
            )
            for emb in batch_embeddings:
                self._validate_vector(emb)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Tạo embedding cho một câu truy vấn tìm kiếm."""
        # task_type khác khi embed câu hỏi (retrieval_query) so với nạp tài liệu
        vector: list[float] = self._call_embed(
            content=query,
            task_type="retrieval_query",
            error_prefix="Lỗi tạo embedding query",
        )
        self._validate_vector(vector)
        return vector


# Instance mặc định dùng cho toàn ứng dụng
embedding_service = EmbeddingService()
