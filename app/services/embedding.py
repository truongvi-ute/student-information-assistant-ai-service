import time

import google.generativeai as genai

from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import logger


class EmbeddingService:
    """Quản lý việc chuyển đổi văn bản sang vector embedding qua Google Gemini API."""

    RATE_LIMIT_WAIT_SECONDS = 61.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        max_retries: int = 3,
        timeout_seconds: float | None = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.expected_dim = settings.QDRANT_VECTOR_SIZE
        self.rpm_limit = settings.EMBEDDING_RPM_LIMIT if rpm_limit is None else rpm_limit
        self.tpm_limit = (
            (0 if self.rpm_limit <= 0 else settings.EMBEDDING_TPM_LIMIT)
            if tpm_limit is None
            else tpm_limit
        )
        self.max_retries = max(0, max_retries)
        self.timeout_seconds = (
            settings.GEMINI_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )

        self._configured = False
        self._window_start: float | None = None
        self._window_count = 0
        self._window_tokens = 0

    def _ensure_configured(self) -> None:
        """Cấu hình Gemini API key theo cơ chế lazy loading với HTTP REST transport."""
        if not self._configured:
            genai.configure(api_key=self.api_key, transport="rest")
            self._configured = True

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.expected_dim:
            raise EmbeddingError(
                f"Kích thước vector ({len(vector)}) không khớp với Qdrant ({self.expected_dim})"
            )

    @staticmethod
    def _estimate_tokens(texts: list[str]) -> int:
        """Ước lượng an toàn số token cho văn bản tiếng Việt/bảng biểu (~1 ký tự ≈ 1 token)."""
        return sum(max(1, len(t)) for t in texts)

    @staticmethod
    def _is_rate_limit_error(err_msg: str) -> bool:
        lowered = err_msg.lower()
        return (
            "429" in err_msg
            or "quota" in lowered
            or "rate limit" in lowered
            or "resource exhausted" in lowered
        )

    def _reset_rate_window(self) -> None:
        self._window_start = None
        self._window_count = 0
        self._window_tokens = 0

    def _wait_for_rate_window(self, count: int, estimated_tokens: int = 0) -> None:
        """Chủ động chờ sang cửa sổ 60s mới nếu sắp vượt ngưỡng rpm_limit hoặc tpm_limit."""
        if self.rpm_limit <= 0 and self.tpm_limit <= 0:
            return

        now = time.monotonic()
        if self._window_start is None or now - self._window_start >= 60.0:
            self._window_start = now
            self._window_count = 0
            self._window_tokens = 0

        exceeds_rpm = self.rpm_limit > 0 and self._window_count + count > self.rpm_limit
        exceeds_tpm = (
            self.tpm_limit > 0
            and self._window_tokens > 0
            and self._window_tokens + estimated_tokens > self.tpm_limit
        )

        if exceeds_rpm or exceeds_tpm:
            # Cộng thêm 1s đệm (61s) để chắc chắn cửa sổ 1 phút của Gemini đã reset
            wait_seconds = max(0.0, self.RATE_LIMIT_WAIT_SECONDS - (now - self._window_start))
            if wait_seconds > 0:
                logger.info(
                    "Sắp chạm ngưỡng phút (%d/%d chunks, ~%d/%d tokens). Chờ %.1fs trước khi gửi tiếp %d chunks...",
                    self._window_count,
                    self.rpm_limit,
                    self._window_tokens,
                    self.tpm_limit,
                    wait_seconds,
                    count,
                )
                time.sleep(wait_seconds)
            self._window_start = time.monotonic()
            self._window_count = 0
            self._window_tokens = 0

        self._window_count += count
        self._window_tokens += estimated_tokens

    def _call_embed(
        self,
        content: str | list[str],
        task_type: str,
        error_prefix: str,
    ) -> list[float] | list[list[float]]:
        """Gọi Gemini Embedding API; nếu gặp lỗi 429 theo phút thì chờ 61s rồi thử lại."""
        self._ensure_configured()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=content,
                    task_type=task_type,
                    output_dimensionality=self.expected_dim,
                    request_options={"timeout": self.timeout_seconds},
                )
                return result["embedding"]
            except Exception as e:
                last_error = e
                err_msg = str(e)
                if attempt < self.max_retries and self._is_rate_limit_error(err_msg):
                    logger.warning(
                        "Gemini Embedding chạm giới hạn 429 (lần %d/%d). Chờ %.0fs sang phút mới rồi thử lại...",
                        attempt + 1,
                        self.max_retries,
                        self.RATE_LIMIT_WAIT_SECONDS,
                    )
                    time.sleep(self.RATE_LIMIT_WAIT_SECONDS)
                    self._reset_rate_window()
                    continue

                logger.error("Lỗi khi gọi Gemini Embedding API: %s", err_msg)
                raise EmbeddingError(f"{error_prefix}: {e}") from e

        raise EmbeddingError(f"{error_prefix}: {last_error}")

    def embed_texts(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """Tạo embedding cho danh sách đoạn văn bản theo từng mẻ (batch) vừa giới hạn TPM/RPM."""
        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0")

        effective_batch_size = (
            min(batch_size, self.rpm_limit) if self.rpm_limit > 0 else batch_size
        )
        all_embeddings: list[list[float]] = []

        idx = 0
        total = len(texts)
        while idx < total:
            batch: list[str] = []
            batch_tokens = 0
            while idx < total and len(batch) < effective_batch_size:
                next_text = texts[idx]
                next_tokens = max(1, len(next_text))
                if (
                    batch
                    and self.tpm_limit > 0
                    and batch_tokens + next_tokens > self.tpm_limit
                ):
                    break
                batch.append(next_text)
                batch_tokens += next_tokens
                idx += 1

            self._wait_for_rate_window(len(batch), estimated_tokens=batch_tokens)
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
