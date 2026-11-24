import threading
import time

import google.generativeai as genai

from app.core.config import genai_config_lock, settings
from app.core.exceptions import EmbeddingError, EmbeddingRateLimitError
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
        max_query_wait_seconds: float | None = None,
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
        raw_max_wait = (
            settings.EMBEDDING_QUERY_MAX_WAIT_SECONDS
            if max_query_wait_seconds is None
            else max_query_wait_seconds
        )
        self.max_query_wait_seconds = (
            raw_max_wait if raw_max_wait is not None and raw_max_wait > 0 else None
        )

        self._lock = threading.Lock()
        self._configured = False
        self._window_start: float | None = None
        self._window_count = 0
        self._window_tokens = 0

    def _ensure_configured(self) -> None:
        """Cấu hình Gemini API key theo cơ chế lazy loading với HTTP REST transport."""
        if not self._configured:
            with self._lock:
                if not self._configured:
                    with genai_config_lock:
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

    def _wait_for_rate_window(
        self,
        count: int,
        estimated_tokens: int = 0,
        *,
        after_rate_limit: bool = False,
        max_wait_seconds: float | None = None,
    ) -> float:
        """Chủ động đặt chỗ trong cửa sổ 60s và chờ ngoài khóa nếu vượt ngưỡng hoặc vừa gặp 429."""
        if self.rpm_limit <= 0 and self.tpm_limit <= 0 and not after_rate_limit:
            return 0.0

        wait_seconds = 0.0
        prev_count = 0
        prev_tokens = 0

        with self._lock:
            now = time.monotonic()

            if after_rate_limit and (
                self._window_start is None or self._window_start <= now
            ):
                self._window_start = now + self.RATE_LIMIT_WAIT_SECONDS
                self._window_count = 0
                self._window_tokens = 0

            if self.rpm_limit <= 0 and self.tpm_limit <= 0:
                wait_seconds = max(0.0, (self._window_start or now) - now)
                if max_wait_seconds is not None and wait_seconds > max_wait_seconds:
                    raise EmbeddingRateLimitError(
                        "Hệ thống đang bận xử lý nhiều yêu cầu, vui lòng thử lại sau ít giây."
                    )
            else:
                if self._window_start is None or now - self._window_start >= 60.0:
                    current_start = now
                    current_count = 0
                    current_tokens = 0
                else:
                    current_start = self._window_start
                    current_count = self._window_count
                    current_tokens = self._window_tokens

                exceeds_rpm = (
                    self.rpm_limit > 0
                    and current_count > 0
                    and current_count + count > self.rpm_limit
                )
                exceeds_tpm = (
                    self.tpm_limit > 0
                    and current_tokens > 0
                    and current_tokens + estimated_tokens > self.tpm_limit
                )

                if exceeds_rpm or exceeds_tpm:
                    target_start = max(
                        now,
                        current_start + self.RATE_LIMIT_WAIT_SECONDS,
                    )
                    wait_seconds = target_start - now
                    if max_wait_seconds is not None and wait_seconds > max_wait_seconds:
                        raise EmbeddingRateLimitError(
                            "Hệ thống đang bận xử lý nhiều yêu cầu, vui lòng thử lại sau ít giây."
                        )
                    prev_count = current_count
                    prev_tokens = current_tokens
                    self._window_start = target_start
                    self._window_count = count
                    self._window_tokens = estimated_tokens
                else:
                    wait_seconds = max(0.0, current_start - now)
                    if max_wait_seconds is not None and wait_seconds > max_wait_seconds:
                        raise EmbeddingRateLimitError(
                            "Hệ thống đang bận xử lý nhiều yêu cầu, vui lòng thử lại sau ít giây."
                        )
                    prev_count = current_count
                    prev_tokens = current_tokens
                    self._window_start = current_start
                    self._window_count = current_count + count
                    self._window_tokens = current_tokens + estimated_tokens

        if wait_seconds > 0:
            logger.info(
                "Sắp chạm ngưỡng phút (%d/%d chunks, ~%d/%d tokens). Chờ %.1fs trước khi gửi tiếp %d chunks...",
                prev_count,
                self.rpm_limit,
                prev_tokens,
                self.tpm_limit,
                wait_seconds,
                count,
            )
            time.sleep(wait_seconds)

        return wait_seconds

    def _call_embed(
        self,
        content: str | list[str],
        task_type: str,
        error_prefix: str,
        *,
        item_count: int = 1,
        estimated_tokens: int = 1,
        max_wait_seconds: float | None = None,
    ) -> list[float] | list[list[float]]:
        """Gọi Gemini Embedding API kèm đặt chỗ giới hạn tốc độ cho cả lượt đầu và các lần retry."""
        self._ensure_configured()
        last_error: Exception | None = None
        remaining_wait = max_wait_seconds
        after_rate_limit = False

        for attempt in range(self.max_retries + 1):
            waited = self._wait_for_rate_window(
                item_count,
                estimated_tokens=estimated_tokens,
                after_rate_limit=after_rate_limit,
                max_wait_seconds=remaining_wait,
            )
            if remaining_wait is not None:
                remaining_wait = max(0.0, remaining_wait - waited)

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
                if self._is_rate_limit_error(err_msg):
                    if attempt < self.max_retries:
                        logger.warning(
                            "Gemini Embedding chạm giới hạn 429 (lần %d/%d). Đặt chỗ sang cửa sổ phút mới rồi thử lại...",
                            attempt + 1,
                            self.max_retries,
                        )
                        after_rate_limit = True
                        continue
                    logger.error("Gemini Embedding vượt giới hạn sau khi thử lại: %s", err_msg)
                    raise EmbeddingRateLimitError(f"{error_prefix}: {e}") from e

                logger.error("Lỗi khi gọi Gemini Embedding API: %s", err_msg)
                raise EmbeddingError(f"{error_prefix}: {e}") from e

        raise EmbeddingError(f"{error_prefix}: {last_error}")

    def embed_texts(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
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

            batch_embeddings: list[list[float]] = self._call_embed(
                content=batch,
                task_type="retrieval_document",
                error_prefix="Lỗi tạo embedding",
                item_count=len(batch),
                estimated_tokens=batch_tokens,
            )
            for emb in batch_embeddings:
                self._validate_vector(emb)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Tạo embedding cho một câu truy vấn tìm kiếm."""
        vector: list[float] = self._call_embed(
            content=query,
            task_type="retrieval_query",
            error_prefix="Lỗi tạo embedding query",
            item_count=1,
            estimated_tokens=max(1, len(query)),
            max_wait_seconds=self.max_query_wait_seconds,
        )
        self._validate_vector(vector)
        return vector


# Instance mặc định dùng cho toàn ứng dụng
embedding_service = EmbeddingService()
