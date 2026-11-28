from unittest.mock import patch

import pytest

from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.services.embedding import EmbeddingService


def test_embedding_dimension_alignment():
    """Kiểm tra số chiều vector trả về khớp chính xác với cấu hình Qdrant."""
    mock_vector = [0.01] * settings.QDRANT_VECTOR_SIZE  # 768 chiều
    mock_result = {"embedding": [mock_vector]}

    service = EmbeddingService()

    # Mock genai.embed_content để không tốn tiền API thật
    with patch("app.services.embedding.genai.embed_content", return_value=mock_result):
        vectors = service.embed_texts(["Sinh viên cần đăng ký bao nhiêu tín chỉ?"])

    assert len(vectors) == 1
    assert len(vectors[0]) == settings.QDRANT_VECTOR_SIZE


def test_embedding_dimension_mismatch_raises_error():
    """Nếu vector trả về sai số chiều Qdrant, phải ném EmbeddingError ngay."""
    service = EmbeddingService()
    bad_result = {"embedding": [[0.1] * 10]}

    with (
        patch("app.services.embedding.genai.embed_content", return_value=bad_result),
        pytest.raises(EmbeddingError, match="không khớp với Qdrant"),
    ):
        service.embed_texts(["Nội dung test"])


def test_embed_texts_waits_when_exceeding_rpm_window():
    """File vượt rpm_limit phải chủ động chờ hết cửa sổ 60s trước khi gửi tiếp."""
    mock_vector = [0.02] * settings.QDRANT_VECTOR_SIZE
    texts = [f"chunk {i}" for i in range(158)]
    service = EmbeddingService(rpm_limit=80)

    def _fake_embed(**kwargs):
        return {"embedding": [mock_vector] * len(kwargs["content"])}

    # Giả lập đồng hồ trôi 1s trước khi ngủ, rồi tăng lên +61s sau khi ngủ
    clock = iter([100.0, 100.5, 101.0, 162.0, 162.5])

    with (
        patch("app.services.embedding.genai.embed_content", side_effect=_fake_embed) as mock_embed,
        patch("app.services.embedding.time.monotonic", side_effect=clock),
        patch("app.services.embedding.time.sleep") as mock_sleep,
    ):
        vectors = service.embed_texts(texts, batch_size=40)

    assert len(vectors) == 158
    # 158 chunks với batch_size=40 -> 4 batches (40, 40, 40, 38)
    assert mock_embed.call_count == 4
    # Chỉ chờ đúng 1 lần sau 80 chunks đầu tiên (61.0 - 1.0 = 60.0s)
    mock_sleep.assert_called_once_with(60.0)


def test_embed_texts_no_wait_within_rpm_limit():
    """File nhỏ hơn hoặc bằng rpm_limit không phải chờ."""
    mock_vector = [0.03] * settings.QDRANT_VECTOR_SIZE
    texts = [f"chunk {i}" for i in range(80)]
    service = EmbeddingService(rpm_limit=80)

    def _fake_embed(**kwargs):
        return {"embedding": [mock_vector] * len(kwargs["content"])}

    with (
        patch("app.services.embedding.genai.embed_content", side_effect=_fake_embed) as mock_embed,
        patch("app.services.embedding.time.monotonic", side_effect=[10.0, 10.5]),
        patch("app.services.embedding.time.sleep") as mock_sleep,
    ):
        vectors = service.embed_texts(texts, batch_size=40)

    assert len(vectors) == 80
    assert mock_embed.call_count == 2
    mock_sleep.assert_not_called()


def test_embed_texts_no_wait_when_rpm_limit_disabled():
    """Khi rpm_limit=0 (gói trả phí), không giới hạn và không chờ."""
    mock_vector = [0.04] * settings.QDRANT_VECTOR_SIZE
    texts = [f"chunk {i}" for i in range(158)]
    service = EmbeddingService(rpm_limit=0)

    def _fake_embed(**kwargs):
        return {"embedding": [mock_vector] * len(kwargs["content"])}

    with (
        patch("app.services.embedding.genai.embed_content", side_effect=_fake_embed) as mock_embed,
        patch("app.services.embedding.time.sleep") as mock_sleep,
    ):
        vectors = service.embed_texts(texts, batch_size=64)

    assert len(vectors) == 158
    assert mock_embed.call_count == 3
    mock_sleep.assert_not_called()


def test_embed_texts_waits_when_exceeding_tpm_window():
    """File dài vượt tpm_limit phải tự chia batch vừa ngưỡng TPM và chủ động chờ sang phút mới."""
    mock_vector = [0.05] * settings.QDRANT_VECTOR_SIZE
    texts = ["A" * 800 for _ in range(50)]
    # tpm_limit=20000 -> mỗi batch tối đa 25 chunks (25*800 = 20000), batch thứ 2 chờ 60s
    service = EmbeddingService(rpm_limit=80, tpm_limit=20000)

    def _fake_embed(**kwargs):
        return {"embedding": [mock_vector] * len(kwargs["content"])}

    clock = iter([100.0, 101.0, 162.0])

    with (
        patch("app.services.embedding.genai.embed_content", side_effect=_fake_embed) as mock_embed,
        patch("app.services.embedding.time.monotonic", side_effect=clock),
        patch("app.services.embedding.time.sleep") as mock_sleep,
    ):
        vectors = service.embed_texts(texts, batch_size=40)

    assert len(vectors) == 50
    assert mock_embed.call_count == 2
    mock_sleep.assert_called_once_with(60.0)


def test_call_embed_retries_on_429_with_fixed_minute_wait():
    """Khi gặp lỗi 429 từ Gemini, chờ 61s sang phút mới rồi thử lại thành công."""
    mock_vector = [0.06] * settings.QDRANT_VECTOR_SIZE
    service = EmbeddingService(rpm_limit=80, max_retries=2)

    rate_limit_error = RuntimeError("429 Quota exceeded for minute")

    with (
        patch(
            "app.services.embedding.genai.embed_content",
            side_effect=[rate_limit_error, {"embedding": [mock_vector]}],
        ) as mock_embed,
        patch("app.services.embedding.time.sleep") as mock_sleep,
    ):
        vectors = service.embed_texts(["Nội dung test 429"])

    assert len(vectors) == 1
    assert mock_embed.call_count == 2
    mock_sleep.assert_called_once_with(61.0)
