import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.core.exceptions import EmbeddingError, EmbeddingRateLimitError
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

    with patch("app.services.embedding.genai.embed_content", return_value=bad_result):
        with pytest.raises(EmbeddingError, match="không khớp với Qdrant"):
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


def test_concurrent_rate_window_reserves_slots_without_holding_lock_during_sleep():
    """Nhiều luồng vượt ngưỡng đồng thời phải đặt chỗ tuần tự và không giữ khóa khi sleep."""
    mock_vector = [0.07] * settings.QDRANT_VECTOR_SIZE
    service = EmbeddingService(rpm_limit=50, tpm_limit=0)

    def _fake_embed(**kwargs):
        content = kwargs["content"]
        if isinstance(content, list):
            return {"embedding": [mock_vector] * len(content)}
        return {"embedding": mock_vector}

    both_sleeping = threading.Barrier(2, timeout=2.0)
    sleep_durations: list[float] = []
    lock_states_during_sleep: list[bool] = []
    record_lock = threading.Lock()

    def _fake_sleep(seconds: float) -> None:
        with record_lock:
            lock_states_during_sleep.append(service._lock.locked())
            sleep_durations.append(seconds)
        both_sleeping.wait()

    with (
        patch("app.services.embedding.genai.configure"),
        patch("app.services.embedding.genai.embed_content", side_effect=_fake_embed),
        patch("app.services.embedding.time.monotonic", return_value=100.0),
        patch("app.services.embedding.time.sleep", side_effect=_fake_sleep),
    ):
        initial = service.embed_texts([f"init {i}" for i in range(50)], batch_size=50)
        assert len(initial) == 50

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(service.embed_texts, [f"a {i}" for i in range(30)], 30)
            f2 = pool.submit(service.embed_texts, [f"b {i}" for i in range(30)], 30)
            res1 = f1.result(timeout=3.0)
            res2 = f2.result(timeout=3.0)

    assert len(res1) == 30
    assert len(res2) == 30
    assert lock_states_during_sleep == [False, False]
    assert sorted(sleep_durations) == [61.0, 122.0]


def test_concurrent_embed_query_shares_rate_window_and_configures_once():
    """Nhiều luồng gọi embed_query đồng thời chỉ configure một lần và đếm chung cửa sổ RPM."""
    mock_vector = [0.08] * settings.QDRANT_VECTOR_SIZE
    service = EmbeddingService(rpm_limit=5, tpm_limit=0)
    start_barrier = threading.Barrier(8, timeout=2.0)

    def _worker(idx: int) -> list[float]:
        start_barrier.wait()
        return service.embed_query(f"Câu hỏi số {idx}?")

    with (
        patch("app.services.embedding.genai.configure") as mock_configure,
        patch(
            "app.services.embedding.genai.embed_content",
            return_value={"embedding": mock_vector},
        ),
        patch("app.services.embedding.time.monotonic", return_value=200.0),
        patch("app.services.embedding.time.sleep") as mock_sleep,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        futures = [pool.submit(_worker, i) for i in range(8)]
        results = [f.result(timeout=3.0) for f in futures]

    assert len(results) == 8
    assert mock_configure.call_count == 1
    assert mock_sleep.call_count == 3
    assert [call.args[0] for call in mock_sleep.call_args_list] == [61.0, 61.0, 61.0]
    assert service._window_count == 3


def test_concurrent_429_retries_reserve_windows_without_stampede():
    """Nhiều luồng cùng gặp 429 khi retry phải đặt chỗ vào cửa sổ mới và không làm mất lượt đã ghi nhận."""
    mock_vector = [0.09] * settings.QDRANT_VECTOR_SIZE
    service = EmbeddingService(rpm_limit=50, tpm_limit=0, max_retries=2)
    first_wave_barrier = threading.Barrier(2, timeout=2.0)
    call_lock = threading.Lock()
    seen_first_failure: set[int] = set()

    def _fake_embed(**kwargs):
        content = kwargs["content"]
        if content[0].startswith("retry-"):
            tid = threading.get_ident()
            with call_lock:
                is_first_attempt = tid not in seen_first_failure
                if is_first_attempt:
                    seen_first_failure.add(tid)
            if is_first_attempt:
                first_wave_barrier.wait()
                raise RuntimeError("429 Resource exhausted")
        return {"embedding": [mock_vector] * len(content)}

    with (
        patch("app.services.embedding.genai.configure"),
        patch("app.services.embedding.genai.embed_content", side_effect=_fake_embed),
        patch("app.services.embedding.time.monotonic", return_value=100.0),
        patch("app.services.embedding.time.sleep") as mock_sleep,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        f1 = pool.submit(service.embed_texts, [f"retry-a-{i}" for i in range(25)], 25)
        f2 = pool.submit(service.embed_texts, [f"retry-b-{i}" for i in range(25)], 25)
        r1 = f1.result(timeout=3.0)
        r2 = f2.result(timeout=3.0)
        r3 = service.embed_texts([f"next-{i}" for i in range(20)], batch_size=20)

    assert len(r1) == 25
    assert len(r2) == 25
    assert len(r3) == 20
    assert sorted(call.args[0] for call in mock_sleep.call_args_list) == [61.0, 61.0, 122.0]
    assert service._window_start == 222.0
    assert service._window_count == 20


def test_embed_query_fails_fast_when_wait_exceeds_max_query_wait_without_consuming_slot():
    """Khi thời gian chờ vượt ngưỡng tối đa của chat, từ chối ngay mà không ngủ hay giữ suất."""
    mock_vector = [0.1] * settings.QDRANT_VECTOR_SIZE
    service = EmbeddingService(
        rpm_limit=50,
        tpm_limit=0,
        max_query_wait_seconds=65.0,
    )

    def _fake_embed(**kwargs):
        content = kwargs["content"]
        if isinstance(content, list):
            return {"embedding": [mock_vector] * len(content)}
        return {"embedding": mock_vector}

    with (
        patch("app.services.embedding.genai.configure"),
        patch("app.services.embedding.genai.embed_content", side_effect=_fake_embed),
        patch("app.services.embedding.time.monotonic", return_value=100.0),
        patch("app.services.embedding.time.sleep") as mock_sleep,
    ):
        service.embed_texts([f"c {i}" for i in range(100)], batch_size=50)
        assert mock_sleep.call_count == 1
        assert service._window_start == 161.0
        assert service._window_count == 50

        mock_sleep.reset_mock()
        with pytest.raises(EmbeddingRateLimitError, match="Hệ thống đang bận"):
            service.embed_query("Câu hỏi đến khi hàng đợi đã lùi quá 65 giây")

        mock_sleep.assert_not_called()
        assert service._window_start == 161.0
        assert service._window_count == 50
