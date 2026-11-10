from unittest.mock import MagicMock, patch
from app.core.config import settings
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
