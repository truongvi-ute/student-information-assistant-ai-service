from unittest.mock import MagicMock
from app.services.inference.retriever import VectorRetriever


def test_retriever_empty_collection_returns_empty_list():
    """Khi collection chưa tồn tại, retriever trả về danh sách rỗng thay vì lỗi."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_embedding = MagicMock()

    service = VectorRetriever(qdrant_client=mock_client, embed_service=mock_embedding)
    results = service.retrieve("Học phí kỳ này bao nhiêu?")

    assert results == []
    mock_embedding.embed_query.assert_not_called()


def test_retriever_returns_matching_chunks_and_applies_filter():
    """Kiểm tra retriever chuyển đổi ScoredPoint đúng chuẩn và áp dụng bộ lọc filename."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    mock_point = MagicMock()
    mock_point.id = "point-uuid-1"
    mock_point.score = 0.89
    mock_point.payload = {
        "content": "Sinh viên phải đăng ký tối thiểu 14 tín chỉ.",
        "chunk_index": 0,
        "filename": "quy_che.pdf",
        "page_number": 2,
    }
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_client.query_points.return_value = mock_response

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.01] * 768

    service = VectorRetriever(qdrant_client=mock_client, embed_service=mock_embedding)
    results = service.retrieve("Đăng ký tối thiểu bao nhiêu tín chỉ?", top_k=3, filename_filter="quy_che.pdf")

    assert len(results) == 1
    assert results[0].filename == "quy_che.pdf"
    assert results[0].page_number == 2
    assert results[0].score == 0.89

    # Kiểm tra query_filter đã được truyền vào query_points
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["limit"] == 3
    assert call_kwargs["query_filter"] is not None

