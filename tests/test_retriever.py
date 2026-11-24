from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import app.db.qdrant as qdrant_module
from app.core.config import settings
from app.services.inference.retriever import VectorRetriever


def _make_scored_point(
    point_id: str,
    score: float,
    filename: str,
    content: str,
) -> MagicMock:
    point = MagicMock()
    point.id = point_id
    point.score = score
    point.payload = {
        "content": content,
        "chunk_index": 0,
        "document_id": f"doc-{filename}",
        "filename": filename,
        "page_number": 1,
        "char_start": 0,
        "char_end": len(content),
    }
    return point


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
        "document_id": "11111111-1111-1111-1111-111111111111",
        "filename": "quy_che.pdf",
        "page_number": 2,
        "char_start": 120,
        "char_end": 165,
    }
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_client.query_points.return_value = mock_response

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.01] * 768

    service = VectorRetriever(qdrant_client=mock_client, embed_service=mock_embedding)
    results = service.retrieve("Đăng ký tối thiểu bao nhiêu tín chỉ?", top_k=3, filename_filter="quy_che.pdf")

    assert len(results) == 1
    assert results[0].document_id == "11111111-1111-1111-1111-111111111111"
    assert results[0].filename == "quy_che.pdf"
    assert results[0].page_number == 2
    assert results[0].char_start == 120
    assert results[0].char_end == 165
    assert results[0].score == 0.89

    # Kiểm tra query_filter đã được truyền vào query_points
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["limit"] == 3
    assert call_kwargs["query_filter"] is not None


def test_retriever_applies_default_score_threshold_and_override():
    """Kiểm tra retrieve() dùng settings.RETRIEVAL_SCORE_THRESHOLD mặc định và cho phép ghi đè."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.02] * 768

    service = VectorRetriever(qdrant_client=mock_client, embed_service=mock_embedding)

    service.retrieve("Học phí kỳ này bao nhiêu?")
    assert (
        mock_client.query_points.call_args.kwargs["score_threshold"]
        == settings.RETRIEVAL_SCORE_THRESHOLD
    )

    service.retrieve("Học phí kỳ này bao nhiêu?", score_threshold=0.72)
    assert mock_client.query_points.call_args.kwargs["score_threshold"] == 0.72


def test_retrieve_multi_queries_round_robin_and_deduplicates():
    """Kiểm tra retrieve_multi_queries gộp xen kẽ đa tài liệu, khử trùng lặp Point ID và giữ score cao nhất."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    resp_attt = MagicMock()
    resp_attt.points = [
        _make_scored_point("p-attt-1", 0.85, "attt.docx", "Môn chuyên ngành ATTT 1"),
        _make_scored_point("p-shared", 0.70, "chung.docx", "Quy chế đào tạo chung"),
        _make_scored_point("p-attt-2", 0.65, "attt.docx", "Môn chuyên ngành ATTT 2"),
    ]
    resp_ktdl = MagicMock()
    resp_ktdl.points = [
        _make_scored_point("p-ktdl-1", 0.80, "ktdl.docx", "Môn chuyên ngành KTDL 1"),
        _make_scored_point("p-shared", 0.90, "chung.docx", "Quy chế đào tạo chung"),
        _make_scored_point("p-ktdl-2", 0.60, "ktdl.docx", "Môn chuyên ngành KTDL 2"),
    ]
    mock_client.query_points.side_effect = [resp_attt, resp_ktdl]

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.03] * 768

    service = VectorRetriever(qdrant_client=mock_client, embed_service=mock_embedding)
    results = service.retrieve_multi_queries(
        queries=["Môn chuyên ngành ATTT", "Môn chuyên ngành KTDL"],
        top_k_per_query=3,
        max_total_chunks=5,
    )

    assert [c.id for c in results] == [
        "p-attt-1",
        "p-ktdl-1",
        "p-shared",
        "p-attt-2",
        "p-ktdl-2",
    ]
    shared_chunk = next(c for c in results if c.id == "p-shared")
    assert shared_chunk.score == 0.90
    filenames = {c.filename for c in results}
    assert "attt.docx" in filenames
    assert "ktdl.docx" in filenames


def test_retrieve_multi_queries_handles_empty_and_single_query():
    """Kiểm tra retrieve_multi_queries trả rỗng khi queries trống và gọi retrieve 1 lần khi chỉ có 1 query."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_response = MagicMock()
    mock_response.points = [
        _make_scored_point("p-1", 0.88, "attt.docx", "Thông tin ngành ATTT"),
    ]
    mock_client.query_points.return_value = mock_response

    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.04] * 768

    service = VectorRetriever(qdrant_client=mock_client, embed_service=mock_embedding)

    assert service.retrieve_multi_queries(["   ", ""]) == []
    assert mock_client.query_points.call_count == 0

    single_results = service.retrieve_multi_queries(["Ngành ATTT", "   "], top_k_per_query=2)
    assert len(single_results) == 1
    assert single_results[0].id == "p-1"
    assert mock_client.query_points.call_count == 1
    assert mock_client.query_points.call_args.kwargs["limit"] == 2


def test_get_qdrant_client_normalizes_blank_api_key_and_initializes_once():
    """Khởi tạo QdrantClient đồng thời chỉ tạo 1 singleton và chuẩn hóa api_key rỗng thành None."""
    mock_instance = MagicMock()
    previous_client = qdrant_module._client
    qdrant_module._client = None
    try:
        with (
            patch.object(settings, "QDRANT_API_KEY", "   "),
            patch("app.db.qdrant.QdrantClient", return_value=mock_instance) as mock_cls,
            ThreadPoolExecutor(max_workers=6) as pool,
        ):
            clients = list(pool.map(lambda _: qdrant_module.get_qdrant_client(), range(6)))

        assert all(c is mock_instance for c in clients)
        assert mock_cls.call_count == 1
        assert mock_cls.call_args.kwargs["api_key"] is None
    finally:
        qdrant_module._client = previous_client

