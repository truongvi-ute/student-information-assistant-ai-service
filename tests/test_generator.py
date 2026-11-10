from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.response import RetrievedChunk
from app.services.inference.generator import NO_CONTEXT_FALLBACK_ANSWER, RAGGenerator


def test_generator_with_context_builds_prompt_and_sources():
    """Kiểm tra generator ghép ngữ cảnh vào prompt và trích xuất đúng danh sách nguồn."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(
            id="id-1",
            content="Sinh viên phải đăng ký tối thiểu 14 tín chỉ mỗi học kỳ.",
            score=0.92,
            chunk_index=1,
            filename="quy_che_2024.pdf",
            page_number=5,
        )
    ]

    generator = RAGGenerator(retriever=mock_retriever)
    mock_model = MagicMock()
    mock_llm_resp = MagicMock()
    mock_llm_resp.text = "Theo quy chế trang 5, sinh viên cần đăng ký ít nhất 14 tín chỉ."
    mock_model.generate_content.return_value = mock_llm_resp
    generator._model = mock_model

    response = generator.answer_question("Tối thiểu bao nhiêu tín chỉ?")

    assert "14 tín chỉ" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].filename == "quy_che_2024.pdf"
    assert response.sources[0].page_number == 5

    # Đảm bảo prompt gửi vào LLM chứa cả nội dung chunk và câu hỏi
    sent_prompt = mock_model.generate_content.call_args.args[0]
    assert "Sinh viên phải đăng ký tối thiểu 14 tín chỉ" in sent_prompt
    assert "Tối thiểu bao nhiêu tín chỉ?" in sent_prompt


def test_generator_without_context_returns_safe_fallback():
    """Khi không tìm thấy chunk nào, trả lời an toàn và không gọi LLM."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    generator = RAGGenerator(retriever=mock_retriever)
    mock_model = MagicMock()
    generator._model = mock_model

    response = generator.answer_question("Câu hỏi ngoài lề không có tài liệu?")

    assert response.answer == NO_CONTEXT_FALLBACK_ANSWER
    assert response.sources == []
    mock_model.generate_content.assert_not_called()


def test_api_validation_rejects_invalid_top_k_or_blank_query():
    """Kiểm tra FastAPI trả về 422 khi top_k vượt giới hạn hoặc câu hỏi rỗng."""
    client = TestClient(app)

    res_blank = client.post("/v1/search", json={"query": "   ", "top_k": 5})
    assert res_blank.status_code == 422

    res_top_k = client.post("/v1/chat", json={"question": "Hỏi hợp lệ", "top_k": 99})
    assert res_top_k.status_code == 422

