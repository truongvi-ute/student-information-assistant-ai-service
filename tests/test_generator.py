import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import LLMTimeoutError
from app.main import app
from app.schemas.response import RetrievedChunk
from app.services.inference.generator import NO_CONTEXT_FALLBACK_ANSWER, RAGGenerator


def _make_chunks(count: int) -> list[RetrievedChunk]:
    """Tạo danh sách RetrievedChunk giả lập phục vụ kiểm thử."""
    chunks: list[RetrievedChunk] = []
    for idx in range(1, count + 1):
        content = f"Nội dung tài liệu số {idx}: Quy định đào tạo phần {idx}."
        chunks.append(
            RetrievedChunk(
                id=f"id-{idx}",
                content=content,
                score=round(0.95 - idx * 0.05, 2),
                chunk_index=idx - 1,
                document_id=f"doc-uuid-{idx}",
                filename=f"tai_lieu_{idx}.pdf",
                page_number=idx,
                char_start=idx * 100,
                char_end=idx * 100 + len(content),
            )
        )
    return chunks


def test_generator_with_context_builds_prompt_and_sources():
    """Kiểm tra generator ghép ngữ cảnh vào prompt và trích xuất đúng danh sách nguồn."""
    long_content = (
        "Sinh viên phải đăng ký tối thiểu 14 tín chỉ mỗi học kỳ. "
        + "Quy định này áp dụng cho tất cả sinh viên hệ chính quy đang theo học tại trường, "
        * 3
    ).strip()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(
            id="id-1",
            content=long_content,
            score=0.92,
            chunk_index=1,
            document_id="doc-uuid-1",
            filename="quy_che_2024.pdf",
            page_number=5,
            char_start=300,
            char_end=300 + len(long_content),
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
    source = response.sources[0]
    assert source.document_id == "doc-uuid-1"
    assert source.filename == "quy_che_2024.pdf"
    assert source.page_number == 5
    assert source.char_start == 300
    assert source.char_end == 300 + len(long_content)
    assert source.content == long_content
    assert source.snippet.endswith("...")
    assert len(source.snippet) == 203

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


def test_generator_passes_timeout_and_maps_read_timeout():
    """Kiểm tra generate_answer truyền timeout và ánh xạ 'timed out' sang LLMTimeoutError."""
    generator = RAGGenerator(retriever=MagicMock(), timeout_seconds=12.5)
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("HTTPSConnectionPool: Read timed out.")
    generator._model = mock_model

    with pytest.raises(LLMTimeoutError):
        generator.generate_answer("Prompt kiểm thử")

    call_kwargs = mock_model.generate_content.call_args.kwargs
    assert call_kwargs["request_options"] == {"timeout": 12.5}


def test_generator_filters_sources_by_used_indices():
    """Retriever trả về 5 chunks, LLM chỉ trích dẫn [2] -> response.sources chỉ giữ đúng chunk 2."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _make_chunks(5)

    generator = RAGGenerator(retriever=mock_retriever)
    mock_model = MagicMock()
    mock_llm_resp = MagicMock()
    mock_llm_resp.text = json.dumps(
        {
            "answer": "Theo tài liệu số 2 (trang 2), quy định đào tạo phần 2 được áp dụng.",
            "used_source_indices": [2],
            "has_relevant_context": True,
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_llm_resp
    generator._model = mock_model

    response = generator.answer_question("Quy định phần 2 là gì?", top_k=5)

    assert "tài liệu số 2" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].document_id == "doc-uuid-2"
    assert response.sources[0].filename == "tai_lieu_2.pdf"
    assert response.sources[0].page_number == 2


def test_generator_returns_empty_sources_when_no_relevant_context():
    """Khi ngữ cảnh nhiễu (has_relevant_context=false), ép danh sách sources về rỗng dù LLM có lỡ liệt kê index."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _make_chunks(5)

    generator = RAGGenerator(retriever=mock_retriever)
    mock_model = MagicMock()
    mock_llm_resp = MagicMock()
    mock_llm_resp.text = json.dumps(
        {
            "answer": (
                "Hiện tại tài liệu chưa có thông tin về học phí ký túc xá. "
                "Bạn có thể hỏi về chương trình đào tạo hoặc liên hệ Phòng Đào tạo nhé."
            ),
            "used_source_indices": [1],
            "has_relevant_context": False,
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_llm_resp
    generator._model = mock_model

    response = generator.answer_question("Học phí ký túc xá bao nhiêu?")

    assert "chưa có thông tin" in response.answer
    assert response.sources == []


def test_generator_parses_fenced_json_and_deduplicates_invalid_indices():
    """Hỗ trợ bóc tách JSON bọc trong ```json ... ``` và bỏ qua chỉ số trùng lặp hoặc ngoài phạm vi."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _make_chunks(3)

    generator = RAGGenerator(retriever=mock_retriever)
    mock_model = MagicMock()
    mock_llm_resp = MagicMock()
    mock_llm_resp.text = (
        "```json\n"
        '{"answer": "Thông tin tổng hợp từ nguồn 3 và nguồn 1.", '
        '"used_source_indices": [3, 3, 99, 0, -1, 1], "has_relevant_context": true}\n'
        "```"
    )
    mock_model.generate_content.return_value = mock_llm_resp
    generator._model = mock_model

    response = generator.answer_question("Tổng hợp thông tin?")

    assert response.answer == "Thông tin tổng hợp từ nguồn 3 và nguồn 1."
    assert [src.filename for src in response.sources] == ["tai_lieu_3.pdf", "tai_lieu_1.pdf"]


def test_generator_plain_text_fallback_remains_safe():
    """Đảm bảo khi LLM trả văn bản thuần không phải JSON thì vẫn hoạt động an toàn và không crash."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _make_chunks(3)

    generator = RAGGenerator(retriever=mock_retriever)
    mock_model = MagicMock()

    # Trường hợp 1: Văn bản thuần có nhắc rõ [Nguồn 2]
    resp_with_ref = MagicMock()
    resp_with_ref.text = "Theo [Nguồn 2], sinh viên thực hiện theo quy định phần 2."
    mock_model.generate_content.return_value = resp_with_ref
    generator._model = mock_model

    response_ref = generator.answer_question("Quy định phần 2?")
    assert response_ref.answer == resp_with_ref.text
    assert [src.filename for src in response_ref.sources] == ["tai_lieu_2.pdf"]

    # Trường hợp 2: Văn bản thuần là câu từ chối
    resp_refusal = MagicMock()
    resp_refusal.text = "Hiện tại hệ thống không tìm thấy thông tin về nội dung này."
    mock_model.generate_content.return_value = resp_refusal

    response_refusal = generator.answer_question("Hỏi ngoài phạm vi?")
    assert response_refusal.sources == []

    # Trường hợp 3: Văn bản thuần thông thường không nhắc [Nguồn i]
    resp_plain = MagicMock()
    resp_plain.text = "Sinh viên cần hoàn thành đầy đủ học phần theo kế hoạch."
    mock_model.generate_content.return_value = resp_plain

    response_plain = generator.answer_question("Kế hoạch học tập?")
    assert response_plain.answer == resp_plain.text
    assert len(response_plain.sources) == 3

