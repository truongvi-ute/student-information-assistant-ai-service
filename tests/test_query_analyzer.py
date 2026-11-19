import json
from unittest.mock import MagicMock

from app.schemas.request import ChatHistoryMessage
from app.services.inference.query_analyzer import (
    DEFAULT_GREETING_RESPONSE,
    DEFAULT_NON_ACADEMIC_RESPONSE,
    QueryAnalyzer,
)


def test_query_analyzer_rewrites_pronoun_and_ellipsis_from_history():
    """Khi câu mới dùng đại từ thay thế/ẩn chủ ngữ và history đã có ngành học -> viết lại standalone_query."""
    analyzer = QueryAnalyzer(api_key="test-key")
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(
        {
            "intent": "academic_rag",
            "standalone_query": "Ngành An toàn thông tin học mấy năm và chuẩn đầu ra ngoại ngữ thế nào?",
            "direct_response": None,
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_resp
    analyzer._model = mock_model

    history = [
        ChatHistoryMessage(role="user", content="Cho mình hỏi thông tin về ngành An toàn thông tin"),
        ChatHistoryMessage(role="assistant", content="Bạn cần xem thông tin gì về ngành An toàn thông tin?"),
    ]
    result = analyzer.analyze(
        question="Học mấy năm và chuẩn đầu ra ngoại ngữ của ngành đó thế nào?",
        history=history,
    )

    assert result.intent == "academic_rag"
    assert "An toàn thông tin" in result.standalone_query
    assert result.direct_response is None

    sent_prompt = mock_model.generate_content.call_args.args[0]
    assert "ngành An toàn thông tin" in sent_prompt
    assert mock_model.generate_content.call_args.kwargs["generation_config"]["temperature"] == 0.0


def test_query_analyzer_keeps_complete_query_or_topic_shift():
    """Khi câu hỏi đã có chủ thể mới rõ ràng thì giữ nguyên, không ghép ngành cũ từ lịch sử."""
    analyzer = QueryAnalyzer(api_key="test-key")
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(
        {
            "intent": "academic_rag",
            "standalone_query": "Ngành Kỹ thuật phần mềm có bao nhiêu tín chỉ?",
            "direct_response": None,
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_resp
    analyzer._model = mock_model

    history = [
        ChatHistoryMessage(role="user", content="Ngành An toàn thông tin học mấy năm?"),
        ChatHistoryMessage(role="assistant", content="Ngành An toàn thông tin học 4 năm."),
    ]
    result = analyzer.analyze(
        question="Ngành Kỹ thuật phần mềm có bao nhiêu tín chỉ?",
        history=history,
    )

    assert result.intent == "academic_rag"
    assert result.standalone_query == "Ngành Kỹ thuật phần mềm có bao nhiêu tín chỉ?"
    assert result.direct_response is None


def test_query_analyzer_asks_clarification_when_history_lacks_subject():
    """Khi câu hỏi phụ thuộc ngành học nhưng cả câu hỏi lẫn history đều thiếu tên ngành -> hỏi ngược lại qua non_academic."""
    analyzer = QueryAnalyzer(api_key="test-key")
    mock_model = MagicMock()
    clarification = (
        "Số tín chỉ học tập sẽ khác nhau tùy theo chương trình đào tạo. "
        "Bạn đang muốn tra cứu cho ngành học nào để mình kiểm tra giúp bạn nhé?"
    )
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(
        {
            "intent": "non_academic",
            "standalone_query": "Ngành này học bao nhiêu tín chỉ?",
            "direct_response": clarification,
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_resp
    analyzer._model = mock_model

    result = analyzer.analyze(question="Ngành này học bao nhiêu tín chỉ?", history=[])

    assert result.intent == "non_academic"
    assert result.direct_response == clarification


def test_query_analyzer_resolves_clarification_followup_turn():
    """Lượt tiếp theo sau câu hỏi làm rõ: ghép tên ngành sinh viên trả lời thành standalone_query đầy đủ."""
    analyzer = QueryAnalyzer(api_key="test-key")
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(
        {
            "intent": "academic_rag",
            "standalone_query": "Ngành Kỹ thuật dữ liệu học bao nhiêu tín chỉ?",
            "direct_response": None,
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_resp
    analyzer._model = mock_model

    history = [
        ChatHistoryMessage(role="user", content="Ngành này học bao nhiêu tín chỉ?"),
        ChatHistoryMessage(role="assistant", content="Bạn đang muốn tra cứu cho ngành học nào?"),
    ]
    result = analyzer.analyze(question="Ngành Kỹ thuật dữ liệu", history=history)

    assert result.intent == "academic_rag"
    assert result.standalone_query == "Ngành Kỹ thuật dữ liệu học bao nhiêu tín chỉ?"
    sent_prompt = mock_model.generate_content.call_args.args[0]
    assert "Sinh viên: Ngành này học bao nhiêu tín chỉ?" in sent_prompt
    assert "Trợ lý: Bạn đang muốn tra cứu cho ngành học nào?" in sent_prompt


def test_query_analyzer_fast_path_and_non_academic_guidance():
    """Fast-path bắt chào hỏi thuần túy không gọi LLM; câu chào kèm câu hỏi học vụ vẫn đi qua LLM."""
    analyzer = QueryAnalyzer(api_key="test-key")
    mock_model = MagicMock()
    analyzer._model = mock_model

    greeting_res = analyzer.analyze(question="Xin chào!", history=[])
    assert greeting_res.intent == "non_academic"
    assert greeting_res.direct_response == DEFAULT_GREETING_RESPONSE
    mock_model.generate_content.assert_not_called()

    # Câu vừa chào vừa hỏi thông tin học vụ phải đi qua LLM, không bị fast-path chặn nhầm
    mock_academic_resp = MagicMock()
    mock_academic_resp.text = json.dumps(
        {
            "intent": "academic_rag",
            "standalone_query": "Ngành Công nghệ thông tin học mấy năm?",
            "direct_response": None,
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_academic_resp
    mixed_res = analyzer.analyze(question="Xin chào, ngành CNTT học mấy năm?", history=[])
    assert mixed_res.intent == "academic_rag"
    assert mock_model.generate_content.call_count == 1

    # Câu hỏi ngoài phạm vi trả về non_academic kèm câu dẫn dắt
    mock_oos_resp = MagicMock()
    mock_oos_resp.text = json.dumps(
        {
            "intent": "non_academic",
            "standalone_query": "Hôm nay thời tiết thế nào?",
            "direct_response": "",
        },
        ensure_ascii=False,
    )
    mock_model.generate_content.return_value = mock_oos_resp
    oos_res = analyzer.analyze(question="Hôm nay thời tiết thế nào?", history=[])
    assert oos_res.intent == "non_academic"
    assert oos_res.direct_response == DEFAULT_NON_ACADEMIC_RESPONSE


def test_query_analyzer_falls_back_safely_on_llm_error_or_invalid_json():
    """Khi LLM gặp lỗi hoặc trả JSON hỏng, fallback an toàn về academic_rag với câu hỏi gốc."""
    analyzer = QueryAnalyzer(api_key="test-key")
    mock_model = MagicMock()
    analyzer._model = mock_model

    mock_model.generate_content.side_effect = RuntimeError("Gemini connection error")
    err_res = analyzer.analyze(question="Điều kiện tốt nghiệp ngành ATTT?")
    assert err_res.intent == "academic_rag"
    assert err_res.standalone_query == "Điều kiện tốt nghiệp ngành ATTT?"
    assert err_res.direct_response is None

    mock_model.generate_content.side_effect = None
    bad_json_resp = MagicMock()
    bad_json_resp.text = "Không phải định dạng JSON"
    mock_model.generate_content.return_value = bad_json_resp
    invalid_res = analyzer.analyze(question="Quy chế thi kết thúc học phần?")
    assert invalid_res.intent == "academic_rag"
    assert invalid_res.standalone_query == "Quy chế thi kết thúc học phần?"
