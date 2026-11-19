import json
import re
from typing import Any, Literal

import google.generativeai as genai
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.schemas.request import ChatHistoryMessage
from app.services.inference.prompts import build_query_analysis_prompt

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

DEFAULT_GREETING_RESPONSE = (
    "Chào bạn! Mình là Trợ lý Thông tin Sinh viên (SIA). "
    "Bạn có thể hỏi mình về **chương trình đào tạo, danh mục môn học, số tín chỉ, "
    "chuẩn đầu ra, quy chế học vụ hoặc điều kiện tốt nghiệp** nhé!"
)

DEFAULT_THANKS_RESPONSE = (
    "Không có gì đâu! Nếu bạn cần tra cứu thêm thông tin về **chương trình đào tạo, "
    "môn học, số tín chỉ, chuẩn đầu ra hoặc quy chế học vụ**, cứ nhắn mình nhé!"
)

DEFAULT_NON_ACADEMIC_RESPONSE = (
    "Mình là Trợ lý Thông tin Sinh viên (SIA), chuyên hỗ trợ tra cứu thông tin học vụ. "
    "Bạn có thể hỏi mình về **chương trình đào tạo, danh mục môn học, số tín chỉ, "
    "chuẩn đầu ra, quy chế học vụ hoặc điều kiện tốt nghiệp** nhé!"
)

_EXACT_GREETINGS = frozenset(
    {
        "xin chào",
        "chào",
        "chào bạn",
        "chào ad",
        "chào bot",
        "chào sia",
        "hello",
        "hi",
        "hi bạn",
        "hey",
        "alo",
    }
)

_EXACT_THANKS = frozenset(
    {
        "cảm ơn",
        "cảm ơn bạn",
        "cảm ơn nhiều",
        "cám ơn",
        "cám ơn bạn",
        "thanks",
        "thank you",
        "tạm biệt",
        "bye",
    }
)


def extract_json_dict(raw_text: str) -> dict[str, Any] | None:
    """Thử phân tích chuỗi JSON trực tiếp, trong khối ```json``` hoặc đoạn {...}."""
    candidates = [raw_text]
    fence_match = _JSON_FENCE_RE.search(raw_text)
    if fence_match:
        candidates.append(fence_match.group(1))

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(raw_text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class QueryAnalysisResult(BaseModel):
    """Kết quả phân tích ý định, làm rõ ngữ cảnh và viết lại câu truy vấn."""

    intent: Literal["academic_rag", "non_academic"] = Field(
        default="academic_rag",
        description="Nhánh xử lý: 'academic_rag' (tra cứu tài liệu) hoặc 'non_academic' (trả lời/hỏi lại trực tiếp)",
    )
    standalone_query: str = Field(
        ...,
        description="Câu hỏi độc lập đầy đủ ngữ cảnh để tìm kiếm trên Qdrant",
    )
    direct_response: str | None = Field(
        default=None,
        description="Câu trả lời hoặc câu hỏi ngược lại làm rõ ngữ cảnh khi intent='non_academic'",
    )


class QueryAnalyzer:
    """Bộ phân tích câu hỏi sử dụng Gemini cho Intent Routing, Clarification và Query Rewriting."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.CHAT_MODEL
        self.timeout_seconds = (
            settings.GEMINI_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        self._model: genai.GenerativeModel | None = None

    @property
    def model(self) -> genai.GenerativeModel:
        """Khởi tạo Gemini GenerativeModel theo cơ chế lazy loading với HTTP REST transport."""
        if self._model is None:
            genai.configure(api_key=self.api_key, transport="rest")
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    @staticmethod
    def _match_fast_path(question: str) -> str | None:
        """Nhận diện khớp nguyên câu cho các lời chào/cảm ơn ngắn gọn khi chưa có lịch sử."""
        normalized = " ".join(question.strip().lower().rstrip("!.?,~").split())
        if normalized in _EXACT_GREETINGS:
            return DEFAULT_GREETING_RESPONSE
        if normalized in _EXACT_THANKS:
            return DEFAULT_THANKS_RESPONSE
        return None

    @staticmethod
    def _parse_analysis_output(raw_text: str, original_question: str) -> QueryAnalysisResult:
        """Bóc tách JSON đầu ra từ LLM thành QueryAnalysisResult an toàn."""
        data = extract_json_dict((raw_text or "").strip())
        if data is None:
            return QueryAnalysisResult(
                intent="academic_rag",
                standalone_query=original_question,
            )

        raw_intent = str(data.get("intent", "academic_rag")).strip().lower()
        intent: Literal["academic_rag", "non_academic"] = (
            "non_academic" if raw_intent == "non_academic" else "academic_rag"
        )

        raw_standalone = data.get("standalone_query")
        standalone_query = (
            raw_standalone.strip()
            if isinstance(raw_standalone, str) and raw_standalone.strip()
            else original_question
        )

        if intent == "non_academic":
            raw_direct = data.get("direct_response")
            direct_response = (
                raw_direct.strip()
                if isinstance(raw_direct, str) and raw_direct.strip()
                else DEFAULT_NON_ACADEMIC_RESPONSE
            )
            return QueryAnalysisResult(
                intent="non_academic",
                standalone_query=original_question,
                direct_response=direct_response,
            )

        return QueryAnalysisResult(
            intent="academic_rag",
            standalone_query=standalone_query,
            direct_response=None,
        )

    def analyze(
        self,
        question: str,
        history: list[ChatHistoryMessage] | None = None,
        filename_filter: str | None = None,
    ) -> QueryAnalysisResult:
        """Phân tích ý định câu hỏi, hỏi làm rõ nếu thiếu ngữ cảnh hoặc viết lại câu truy vấn độc lập."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return QueryAnalysisResult(
                intent="non_academic",
                standalone_query=question,
                direct_response=DEFAULT_NON_ACADEMIC_RESPONSE,
            )

        if not history:
            fast_reply = self._match_fast_path(cleaned_question)
            if fast_reply is not None:
                logger.info(
                    "QueryAnalyzer | source=fast_path | intent=non_academic | rewritten=False | question=%r",
                    cleaned_question,
                )
                return QueryAnalysisResult(
                    intent="non_academic",
                    standalone_query=cleaned_question,
                    direct_response=fast_reply,
                )

        prompt = build_query_analysis_prompt(
            question=cleaned_question,
            history=history,
            filename_filter=filename_filter,
        )
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
                request_options={"timeout": self.timeout_seconds},
            )
            raw_text = (response.text or "").strip()
            result = self._parse_analysis_output(raw_text, original_question=cleaned_question)
            rewritten = (
                result.intent == "academic_rag"
                and result.standalone_query.strip() != cleaned_question
            )
            if rewritten:
                logger.info(
                    "QueryAnalyzer | source=llm | intent=%s | rewritten=True | original=%r -> standalone=%r",
                    result.intent,
                    cleaned_question,
                    result.standalone_query,
                )
            else:
                logger.info(
                    "QueryAnalyzer | source=llm | intent=%s | rewritten=False | question=%r",
                    result.intent,
                    cleaned_question,
                )
            return result
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "QueryAnalyzer gặp lỗi khi gọi LLM, fallback về academic_rag: %s",
                str(e),
            )
            return QueryAnalysisResult(
                intent="academic_rag",
                standalone_query=cleaned_question,
            )


query_analyzer = QueryAnalyzer()
