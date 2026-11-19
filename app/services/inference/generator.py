import re
import time

import google.generativeai as genai

from app.core.config import settings
from app.core.exceptions import LLMGenerationError, LLMTimeoutError
from app.core.logging import logger
from app.schemas.request import ChatHistoryMessage
from app.schemas.response import (
    ChatResponse,
    RetrievedChunk,
    SourceCitation,
    StructuredRAGOutput,
)
from app.services.inference.prompts import build_rag_prompt
from app.services.inference.query_analyzer import (
    QueryAnalyzer,
    extract_json_dict,
)
from app.services.inference.query_analyzer import (
    query_analyzer as default_query_analyzer,
)
from app.services.inference.retriever import VectorRetriever, vector_retriever

NO_CONTEXT_FALLBACK_ANSWER = (
    "Hiện tại mình chưa tìm thấy thông tin phù hợp với câu hỏi của bạn trong kho tài liệu học vụ. "
    "Bạn có thể thử diễn đạt rõ hơn (kèm tên ngành học hoặc tên quy chế), hoặc hỏi mình về các nội dung "
    "như: **chương trình đào tạo, danh mục môn học, số tín chỉ, chuẩn đầu ra và điều kiện tốt nghiệp** nhé!"
)

_SOURCE_TAG_PATTERN = re.compile(r"\[Nguồn\s*(\d+)(?:\s*:[^\]]*)?\]", re.IGNORECASE)
_REFUSAL_MARKERS = (
    "không tìm thấy thông tin",
    "chưa tìm thấy thông tin",
    "không có thông tin",
    "chưa đề cập",
    "không đề cập",
    "không thể trả lời",
)


class RAGGenerator:
    """Bộ điều phối RAG: Truy xuất ngữ cảnh, gọi LLM sinh câu trả lời và chọn lọc trích dẫn nguồn."""

    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.vector_retriever = retriever or vector_retriever
        self.query_analyzer = query_analyzer or default_query_analyzer
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

    @classmethod
    def _parse_llm_output(
        cls,
        raw_text: str,
        total_chunks: int,
    ) -> StructuredRAGOutput:
        """Bóc tách JSON đầu ra từ LLM và chuẩn hóa chỉ số nguồn; fallback an toàn nếu nhận văn bản thường."""
        cleaned = (raw_text or "").strip()
        data = extract_json_dict(cleaned)

        if data is not None and isinstance(data.get("answer"), str) and data["answer"].strip():
            answer = data["answer"].strip()
            raw_has_context = data.get("has_relevant_context", True)
            if isinstance(raw_has_context, bool):
                has_relevant_context = raw_has_context
            elif isinstance(raw_has_context, str):
                has_relevant_context = raw_has_context.strip().lower() not in ("false", "0", "no")
            else:
                has_relevant_context = bool(raw_has_context)

            if not has_relevant_context:
                return StructuredRAGOutput(
                    answer=answer,
                    used_source_indices=[],
                    has_relevant_context=False,
                )

            raw_indices = data.get("used_source_indices", [])
            valid_indices: list[int] = []
            seen_indices: set[int] = set()
            if isinstance(raw_indices, list):
                for item in raw_indices:
                    if isinstance(item, bool):
                        continue
                    if isinstance(item, int):
                        idx = item
                    elif isinstance(item, str) and item.strip().isdigit():
                        idx = int(item.strip())
                    else:
                        continue
                    if 1 <= idx <= total_chunks and idx not in seen_indices:
                        seen_indices.add(idx)
                        valid_indices.append(idx)

            return StructuredRAGOutput(
                answer=answer,
                used_source_indices=valid_indices,
                has_relevant_context=True,
            )

        # Fallback an toàn khi LLM trả về văn bản thuần (không phải JSON)
        mentioned_source_indices: list[int] = []
        seen_indices = set()
        for match in _SOURCE_TAG_PATTERN.finditer(cleaned):
            idx = int(match.group(1))
            if 1 <= idx <= total_chunks and idx not in seen_indices:
                seen_indices.add(idx)
                mentioned_source_indices.append(idx)

        if mentioned_source_indices:
            return StructuredRAGOutput(
                answer=cleaned,
                used_source_indices=mentioned_source_indices,
                has_relevant_context=True,
            )

        lowered = cleaned.lower()
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            return StructuredRAGOutput(
                answer=cleaned,
                used_source_indices=[],
                has_relevant_context=False,
            )

        fallback_indices = list(range(1, total_chunks + 1)) if total_chunks > 0 else []
        return StructuredRAGOutput(
            answer=cleaned,
            used_source_indices=fallback_indices,
            has_relevant_context=True,
        )

    @staticmethod
    def _build_sources(
        chunks: list[RetrievedChunk],
        used_indices: list[int] | None = None,
    ) -> list[SourceCitation]:
        """Chuyển danh sách chunk đã dùng thành danh sách trích dẫn nguồn cho response."""
        if used_indices is None:
            selected_chunks = chunks
        else:
            selected_chunks = []
            seen_indices: set[int] = set()
            for idx in used_indices:
                if 1 <= idx <= len(chunks) and idx not in seen_indices:
                    seen_indices.add(idx)
                    selected_chunks.append(chunks[idx - 1])

        sources: list[SourceCitation] = []
        for chunk in selected_chunks:
            snippet = chunk.content[:200] + ("..." if len(chunk.content) > 200 else "")
            sources.append(
                SourceCitation(
                    document_id=chunk.document_id,
                    filename=chunk.filename or "unknown",
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    score=round(chunk.score, 4),
                    snippet=snippet,
                    content=chunk.content,
                )
            )
        return sources

    def generate_answer(self, prompt: str) -> str:
        """Gửi prompt tới Gemini LLM và nhận câu trả lời văn bản."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
                request_options={"timeout": self.timeout_seconds},
            )
            answer_text = (response.text or "").strip()
            if not answer_text:
                raise LLMGenerationError("Mô hình LLM trả về câu trả lời rỗng.")
            return answer_text
        except LLMGenerationError:
            raise
        except TimeoutError as e:
            logger.error("Quá thời gian chờ khi gọi Gemini LLM: %s", str(e))
            raise LLMTimeoutError(f"LLM phản hồi quá thời gian quy định: {e}") from e
        except Exception as e:
            err_msg = str(e).lower()
            if (
                "timeout" in err_msg
                or "timed out" in err_msg
                or "deadline" in err_msg
            ):
                logger.error("Deadline exceeded khi gọi Gemini LLM: %s", str(e))
                raise LLMTimeoutError(f"LLM quá thời gian phản hồi: {e}") from e

            logger.error("Lỗi khi gọi Gemini LLM: %s", str(e))
            raise LLMGenerationError(f"Lỗi từ nhà cung cấp LLM: {e}") from e

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        filename_filter: str | None = None,
        history: list[ChatHistoryMessage] | None = None,
    ) -> ChatResponse:
        """Thực hiện toàn bộ luồng Adaptive RAG: Analyze/Route/Rewrite -> Retrieve -> Prompt -> Generate -> Selective Citation."""
        t0 = time.perf_counter()

        # 1. Phân tích ý định câu hỏi, hỏi làm rõ nếu thiếu ngữ cảnh hoặc viết lại truy vấn độc lập
        analysis = self.query_analyzer.analyze(
            question=question,
            history=history,
            filename_filter=filename_filter,
        )
        analyze_ms = (time.perf_counter() - t0) * 1000

        if analysis.intent == "non_academic" and analysis.direct_response:
            logger.info(
                "Adaptive routing | intent=non_academic | question=%r | analyze=%.1fms | skip_retrieval=True",
                question,
                analyze_ms,
            )
            return ChatResponse(
                question=question,
                answer=analysis.direct_response,
                sources=[],
            )

        effective_query = analysis.standalone_query.strip() or question
        rewritten = effective_query != question.strip()
        sub_queries = [
            q.strip()
            for q in analysis.sub_queries
            if isinstance(q, str) and q.strip()
        ] or [effective_query]
        is_multi_query = len(sub_queries) > 1

        # 2. Truy xuất các đoạn văn bản liên quan (đơn truy vấn hoặc đa truy vấn cân bằng)
        t_ret = time.perf_counter()
        if is_multi_query:
            top_k_per_query = max(3, (top_k + 1) // 2)
            max_total_chunks = max(top_k, len(sub_queries) * 3)
            chunks = self.vector_retriever.retrieve_multi_queries(
                queries=sub_queries,
                top_k_per_query=top_k_per_query,
                max_total_chunks=max_total_chunks,
                filename_filter=filename_filter,
            )
        else:
            chunks = self.vector_retriever.retrieve(
                query=effective_query,
                top_k=top_k,
                filename_filter=filename_filter,
            )
        retrieve_ms = (time.perf_counter() - t_ret) * 1000

        # 3. Nếu không tìm thấy ngữ cảnh nào: trả lời an toàn ngay, không gọi LLM để tránh bịa đặt
        if not chunks:
            logger.info(
                "Không tìm thấy ngữ cảnh nào cho truy vấn: '%s' (multi_query=%s | analyze=%.1fms | retrieve=%.1fms)",
                effective_query[:50],
                is_multi_query,
                analyze_ms,
                retrieve_ms,
            )
            return ChatResponse(
                question=question,
                answer=NO_CONTEXT_FALLBACK_ANSWER,
                sources=[],
            )

        # 4. Dựng prompt kèm ngữ cảnh, lịch sử hội thoại và gọi LLM
        t_gen = time.perf_counter()
        prompt = build_rag_prompt(
            question=question,
            chunks=chunks,
            history=history,
            standalone_query=effective_query,
            sub_queries=sub_queries if is_multi_query else None,
        )
        raw_output = self.generate_answer(prompt)
        parsed_output = self._parse_llm_output(raw_output, total_chunks=len(chunks))
        generate_ms = (time.perf_counter() - t_gen) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "RAG timing | intent=academic_rag | rewritten=%s | multi_query=%s | sub_queries=%d | query=%r | analyze=%.1fms | retrieve=%.1fms | generate=%.1fms | total=%.1fms | used_sources=%d/%d",
            rewritten,
            is_multi_query,
            len(sub_queries),
            effective_query,
            analyze_ms,
            retrieve_ms,
            generate_ms,
            total_ms,
            len(parsed_output.used_source_indices),
            len(chunks),
        )
        sources = self._build_sources(chunks, used_indices=parsed_output.used_source_indices)

        return ChatResponse(
            question=question,
            answer=parsed_output.answer,
            sources=sources,
        )


# Instance mặc định dùng cho toàn ứng dụng
rag_generator = RAGGenerator()
