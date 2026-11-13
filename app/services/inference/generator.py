import google.generativeai as genai
from app.core.config import settings
from app.core.exceptions import LLMGenerationError, LLMTimeoutError
from app.core.logging import logger
from app.schemas.response import ChatResponse, RetrievedChunk, SourceCitation
from app.services.inference.prompts import build_rag_prompt
from app.services.inference.retriever import VectorRetriever, vector_retriever

NO_CONTEXT_FALLBACK_ANSWER = (
    "Hiện tại tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu tài liệu học vụ."
)


class RAGGenerator:
    """Bộ điều phối RAG cơ sở (Naive RAG): Truy xuất ngữ cảnh và gọi LLM sinh câu trả lời."""

    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        self.vector_retriever = retriever or vector_retriever
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.CHAT_MODEL
        self._model: genai.GenerativeModel | None = None

    @property
    def model(self) -> genai.GenerativeModel:
        """Khởi tạo Gemini GenerativeModel theo cơ chế lazy loading."""
        if self._model is None:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    @staticmethod
    def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        """Chuyển danh sách chunk đã dùng thành danh sách trích dẫn nguồn cho response."""
        sources: list[SourceCitation] = []
        for chunk in chunks:
            snippet = chunk.content[:200] + ("..." if len(chunk.content) > 200 else "")
            sources.append(
                SourceCitation(
                    filename=chunk.filename or "unknown",
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=round(chunk.score, 4),
                    snippet=snippet,
                )
            )
        return sources

    def generate_answer(self, prompt: str) -> str:
        """Gửi prompt tới Gemini LLM và nhận câu trả lời văn bản."""
        try:
            response = self.model.generate_content(prompt)
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
            if "timeout" in err_msg or "deadline" in err_msg:
                logger.error("Deadline exceeded khi gọi Gemini LLM: %s", str(e))
                raise LLMTimeoutError(f"LLM quá thời gian phản hồi: {e}") from e

            logger.error("Lỗi khi gọi Gemini LLM: %s", str(e))
            raise LLMGenerationError(f"Lỗi từ nhà cung cấp LLM: {e}") from e

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        filename_filter: str | None = None,
    ) -> ChatResponse:
        """Thực hiện toàn bộ luồng RAG: Retrieve -> Prompt -> Generate -> Return."""
        # 1. Truy xuất các đoạn văn bản liên quan
        chunks = self.vector_retriever.retrieve(
            query=question,
            top_k=top_k,
            filename_filter=filename_filter,
        )

        # 2. Nếu không tìm thấy ngữ cảnh nào: trả lời an toàn ngay, không gọi LLM để tránh bịa đặt
        if not chunks:
            logger.info("Không tìm thấy ngữ cảnh nào cho câu hỏi: '%s'", question[:50])
            return ChatResponse(
                question=question,
                answer=NO_CONTEXT_FALLBACK_ANSWER,
                sources=[],
            )

        # 3. Dựng prompt kèm ngữ cảnh và gọi LLM
        prompt = build_rag_prompt(question=question, chunks=chunks)
        answer = self.generate_answer(prompt)
        sources = self._build_sources(chunks)

        return ChatResponse(
            question=question,
            answer=answer,
            sources=sources,
        )


# Instance mặc định dùng cho toàn ứng dụng
rag_generator = RAGGenerator()
