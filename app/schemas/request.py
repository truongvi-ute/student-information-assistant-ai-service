from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Yêu cầu truy xuất các đoạn tài liệu liên quan từ Vector DB."""

    query: str = Field(..., min_length=1, description="Câu hỏi hoặc từ khóa cần tìm kiếm")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng đoạn văn bản tối đa cần lấy (1-20)")
    filename_filter: str | None = Field(
        default=None,
        description="Lọc theo tên file tài liệu cụ thể (ví dụ: 'quy_che.pdf')",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Câu truy vấn không được chỉ chứa khoảng trắng")
        return cleaned


class ChatRequest(BaseModel):
    """Yêu cầu hỏi đáp RAG dành cho sinh viên."""

    question: str = Field(..., min_length=1, description="Câu hỏi của sinh viên")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng đoạn ngữ cảnh truy xuất (1-20)")
    filename_filter: str | None = Field(
        default=None,
        description="Tùy chọn chỉ tìm trong một file tài liệu cụ thể (ví dụ: 'quy_che.pdf')",
    )

    @field_validator("question")
    @classmethod
    def validate_question_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Câu hỏi không được chỉ chứa khoảng trắng")
        return cleaned
