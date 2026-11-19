from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatHistoryMessage(BaseModel):
    """Một lượt tin nhắn trong lịch sử hội thoại trước đó."""

    role: Literal["user", "assistant"] = Field(
        ..., description="Vai trò người gửi: 'user' (sinh viên) hoặc 'assistant' (AI)"
    )
    content: str = Field(..., min_length=1, description="Nội dung tin nhắn")

    @field_validator("content")
    @classmethod
    def validate_content_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Nội dung tin nhắn lịch sử không được rỗng")
        return cleaned


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
    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Danh sách các tin nhắn gần nhất trong cuộc hội thoại (cũ trước, mới sau)",
    )
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
