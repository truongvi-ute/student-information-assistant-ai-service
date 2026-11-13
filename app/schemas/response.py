from typing import Any
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Một đoạn văn bản được truy xuất từ Qdrant kèm điểm tương đồng."""

    id: str = Field(..., description="Point ID trong Qdrant")
    content: str = Field(..., description="Nội dung đoạn văn bản")
    score: float = Field(..., description="Điểm tương đồng Cosine")
    chunk_index: int = Field(default=0, description="Số thứ tự chunk trong tài liệu")
    filename: str | None = Field(default=None, description="Tên file tài liệu nguồn")
    page_number: int | None = Field(default=None, description="Số trang (nếu là file PDF)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Toàn bộ metadata đi kèm")


class SearchResponse(BaseModel):
    """Kết quả trả về của API tìm kiếm /v1/search."""

    query: str
    total_found: int
    results: list[RetrievedChunk] = Field(default_factory=list)


class SourceCitation(BaseModel):
    """Thông tin trích dẫn nguồn tài liệu đính kèm câu trả lời."""

    filename: str = Field(..., description="Tên file tài liệu")
    page_number: int | None = Field(default=None, description="Trang số mấy trong file PDF")
    chunk_index: int = Field(default=0, description="Chỉ số đoạn văn bản")
    score: float = Field(..., description="Điểm tương đồng với câu hỏi")
    snippet: str = Field(..., description="Trích đoạn nội dung ngắn")


class ChatResponse(BaseModel):
    """Kết quả trả về của API hỏi đáp /v1/chat."""

    question: str = Field(..., description="Câu hỏi gốc của người dùng")
    answer: str = Field(..., description="Câu trả lời do AI tổng hợp từ tài liệu")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Danh sách các nguồn tài liệu đã dùng để trả lời",
    )
