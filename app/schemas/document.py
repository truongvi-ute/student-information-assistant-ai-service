from typing import Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Đại diện cho một tài liệu thô được đọc từ nguồn."""
    content: str = Field(..., description="Nội dung văn bản")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata kèm theo (source, page, filename...)")


class Chunk(BaseModel):
    """Đại diện cho một đoạn văn bản sau khi cắt nhỏ."""
    content: str = Field(..., description="Nội dung đoạn văn bản")
    chunk_index: int = Field(..., description="Chỉ số thứ tự của chunk trong tài liệu")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata kế thừa và bổ sung")


class IngestionResult(BaseModel):
    """Kết quả tổng hợp của tiến trình nạp tài liệu."""
    total_files: int = 0
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_files: list[str] = Field(default_factory=list)
    status: str = "success"


class DocumentStatsResponse(BaseModel):
    """Thông tin thống kê collection trong Qdrant."""
    collection_name: str
    points_count: int
    vector_size: int
    status: str