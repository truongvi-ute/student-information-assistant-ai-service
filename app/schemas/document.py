from typing import Any
from pydantic import BaseModel, Field


class ExtractedDocument(BaseModel):
    """Đoạn/trang tài liệu thô được trích xuất từ nguồn (file .txt, .md hoặc từng trang .pdf)."""

    content: str = Field(..., description="Nội dung văn bản trích xuất")
    # default_factory=dict giúp mỗi object có dict độc lập trong bộ nhớ
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata đi kèm (source, page_number, filename...)",
    )


class Chunk(BaseModel):
    """Đoạn văn bản nhỏ sau khi cắt từ ExtractedDocument, sẵn sàng cho embedding."""

    content: str = Field(..., description="Nội dung đoạn văn bản đã chia nhỏ")
    chunk_index: int = Field(
        ...,
        description="Số thứ tự của chunk trong tài liệu gốc (bắt đầu từ 0)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata kế thừa từ tài liệu gốc và thông tin bổ sung của chunk",
    )


class IngestionResult(BaseModel):
    """Báo cáo kết quả sau khi chạy pipeline nạp tài liệu vào Qdrant."""

    total_files: int = Field(default=0, description="Tổng số file quét được")
    total_chunks: int = Field(default=0, description="Tổng số chunk được tạo ra")
    successful_chunks: int = Field(default=0, description="Số chunk nạp thành công vào Qdrant")
    failed_files: list[str] = Field(
        default_factory=list,
        description="Danh sách tên file gặp lỗi trong quá trình xử lý",
    )
    status: str = Field(default="success", description="Trạng thái thực thi (success, empty, failed...)")


class CollectionStatsResponse(BaseModel):
    """Thông tin thống kê collection vector hiện tại trong Qdrant."""

    collection_name: str = Field(..., description="Tên collection (sia_documents)")
    points_count: int = Field(..., description="Tổng số vector points (chunks) hiện có")
    vector_size: int = Field(..., description="Số chiều của mỗi vector (768)")
    status: str = Field(..., description="Trạng thái hoạt động của collection")
