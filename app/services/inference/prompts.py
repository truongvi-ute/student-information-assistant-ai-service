from app.schemas.response import RetrievedChunk

SYSTEM_PROMPT = """Bạn là Trợ lý Thông tin Sinh viên (SIA) chính thức của nhà trường.
Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên một cách chính xác, rõ ràng và lịch sự.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin có trong phần [NGỮ CẢNH TÀI LIỆU] bên dưới để trả lời.
2. Nếu thông tin không có trong [NGỮ CẢNH TÀI LIỆU], hãy trả lời thẳng thắn: "Hiện tại tôi không tìm thấy thông tin này trong các tài liệu quy chế hiện có." Tuyệt đối KHÔNG tự suy đoán hay bịa đặt con số, quy định.
3. Trích dẫn rõ tên tài liệu hoặc số trang (nếu có) trong câu trả lời để sinh viên tiện tra cứu."""


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Ghép nối danh sách các đoạn tài liệu truy xuất được và câu hỏi thành prompt hoàn chỉnh."""
    context_blocks: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        source_label = chunk.filename or "Tài liệu"
        if chunk.page_number is not None:
            source_label += f" - Trang {chunk.page_number}"
        context_blocks.append(f"[Nguồn {idx}: {source_label}]\n{chunk.content}")

    formatted_context = "\n\n---\n\n".join(context_blocks)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"[NGỮ CẢNH TÀI LIỆU]\n{formatted_context}\n\n"
        f"[CÂU HỎI CỦA SINH VIÊN]\n{question}\n\n"
        f"[CÂU TRẢ LỜI]"
    )
