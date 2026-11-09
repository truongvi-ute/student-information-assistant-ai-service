# 1. Tạo ra lỗi "GỐC" của toàn bộ dự án SIA
class SIAException(Exception):
    """Mọi lỗi do chúng ta tự viết ra sau này đều là con cháu của class này."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message        # Lời nhắn giải thích lỗi (ví dụ: "File bị hỏng")
        self.details = details or {}  # Thông tin phụ kèm theo (ví dụ: {"file": "abc.pdf"})


# 2. Nhóm lỗi liên quan đến khâu NẠP TÀI LIỆU (kế thừa từ SIAException)
class IngestionError(SIAException):
    pass  # 'pass' nghĩa là giữ nguyên như class cha, chỉ cần một CÁI TÊN RIÊNG BIỆT


# 3. Lỗi cụ thể khi ĐỌC FILE (là con của IngestionError)
class DocumentLoadError(IngestionError):
    pass  # Dùng khi: file rỗng, file không đúng định dạng .txt/.pdf


# 4. Lỗi liên quan đến DATABASE QDRANT
class VectorDBError(SIAException):
    pass  # Dùng khi: Quên bật Docker Qdrant, mất mạng, sai tên collection


# 5. Lỗi khi gọi MODEL EMBEDDING
class EmbeddingError(SIAException):
    pass  # Dùng khi: Hết tiền OpenAI, sai API Key, sai số chiều vector