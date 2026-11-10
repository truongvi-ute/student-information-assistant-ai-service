from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.core.config import settings
from app.core.exceptions import VectorDBError
from app.core.logging import logger

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Khởi tạo và tái sử dụng QdrantClient singleton."""
    global _client
    if _client is None:
        try:
            _client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY,
                timeout=10.0,
            )
            logger.info("Kết nối Qdrant thành công tại %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT)
        except Exception as e:
            logger.error("Không thể kết nối tới Qdrant: %s", str(e))
            raise VectorDBError(f"Lỗi kết nối Qdrant: {e}") from e
    return _client


def init_collection(client: QdrantClient | None = None, recreate: bool = False) -> None:
    """Tạo collection nếu chưa có, hoặc tạo lại nếu recreate=True."""
    client = client or get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    try:
        exists = client.collection_exists(collection_name)
        if exists and recreate:
            logger.warning("Đang xoá collection cũ: %s", collection_name)
            client.delete_collection(collection_name)
            exists = False

        if not exists:
            logger.info(
                "Đang tạo collection '%s' (vector_size=%d, distance=Cosine)...",
                collection_name,
                settings.QDRANT_VECTOR_SIZE,
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Collection '%s' đã được tạo thành công.", collection_name)
        else:
            logger.info("Collection '%s' đã tồn tại sẵn sàng.", collection_name)
    except Exception as e:
        logger.error("Lỗi khởi tạo collection Qdrant: %s", str(e))
        raise VectorDBError(f"Không thể khởi tạo collection: {e}") from e