from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.exceptions import DocumentLoadError, IngestionError
from app.db.qdrant import get_qdrant_client
from app.schemas.document import CollectionStatsResponse, IngestionResult
from app.services.ingestion.loader import document_loader
from app.services.ingestion.pipeline import ingestion_pipeline

router = APIRouter()


@router.post("/ingest", response_model=IngestionResult)
async def ingest_document(file: UploadFile = File(...)):
    """Nhận một file tài liệu (.txt, .md, .pdf, .docx) và nạp vào Qdrant Vector DB."""
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ.")

    suffix = Path(filename).suffix.lower()
    if suffix not in document_loader.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(document_loader.SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ chấp nhận: {supported}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File tải lên bị rỗng.")

    try:
        return ingestion_pipeline.ingest_bytes(
            file_bytes=file_bytes,
            filename=filename,
        )
    except (DocumentLoadError, IngestionError) as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi nạp tài liệu: {str(e)}"
        ) from e


@router.get("/collection-stats", response_model=CollectionStatsResponse)
async def get_collection_stats():
    """Xem số lượng vector points và thông số collection hiện tại trên Qdrant."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.QDRANT_COLLECTION)
        return CollectionStatsResponse(
            collection_name=settings.QDRANT_COLLECTION,
            points_count=info.points_count or 0,
            vector_size=settings.QDRANT_VECTOR_SIZE,
            status=info.status.value if info.status else "unknown",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi truy vấn Qdrant: {str(e)}"
        ) from e
