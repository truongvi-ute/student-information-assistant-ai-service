import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.core.config import settings
from app.db.qdrant import get_qdrant_client
from app.schemas.document import DocumentStatsResponse, IngestionResult
from app.services.ingestion.pipeline import pipeline

router = APIRouter()


@router.post("/upload", response_model=IngestionResult)
async def upload_and_ingest(file: UploadFile = File(...)):
    """Upload một file tài liệu (.txt, .md, .pdf) và nạp ngay vào Vector DB."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ chấp nhận .txt, .md, .pdf",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = Path(tmp_file.name)
        shutil.copyfileobj(file.file, tmp_file)

    try:
        result = pipeline.ingest_path(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi nạp tài liệu: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_collection_stats():
    """Xem số lượng vector và thông số collection hiện tại trên Qdrant."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.QDRANT_COLLECTION)
        return DocumentStatsResponse(
            collection_name=settings.QDRANT_COLLECTION,
            points_count=info.points_count or 0,
            vector_size=settings.QDRANT_VECTOR_SIZE,
            status=info.status.value if info.status else "unknown",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn Qdrant: {str(e)}")
