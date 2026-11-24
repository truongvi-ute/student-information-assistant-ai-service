import threading
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import DocumentLoadError, IngestionError
from app.core.logging import logger
from app.db.qdrant import get_qdrant_client
from app.schemas.document import (
    CollectionStatsResponse,
    IngestFromStorageRequest,
    IngestFromStorageResponse,
    IngestionResult,
)
from app.services.ingestion.loader import document_loader
from app.services.ingestion.pipeline import ingestion_pipeline

router = APIRouter()
_ingestion_semaphore = threading.BoundedSemaphore(max(1, settings.MAX_CONCURRENT_INGESTIONS))


def _resolve_callback_url(payload: IngestFromStorageRequest) -> str:
    if payload.callback_url and payload.callback_url.strip():
        return payload.callback_url.strip()
    base = settings.BACKEND_INTERNAL_URL.rstrip("/")
    return f"{base}/api/internal/documents/{payload.document_id}/rag-status"


def _send_rag_status_callback(callback_url: str, body: dict[str, Any]) -> None:
    try:
        response = httpx.post(
            callback_url,
            json=body,
            headers={"X-Internal-Api-Key": settings.INTERNAL_API_KEY},
            timeout=10.0,
        )
        response.raise_for_status()
        logger.info("Đã gửi callback trạng thái RAG tới %s: %s", callback_url, body.get("status"))
    except Exception as e:
        logger.error("Không thể gửi callback trạng thái RAG tới %s: %s", callback_url, str(e))


def _run_storage_ingestion_task(payload: IngestFromStorageRequest) -> None:
    callback_url = _resolve_callback_url(payload)
    with _ingestion_semaphore:
        try:
            result = ingestion_pipeline.ingest_from_storage(
                document_id=payload.document_id,
                file_key=payload.file_key,
                filename=payload.filename,
                metadata=payload.metadata,
            )
            if result.status == "success" and result.successful_chunks > 0:
                _send_rag_status_callback(
                    callback_url,
                    {
                        "status": "READY",
                        "total_chunks": result.successful_chunks,
                        "error_message": None,
                    },
                )
            else:
                _send_rag_status_callback(
                    callback_url,
                    {
                        "status": "FAILED",
                        "total_chunks": 0,
                        "error_message": f"Tài liệu không tạo được dữ liệu vector (status={result.status}).",
                    },
                )
        except Exception as e:
            logger.error(
                "Lỗi xử lý nền khi nạp tài liệu '%s' (%s): %s",
                payload.document_id,
                payload.filename,
                str(e),
            )
            _send_rag_status_callback(
                callback_url,
                {
                    "status": "FAILED",
                    "total_chunks": 0,
                    "error_message": str(e),
                },
            )


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
        return await run_in_threadpool(
            ingestion_pipeline.ingest_bytes,
            file_bytes=file_bytes,
            filename=filename,
        )
    except (DocumentLoadError, IngestionError) as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi nạp tài liệu: {str(e)}"
        ) from e


@router.post(
    "/ingest-from-storage",
    status_code=202,
    response_model=IngestFromStorageResponse,
)
async def ingest_document_from_storage(
    payload: IngestFromStorageRequest,
    background_tasks: BackgroundTasks,
):
    """Nhận yêu cầu từ Backend để kéo file từ RustFS/S3 và nạp ngầm vào Qdrant."""
    filename = Path(payload.filename).name
    if not filename:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ.")

    suffix = Path(filename).suffix.lower()
    if suffix not in document_loader.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(document_loader.SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng '{suffix}' không được hỗ trợ. Chỉ chấp nhận: {supported}",
        )

    normalized_payload = payload.model_copy(update={"filename": filename})
    background_tasks.add_task(_run_storage_ingestion_task, normalized_payload)

    return IngestFromStorageResponse(
        status="processing",
        document_id=normalized_payload.document_id,
        message="Đã tiếp nhận tài liệu và đang xử lý nạp vào Qdrant trong nền.",
    )


@router.get("/collection-stats", response_model=CollectionStatsResponse)
def get_collection_stats():
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
