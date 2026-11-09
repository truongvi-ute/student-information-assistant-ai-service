from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.routers import documents
from app.core.config import settings
from app.core.logging import logger
from app.db.qdrant import init_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động: Kiểm tra hoặc tạo collection trên Qdrant
    logger.info("Đang khởi động %s...", settings.APP_NAME)
    try:
        init_collection()
    except Exception as e:
        logger.warning("Cảnh báo: Chưa kết nối được Qdrant lúc khởi động (%s)", e)
    yield
    logger.info("Đang tắt %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Student Information Assistant AI & RAG Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/v1/documents", tags=["documents"])


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "status": "online",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}