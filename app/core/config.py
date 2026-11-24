import threading

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "sia-ai-service"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Gemini API
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_RPM_LIMIT: int = 50  # Đặt 0 nếu dùng gói trả phí để tắt giới hạn chờ mỗi phút
    EMBEDDING_TPM_LIMIT: int = 29000  # Giới hạn token/phút (Free Tier Gemini là 30,000 TPM)
    EMBEDDING_QUERY_MAX_WAIT_SECONDS: float = 65.0
    MAX_CONCURRENT_INGESTIONS: int = 2
    CHAT_MODEL: str = "gemini-2.5-flash"
    GEMINI_TIMEOUT_SECONDS: float = 30.0

    # Vector DB (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "sia_documents"
    
    # text-embedding-004 của Gemini có 768 chiều (khác với OpenAI 1536)
    QDRANT_VECTOR_SIZE: int = 768
    RETRIEVAL_SCORE_THRESHOLD: float = 0.45

    # Chunker Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Storage Settings (RustFS Dev / S3 Prod)
    STORAGE_ENDPOINT_URL: str = "http://localhost:9000"
    STORAGE_ACCESS_KEY: str = "sia-minio"
    STORAGE_SECRET_KEY: str = "12345678"
    STORAGE_BUCKET_NAME: str = "sia-documents"
    STORAGE_REGION: str = "us-east-1"

    # Internal Backend Callback
    BACKEND_INTERNAL_URL: str = "http://localhost:8080"
    INTERNAL_API_KEY: str = "sia-internal-secret-change-me"


settings = Settings()
genai_config_lock = threading.Lock()
