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
    CHAT_MODEL: str = "gemini-2.5-flash"

    # Vector DB (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "sia_documents"
    
    # text-embedding-004 của Gemini có 768 chiều (khác với OpenAI 1536)
    QDRANT_VECTOR_SIZE: int = 768

    # Chunker Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150


settings = Settings()
