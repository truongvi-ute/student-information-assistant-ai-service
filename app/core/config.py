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

    # OpenAI / Embedding
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_BASE_URL: str | None = Field(default=None, description="Custom OpenAI base URL")
    CHAT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Vector DB (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "sia_documents"
    QDRANT_VECTOR_SIZE: int = 1536

    # Chunker Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150


settings = Settings()