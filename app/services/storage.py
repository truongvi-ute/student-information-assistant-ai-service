import threading

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings
from app.core.exceptions import StorageError
from app.core.logging import logger


class StorageAdapter:
    """Adapter kết nối kho lưu trữ đối tượng S3-compatible (RustFS Dev / AWS S3 Prod)."""

    def __init__(
        self,
        s3_client: BaseClient | None = None,
        bucket_name: str | None = None,
    ) -> None:
        self._custom_client = s3_client
        self._lazy_client: BaseClient | None = None
        self._lock = threading.Lock()
        self.bucket_name = bucket_name or settings.STORAGE_BUCKET_NAME

    @property
    def client(self) -> BaseClient:
        if self._custom_client is not None:
            return self._custom_client
        if self._lazy_client is None:
            with self._lock:
                if self._lazy_client is None:
                    self._lazy_client = boto3.client(
                        "s3",
                        endpoint_url=settings.STORAGE_ENDPOINT_URL,
                        aws_access_key_id=settings.STORAGE_ACCESS_KEY,
                        aws_secret_access_key=settings.STORAGE_SECRET_KEY,
                        region_name=settings.STORAGE_REGION,
                        config=Config(s3={"addressing_style": "path"}),
                    )
        return self._lazy_client

    def download_file(self, file_key: str) -> bytes:
        """Tải toàn bộ nội dung nhị phân của file từ RustFS/S3 theo object key."""
        cleaned_key = (file_key or "").strip()
        if not cleaned_key:
            raise StorageError("Thiếu file_key khi tải file từ Storage.")

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=cleaned_key,
            )
            body = response["Body"]
            try:
                return body.read()
            finally:
                if hasattr(body, "close"):
                    body.close()
        except (ClientError, BotoCoreError, KeyError) as e:
            logger.error("Lỗi khi tải file '%s' từ Storage: %s", cleaned_key, str(e))
            raise StorageError(f"Không thể tải file '{cleaned_key}' từ Storage: {e}") from e

    def upload_text(
        self,
        file_key: str,
        content: str,
        content_type: str = "text/markdown; charset=utf-8",
    ) -> None:
        """Ghi nội dung văn bản UTF-8 (như bản Markdown chuẩn hóa) lên RustFS/S3."""
        cleaned_key = (file_key or "").strip()
        if not cleaned_key:
            raise StorageError("Thiếu file_key khi ghi file lên Storage.")

        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=cleaned_key,
                Body=content.encode("utf-8"),
                ContentType=content_type,
            )
        except (ClientError, BotoCoreError) as e:
            logger.error("Lỗi khi ghi file '%s' lên Storage: %s", cleaned_key, str(e))
            raise StorageError(f"Không thể ghi file '{cleaned_key}' lên Storage: {e}") from e


storage_adapter = StorageAdapter()
