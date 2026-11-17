import io
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.core.exceptions import StorageError
from app.services.storage import StorageAdapter


def test_storage_adapter_creates_path_style_s3_client():
    """Kiểm tra StorageAdapter khởi tạo boto3 client với addressing_style='path' cho RustFS."""
    mock_client = MagicMock()
    with patch("app.services.storage.boto3.client", return_value=mock_client) as mock_boto:
        adapter = StorageAdapter(bucket_name="sia-documents")
        client = adapter.client

    assert client is mock_client
    call_kwargs = mock_boto.call_args.kwargs
    assert call_kwargs["config"].s3 == {"addressing_style": "path"}


def test_storage_adapter_download_file_success_and_empty_key():
    """Kiểm tra download_file đọc đúng bytes và từ chối key rỗng."""
    body_stream = io.BytesIO(b"hello rustfs")
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": body_stream}

    adapter = StorageAdapter(s3_client=mock_client, bucket_name="sia-documents")
    data = adapter.download_file("documents/sample.pdf")

    assert data == b"hello rustfs"
    mock_client.get_object.assert_called_once_with(
        Bucket="sia-documents",
        Key="documents/sample.pdf",
    )

    with pytest.raises(StorageError):
        adapter.download_file("   ")


def test_storage_adapter_upload_text_and_error_handling():
    """Kiểm tra upload_text ghi UTF-8 markdown và map ClientError sang StorageError."""
    mock_client = MagicMock()
    adapter = StorageAdapter(s3_client=mock_client, bucket_name="sia-documents")

    adapter.upload_text("extracted/doc-1.md", "# Quy chế đào tạo")
    mock_client.put_object.assert_called_once_with(
        Bucket="sia-documents",
        Key="extracted/doc-1.md",
        Body="# Quy chế đào tạo".encode("utf-8"),
        ContentType="text/markdown; charset=utf-8",
    )

    mock_client.put_object.side_effect = ClientError(
        {"Error": {"Code": "InternalError", "Message": "S3 failed"}},
        "PutObject",
    )
    with pytest.raises(StorageError):
        adapter.upload_text("extracted/doc-1.md", "content")
