import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.api.v1.routers.documents import _run_storage_ingestion_task, ingest_document
from app.main import app
from app.schemas.document import Chunk, IngestFromStorageRequest, IngestionResult
from app.services.ingestion.pipeline import IngestionPipeline


def _create_docx_bytes(text: str) -> bytes:
    document = docx.Document()
    document.add_heading("Thông báo học vụ", level=1)
    document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_idempotent_point_id():
    """Đảm bảo cùng 1 file và chunk index sẽ luôn sinh cùng Point ID."""
    source = "G:/docs/quy_che.pdf"
    id_1 = IngestionPipeline.generate_point_id(source, chunk_index=0)
    id_2 = IngestionPipeline.generate_point_id(source, chunk_index=0)
    id_diff = IngestionPipeline.generate_point_id(source, chunk_index=1)

    assert id_1 == id_2
    assert id_1 != id_diff


def test_pipeline_with_empty_and_corrupt_files():
    """Kiểm tra pipeline xử lý an toàn khi gặp file rỗng."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        mock_embedding = MagicMock()
        mock_embedding.embed_texts.return_value = []

        test_pipeline = IngestionPipeline(embed_service=mock_embedding)

        with patch("app.services.ingestion.pipeline.init_collection"):
            result = test_pipeline.ingest_path(empty_file)

        assert result.status == "empty"
        assert result.successful_chunks == 0


def test_pipeline_ingest_bytes_docx_preserves_filename_and_metadata():
    """Kiểm tra ingest_bytes với file .docx lưu đúng filename và metadata vào Qdrant payload."""
    docx_bytes = _create_docx_bytes(
        "Sinh viên đăng ký tối thiểu 14 tín chỉ mỗi học kỳ."
    )

    mock_embedding = MagicMock()
    mock_embedding.embed_texts.return_value = [[0.1] * 768]

    mock_qdrant = MagicMock()
    with patch(
        "app.services.ingestion.pipeline.get_qdrant_client", return_value=mock_qdrant
    ):
        test_pipeline = IngestionPipeline(embed_service=mock_embedding)

    with patch("app.services.ingestion.pipeline.init_collection"):
        result = test_pipeline.ingest_bytes(
            file_bytes=docx_bytes,
            filename="quy_che_2025.docx",
            metadata={"document_id": "doc-123", "faculty": "CNTT"},
        )

    assert result.status == "success"
    assert result.successful_chunks == 1

    upserted_points = mock_qdrant.upsert.call_args.kwargs["points"]
    payload = upserted_points[0].payload
    assert payload["filename"] == "quy_che_2025.docx"
    assert payload["extension"] == ".docx"
    assert payload["document_id"] == "doc-123"
    assert payload["faculty"] == "CNTT"
    assert "# Thông báo học vụ" in payload["content"]


def test_upload_endpoint_accepts_docx():
    """Kiểm tra endpoint POST /v1/documents/ingest chấp nhận file .docx và truyền đúng filename."""
    docx_bytes = _create_docx_bytes("Nội dung thử nghiệm upload DOCX.")

    async def _read_bytes() -> bytes:
        return docx_bytes

    upload_file = MagicMock(spec=UploadFile)
    upload_file.filename = "huong_dan.docx"
    upload_file.read = _read_bytes

    with patch(
        "app.api.v1.routers.documents.ingestion_pipeline.ingest_bytes"
    ) as mock_ingest:
        mock_ingest.return_value = IngestionResult(
            total_files=1,
            total_chunks=1,
            successful_chunks=1,
            failed_files=[],
            status="success",
        )
        coro = ingest_document(file=upload_file)
        with pytest.raises(StopIteration) as exc_info:
            coro.send(None)
        result = exc_info.value.value

    assert result.status == "success"
    assert result.successful_chunks == 1
    mock_ingest.assert_called_once_with(
        file_bytes=docx_bytes,
        filename="huong_dan.docx",
    )


def test_pipeline_ingest_from_storage_uploads_markdown_and_deletes_old_points():
    """Kiểm tra ingest_from_storage tải file từ RustFS, xóa point cũ, upsert payload kèm offset và lưu extracted markdown."""
    docx_bytes = _create_docx_bytes("Sinh viên nộp học phí đúng hạn quy định.")

    mock_storage = MagicMock()
    mock_storage.download_file.return_value = docx_bytes

    mock_embedding = MagicMock()
    mock_embedding.embed_texts.return_value = [[0.2] * 768]

    mock_qdrant = MagicMock()
    with patch(
        "app.services.ingestion.pipeline.get_qdrant_client", return_value=mock_qdrant
    ):
        test_pipeline = IngestionPipeline(
            embed_service=mock_embedding,
            storage=mock_storage,
        )

    with patch("app.services.ingestion.pipeline.init_collection"):
        result = test_pipeline.ingest_from_storage(
            document_id="doc-99",
            file_key="documents/doc-99.docx",
            filename="hoc_phi.docx",
            metadata={"title": "Quy định học phí"},
        )

    assert result.status == "success"
    assert result.successful_chunks == 1
    mock_storage.download_file.assert_called_once_with("documents/doc-99.docx")
    mock_qdrant.delete.assert_called_once()

    upserted_points = mock_qdrant.upsert.call_args.kwargs["points"]
    payload = upserted_points[0].payload
    assert payload["document_id"] == "doc-99"
    assert payload["storage_path"] == "documents/doc-99.docx"
    assert payload["filename"] == "hoc_phi.docx"
    assert payload["char_start"] == 0
    assert payload["char_end"] > 0

    mock_storage.upload_text.assert_called_once()
    uploaded_key, uploaded_markdown = mock_storage.upload_text.call_args.args
    assert uploaded_key == "extracted/doc-99.md"
    assert "# Thông báo học vụ" in uploaded_markdown


def test_ingest_from_storage_endpoint_validates_and_accepts():
    """Kiểm tra endpoint POST /v1/documents/ingest-from-storage trả 400 nếu đuôi file sai và 202 nếu hợp lệ."""
    client = TestClient(app)

    bad_res = client.post(
        "/v1/documents/ingest-from-storage",
        json={
            "document_id": "doc-1",
            "file_key": "documents/doc-1.exe",
            "filename": "virus.exe",
        },
    )
    assert bad_res.status_code == 400

    with patch("app.api.v1.routers.documents._run_storage_ingestion_task") as mock_task:
        ok_res = client.post(
            "/v1/documents/ingest-from-storage",
            json={
                "document_id": "doc-2",
                "file_key": "documents/doc-2.pdf",
                "filename": "quy_che.pdf",
            },
        )

    assert ok_res.status_code == 202
    assert ok_res.json()["status"] == "processing"
    assert ok_res.json()["document_id"] == "doc-2"
    mock_task.assert_called_once()


def test_run_storage_ingestion_task_sends_ready_and_failed_callbacks():
    """Kiểm tra background task gửi callback READY hoặc FAILED kèm X-Internal-Api-Key."""
    payload = IngestFromStorageRequest(
        document_id="doc-3",
        file_key="documents/doc-3.pdf",
        filename="quy_che.pdf",
        callback_url="http://localhost:8080/api/internal/documents/doc-3/rag-status",
    )

    with (
        patch(
            "app.api.v1.routers.documents.ingestion_pipeline.ingest_from_storage",
            return_value=IngestionResult(
                total_files=1,
                total_chunks=4,
                successful_chunks=4,
                failed_files=[],
                status="success",
            ),
        ),
        patch("app.api.v1.routers.documents.httpx.post") as mock_post,
    ):
        mock_post.return_value = MagicMock()
        _run_storage_ingestion_task(payload)

    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["json"]["status"] == "READY"
    assert mock_post.call_args.kwargs["json"]["total_chunks"] == 4
    assert "X-Internal-Api-Key" in mock_post.call_args.kwargs["headers"]

    with (
        patch(
            "app.api.v1.routers.documents.ingestion_pipeline.ingest_from_storage",
            side_effect=RuntimeError("RustFS timeout"),
        ),
        patch("app.api.v1.routers.documents.httpx.post") as mock_post_err,
    ):
        mock_post_err.return_value = MagicMock()
        _run_storage_ingestion_task(payload)

    assert mock_post_err.call_count == 1
    assert mock_post_err.call_args.kwargs["json"]["status"] == "FAILED"
    assert "RustFS timeout" in mock_post_err.call_args.kwargs["json"]["error_message"]


def test_pipeline_prepends_title_and_description_only_to_embedding_text():
    """Kiểm tra embed_texts nhận tiền tố [Tài liệu | Mô tả] nhưng Qdrant payload giữ nguyên content gốc."""
    docx_bytes = _create_docx_bytes("Sinh viên học mật mã ứng dụng.")

    mock_embedding = MagicMock()
    mock_embedding.embed_texts.return_value = [[0.3] * 768]

    mock_qdrant = MagicMock()
    with patch(
        "app.services.ingestion.pipeline.get_qdrant_client", return_value=mock_qdrant
    ):
        test_pipeline = IngestionPipeline(embed_service=mock_embedding)

    with patch("app.services.ingestion.pipeline.init_collection"):
        result = test_pipeline.ingest_bytes(
            file_bytes=docx_bytes,
            filename="53f866dc-442b-4b57-a958-aa88430811eb_QD-CTDT-ATTT.docx",
            metadata={
                "title": "Chương trình đào tạo ngành An toàn thông tin (2026)",
                "description": "Ngành An toàn thông tin (ATTT), mã 7480202",
            },
        )

    assert result.status == "success"
    embedded_texts = mock_embedding.embed_texts.call_args.args[0]
    assert len(embedded_texts) == 1
    assert embedded_texts[0].startswith(
        "[Tài liệu: Chương trình đào tạo ngành An toàn thông tin (2026) | Mô tả: Ngành An toàn thông tin (ATTT), mã 7480202]\n"
    )
    assert "Sinh viên học mật mã ứng dụng." in embedded_texts[0]

    upserted_points = mock_qdrant.upsert.call_args.kwargs["points"]
    payload_content = upserted_points[0].payload["content"]
    assert not payload_content.startswith("[Tài liệu:")
    assert "Mô tả:" not in payload_content
    assert payload_content.startswith("# Thông báo học vụ")


def test_build_embedding_text_truncates_to_100_chars_and_strips_uuid_filename():
    """Kiểm tra cắt gọn tối đa 100 ký tự/phần và bỏ tiền tố UUID khi fallback về filename."""
    long_title = "T" * 120
    long_desc = "D" * 130
    chunk_with_long_meta = Chunk(
        content="Nội dung bảng môn học.",
        chunk_index=0,
        metadata={"title": long_title, "description": long_desc},
    )

    embedded_long = IngestionPipeline._build_embedding_text(chunk_with_long_meta)
    expected_prefix = f"[Tài liệu: {'T' * 100} | Mô tả: {'D' * 100}]\n"
    assert embedded_long == f"{expected_prefix}Nội dung bảng môn học."

    chunk_fallback = Chunk(
        content="Bảng điều kiện tốt nghiệp.",
        chunk_index=1,
        metadata={
            "title": "   ",
            "filename": "53f866dc-442b-4b57-a958-aa88430811eb_QD-BanhanhCTDT-Kythuatdulieu-2026.pdf",
        },
    )
    embedded_fallback = IngestionPipeline._build_embedding_text(chunk_fallback)
    assert (
        embedded_fallback
        == "[Tài liệu: QD-BanhanhCTDT-Kythuatdulieu-2026.pdf]\nBảng điều kiện tốt nghiệp."
    )
