import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx
import pytest
from fastapi import UploadFile

from app.api.v1.routers.documents import ingest_document
from app.schemas.document import IngestionResult
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
