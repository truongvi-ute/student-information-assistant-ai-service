import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from app.services.ingestion.pipeline import IngestionPipeline


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

        pipeline = IngestionPipeline(embed_service=mock_embedding)

        # Mock init_collection để test không phụ thuộc vào kết nối mạng Qdrant
        with patch("app.services.ingestion.pipeline.init_collection"):
            result = pipeline.ingest_path(empty_file)

        assert result.status == "empty"
        assert result.successful_chunks == 0
