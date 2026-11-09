import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.logging import logger
from app.services.ingestion.pipeline import pipeline


def main():
    parser = argparse.ArgumentParser(description="SIA Ingestion CLI - Nạp tài liệu vào Qdrant")
    parser.add_argument(
        "--path",
        type=str,
        default="./data",
        help="Đường dẫn file hoặc thư mục tài liệu cần nạp (mặc định: ./data)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xoá và tạo mới lại collection trước khi nạp",
    )
    args = parser.parse_args()

    target_path = Path(args.path)
    if not target_path.exists():
        logger.error("Đường dẫn không tồn tại: %s", target_path)
        sys.exit(1)

    print("=" * 60)
    print(f"BẮT ĐẦU NẠP TÀI LIỆU VÀO QDRANT ({settings.QDRANT_COLLECTION})")
    print(f"Đường dẫn: {target_path.resolve()}")
    print(f"Tạo lại collection: {args.recreate}")
    print("=" * 60)

    try:
        result = pipeline.ingest_path(target_path, recreate_collection=args.recreate)
        print("\n--- KẾT QUẢ INGESTION ---")
        print(f"Trạng thái:            {result.status}")
        print(f"Tổng số chunks nạp:   {result.successful_chunks}/{result.total_chunks}")
        if result.failed_files:
            print(f"Các file gặp lỗi:     {', '.join(result.failed_files)}")
        print("Hoàn tất!")
    except Exception as e:
        logger.error("Quá trình nạp thất bại: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()