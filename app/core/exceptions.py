import logging
import sys
from app.core.config import settings


def setup_logging() -> logging.Logger:
    # 1. Tạo một 'người ghi log' (Logger) có tên lấy từ config (sia-ai-service)
    logger = logging.getLogger(settings.APP_NAME)
    
    # 2. Tránh bị in lặp: Chỉ cấu hình nếu người này chưa có bộ phận xuất log (handlers)
    if not logger.handlers:
        
        # 3. Đọc cấp độ log từ .env (ví dụ: "INFO" hoặc "DEBUG")
        level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        # 4. Tạo kênh xuất log (Handler): sys.stdout = in ra màn hình Console (Terminal)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # 5. Định nghĩa mẫu (Template) cho mỗi dòng log được in ra:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Ý nghĩa từng phần:
        # %(asctime)s             -> Ngày giờ (2026-11-09 16:30:00)
        # [%(levelname)s]         -> Cấp độ ([INFO] hoặc [ERROR])
        # %(name)s                -> Tên app (sia-ai-service)
        # (%(filename)s:%(lineno)d) -> Tên file và dòng code bị lỗi (loader.py:42)
        # %(message)s             -> Lời nhắn của bạn

        # 6. Gắn mẫu định dạng vào kênh xuất
        handler.setFormatter(formatter)
        
        # 7. Gắn kênh xuất vào Logger
        logger.addHandler(handler)

    return logger

# 8. Khởi tạo sẵn một đối tượng 'logger' để các file khác chỉ việc import dùng luôn
logger = setup_logging()