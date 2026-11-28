# SIA AI Service

AI and RAG Service for the **Student Information Assistant (SIA)**.

## 🚀 Hướng dẫn cài đặt & Khởi chạy (Quick Start)

### 1. Yêu cầu môi trường
- Python `>= 3.11`
- Docker & Docker Compose (dùng để chạy Qdrant Vector DB)

---

### 2. Thiết lập môi trường ảo (`.venv`)

Mở terminal tại thư mục `sia-ai-service`:

**Tạo môi trường ảo:**
```bash
python -m venv .venv
```

**Kích hoạt môi trường ảo:**
- Trên Windows (PowerShell):
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi script execution, chạy lệnh: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*
- Trên Windows (Command Prompt):
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- Trên Linux / macOS:
  ```bash
  source .venv/bin/activate
  ```

---

### 3. Cài đặt thư viện

Sau khi môi trường ảo đã được kích hoạt (có tiền tố `(.venv)` ở đầu dòng lệnh):

```bash
# Cài đặt tối thiểu (Web server):
pip install fastapi "uvicorn[standard]"

# Hoặc cài đặt đầy đủ theo pyproject.toml khi cần:
pip install -e ".[dev]"
```

---

### 4. Cấu hình biến môi trường (`.env`)

Tạo file `.env` từ file mẫu `.env.example`:
```powershell
cp .env.example .env
```
Cấu hình tối thiểu cho local development:
```ini
APP_NAME=sia-ai-service
APP_ENV=development
APP_PORT=8000

# Google Gemini / LLM
GEMINI_API_KEY=your-gemini-api-key-here
CHAT_MODEL=gemini-2.5-flash

# Vector Database (Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Storage (RustFS Dev) & Internal Backend Callback
STORAGE_ENDPOINT_URL=http://localhost:9000
STORAGE_ACCESS_KEY=sia-minio
STORAGE_SECRET_KEY=12345678
BACKEND_INTERNAL_URL=http://localhost:8080
INTERNAL_API_KEY=sia-internal-secret-change-me
```

---

### 5. Khởi chạy Qdrant bằng Docker

Khởi chạy container Qdrant trong nền:
```bash
docker compose up -d
```
- Qdrant REST API: `http://localhost:6333`
- Qdrant Dashboard UI: `http://localhost:6333/dashboard`

---

### 6. Khởi chạy AI Service với Uvicorn

Đảm bảo `.venv` đang được kích hoạt và chạy lệnh:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* `--host 0.0.0.0`: Lắng nghe mọi địa chỉ IP mạng nội bộ.
* `--port 8000`: Cổng chạy service.
* `--reload`: Tự động khởi động lại server khi có thay đổi trong mã nguồn.

---

### 7. Kiểm tra kết nối

- **Interactive API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 📥 Nạp tài liệu từ đường dẫn qua CLI (Ingestion)

Đảm bảo container Qdrant đang chạy (`docker compose up -d`), đã cấu hình `GEMINI_API_KEY` trong `.env`, và terminal đang đứng tại thư mục `sia-ai-service` (`.venv` đã kích hoạt hoặc gọi trực tiếp Python trong `.venv`).

Các định dạng hỗ trợ: `.txt`, `.md`, `.pdf`, `.docx`.

- **Nạp toàn bộ thư mục mặc định (`./data`):**
  ```bash
  python scripts/ingest.py
  ```

- **Nạp từ một file hoặc thư mục cụ thể (`--path`):**
  ```powershell
  # Ví dụ trên Windows PowerShell trỏ tới file ở thư mục cha
  .\.venv\Scripts\python.exe scripts\ingest.py --path ..\duong_dan_toi_file.pdf
  ```

- **Xóa collection cũ và nạp lại từ đầu (`--recreate`):**
  ```powershell
  .\.venv\Scripts\python.exe scripts\ingest.py --path ..\duong_dan_toi_file.pdf --recreate
  ```

> **Lưu ý về Gemini Free Tier:** Mặc định `EMBEDDING_RPM_LIMIT=50` và `EMBEDDING_TPM_LIMIT=29000` (cấu hình trong `app/core/config.py` hoặc `.env`). Khi nạp tài liệu qua **cả CLI lẫn API ingestion** vượt ngưỡng 50 chunks/phút hoặc ~29,000 tokens/phút, hệ thống sẽ chủ động tạm dừng ~60 giây sang cửa sổ phút mới để tránh lỗi `429`. Đặt `EMBEDDING_RPM_LIMIT=0` trong `.env` nếu dùng gói trả phí.

---

## 📁 Directory Structure

```text
sia-ai-service/
├── app/
│   ├── main.py                      # FastAPI entry point, lifespan & /health
│   ├── api/
│   │   └── v1/routers/              # API route handlers
│   │       ├── chat.py              # Search & RAG chat endpoints
│   │       └── documents.py         # Document ingestion & stats endpoints
│   ├── core/                        # Core configurations & cross-cutting concerns
│   │   ├── config.py                # Pydantic Settings
│   │   ├── exceptions.py            # Custom exceptions
│   │   └── logging.py               # Logger setup
│   ├── db/
│   │   └── qdrant.py                # Qdrant client & collection management
│   ├── schemas/                     # Pydantic data schemas
│   │   ├── document.py              # Ingestion & storage schemas
│   │   ├── request.py               # Request models
│   │   └── response.py              # Response models
│   └── services/
│       ├── embedding.py             # Shared Gemini embedding service (RPM/TPM)
│       ├── storage.py               # RustFS / S3 storage adapter
│       ├── ingestion/               # Ingestion pipeline modules
│       │   ├── converters.py        # PDF/DOCX/TXT/MD -> Markdown converters
│       │   ├── loader.py            # File & bytes loader
│       │   ├── chunker.py           # Structure-aware Markdown chunker
│       │   └── pipeline.py          # Ingestion orchestration
│       └── inference/               # Inference & Adaptive RAG modules
│           ├── query_analyzer.py    # Intent routing, rewriting & sub-queries
│           ├── retriever.py         # Single & balanced multi-query retriever
│           ├── generator.py         # Grounded generation & selective citation
│           └── prompts.py           # Prompt templates
├── scripts/
│   └── ingest.py                    # CLI batch ingestion script
├── eval/                            # RAG evaluation scripts & dataset
├── tests/                           # Unit tests
├── .env.example                     # Environment variables template
├── docker-compose.yml               # Local Qdrant orchestration
├── pyproject.toml                   # Project dependencies and packaging
└── README.md                        # Project documentation
```
