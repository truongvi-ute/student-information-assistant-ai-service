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
# Cài đặt tối thiểu cho Step 1 (Web server):
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

# Vector Database (Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
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

## 📁 Directory Structure

```text
sia-ai-service/
├── app/
│   ├── main.py                # FastAPI entry point & lifecycle
│   ├── api/
│   │   ├── dependencies.py    # Common dependencies (DB, services)
│   │   └── v1/routers/        # API route handlers
│   │       ├── chat.py        # Chat & conversation endpoints
│   │       ├── documents.py   # Document ingestion endpoints
│   │       └── health.py      # Health & readiness probes
│   ├── core/                  # Core configurations & cross-cutting concerns
│   │   ├── config.py          # Pydantic Settings
│   │   ├── exceptions.py      # Custom exceptions & handlers
│   │   └── logging.py         # Logger setup
│   ├── db/
│   │   └── qdrant.py          # Qdrant client & collection management
│   ├── schemas/               # Pydantic data schemas
│   │   ├── request.py         # Request models
│   │   └── response.py        # Response models
│   ├── services/
│   │   ├── embedding.py       # Shared embedding service (Ingestion + Inference)
│   │   ├── ingestion/         # Ingestion pipeline modules
│   │   │   ├── loader.py      # File & content loaders
│   │   │   ├── chunker.py     # Text chunking & splitting
│   │   │   └── pipeline.py    # Ingestion orchestration
│   │   └── inference/         # Inference & RAG modules
│   │       ├── retriever.py   # Vector retriever
│   │       ├── reranker.py    # Document reranker
│   │       ├── generator.py   # LLM generation
│   │       ├── agent.py       # Agentic reasoning / router
│   │       └── prompts.py     # Prompt templates
│   └── utils/                 # Utility helper functions
├── scripts/
│   └── ingest.py              # CLI batch ingestion script
├── eval/                      # RAG evaluation scripts & benchmarks
├── tests/                     # Unit and integration tests
├── .env.example               # Environment variables template
├── docker-compose.yml         # Local container orchestration (Qdrant)
├── pyproject.toml             # Project dependencies and packaging
└── README.md                  # Project documentation
```
