# SIA AI Service

AI and RAG Service for the **Student Information Assistant (SIA)**.

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
├── Dockerfile                 # Container image definition
├── docker-compose.yml         # Local container orchestration
├── pyproject.toml             # Project dependencies and packaging
└── README.md                  # Project documentation
```
