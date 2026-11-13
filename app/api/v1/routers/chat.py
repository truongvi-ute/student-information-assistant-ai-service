from fastapi import APIRouter, HTTPException
from app.core.exceptions import (
    EmbeddingError,
    LLMGenerationError,
    LLMTimeoutError,
    RetrievalError,
)
from app.schemas.request import ChatRequest, SearchRequest
from app.schemas.response import ChatResponse, SearchResponse
from app.services.inference.generator import rag_generator
from app.services.inference.retriever import vector_retriever

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_documents(payload: SearchRequest):
    """Truy xuất top-k đoạn văn bản liên quan nhất từ Qdrant (Step 3)."""
    try:
        chunks = vector_retriever.retrieve(
            query=payload.query,
            top_k=payload.top_k,
            filename_filter=payload.filename_filter,
        )
        return SearchResponse(
            query=payload.query,
            total_found=len(chunks),
            results=chunks,
        )
    except (RetrievalError, EmbeddingError) as e:
        raise HTTPException(status_code=502, detail=e.message)


@router.post("/chat", response_model=ChatResponse)
async def chat_with_rag(payload: ChatRequest):
    """Hỏi đáp RAG dựa trên tài liệu học vụ kèm trích dẫn nguồn (Step 4)."""
    try:
        return rag_generator.answer_question(
            question=payload.question,
            top_k=payload.top_k,
            filename_filter=payload.filename_filter,
        )
    except LLMTimeoutError as e:
        raise HTTPException(status_code=504, detail=e.message)
    except (LLMGenerationError, RetrievalError, EmbeddingError) as e:
        raise HTTPException(status_code=502, detail=e.message)

