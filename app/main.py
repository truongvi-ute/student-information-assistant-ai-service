from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SIA AI Service",
    description="Student Information Assistant AI & RAG Service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "sia-ai-service",
        "status": "online",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}
