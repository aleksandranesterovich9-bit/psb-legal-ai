
import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents
from app.llm_client import AIClient
from app.retrieval.bm25_search import BM25Retriever
from app.retrieval.vector_search import VectorRetriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranked_search import RerankedRetriever
from app.generation.answer_generator import LegalAnswerGenerator


logger = logging.getLogger(__name__)

app = FastAPI(
    title="PSB Legal AI",
    description="AI Legal Assistant for legal professionals",
    version="1.0.0",
)

COLLECTION_NAME = "legal_documents"
VECTOR_SIZE = 1024
KB_PATH = Path("data/documents/knowledge-base.zip")

qdrant_client = QdrantClient(url="http://localhost:6333")
ai_client = AIClient()

app.state.answer_generator = None
app.state.indexed = False
app.state.qdrant_available = False


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None


@app.on_event("startup")
async def startup_event():
    """Подключается к Qdrant; при его отсутствии оставляет BM25-режим."""

    try:
        collections = qdrant_client.get_collections()
        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

        app.state.qdrant_available = True
        logger.info("Qdrant готов: %s", COLLECTION_NAME)
    except Exception:
        app.state.qdrant_available = False
        logger.warning("Qdrant недоступен. Приложение продолжит работу в BM25-режиме.", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Закрывает подключения при остановке приложения."""

    ai_client.close()
    qdrant_client.close()


@app.get("/")
async def root():
    return {
        "message": "Welcome to PSB Legal AI",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "indexed": app.state.indexed,
    }


@app.get("/qdrant-health")
async def qdrant_health():
    try:
        qdrant_client.get_collections()

        return {
            "status": "connected",
            "database": "Qdrant",
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant connection failed: {error}",
        )


@app.get("/collection-health")
async def collection_health():
    try:
        collection_info = qdrant_client.get_collection(
            collection_name=COLLECTION_NAME,
        )

        return {
            "status": "ready",
            "collection": COLLECTION_NAME,
            "vectors_count": collection_info.points_count,
            "vector_size": VECTOR_SIZE,
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )


@app.post("/index")
async def index_documents():
    """Загружает документы, создаёт чанки и индексы поиска."""

    if not KB_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Knowledge base not found: {KB_PATH}",
        )

    try:
        documents = load_documents_from_zip(KB_PATH)

        if not documents:
            raise ValueError("В архиве не найдено документов.")

        chunks = chunk_documents(documents)

        if not chunks:
            raise ValueError("Не удалось создать чанки.")

        bm25_retriever = BM25Retriever(chunks)

        vector_retriever = None
        indexed_count = 0

        # Эмбеддинги включаются только при наличии API и ключа.
        # При отсутствии доступа проект всё равно работает через BM25.
        if ai_client.base_url and ai_client.embed_key and app.state.qdrant_available:
            try:
                vector_retriever = VectorRetriever(
                    qdrant_client=qdrant_client,
                    ai_client=ai_client,
                    collection_name=COLLECTION_NAME,
                    vector_size=VECTOR_SIZE,
                )
                indexed_count = vector_retriever.index_chunks(chunks)
            except Exception:
                logger.exception("Не удалось создать векторный индекс; переключаемся на BM25.")
                vector_retriever = None
                indexed_count = 0

        hybrid_retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
        )

        reranked_retriever = RerankedRetriever(
            hybrid_retriever=hybrid_retriever,
            ai_client=ai_client,
        )

        app.state.answer_generator = LegalAnswerGenerator(
            retriever=reranked_retriever,
            ai_client=ai_client,
        )

        app.state.indexed = True

        return {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "vectors_indexed": indexed_count,
            "search_mode": "hybrid" if vector_retriever is not None else "bm25",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {error}",
        )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Отвечает на юридический вопрос пользователя."""

    if not app.state.indexed:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not indexed. Call POST /index first.",
        )

    try:
        result = app.state.answer_generator.answer(
            question=request.question,
            top_k=request.top_k,
            category=request.category,
        )

        return {
            "answer": result.answer,
            "sources": [
                asdict(source)
                for source in result.sources
            ],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {error}",
        )
