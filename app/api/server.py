"""Веб-приложение: чат с юридическим ассистентом и просмотр подтверждающих документов."""

from __future__ import annotations

import csv
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.generation.engine import STATUS_LABELS, AnswerResult, LegalAssistant
from app.generation.security import REMOVED_MARKER
from app.llm_client import AIClient
from app.retrieval import text as tx
from app.retrieval.index import KnowledgeBase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("psb-legal-ai")

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "app" / "web"
STATE: dict = {"assistant": None, "error": None, "indexing": True}


def _load_env_file() -> None:
    """Читает .env без сторонних библиотек (переменные окружения имеют приоритет)."""

    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _kb_path() -> Path:
    candidates = [os.getenv("KB_PATH", ""), "data/kb.zip", "data/knowledge-base.zip", "data/documents/knowledge-base.zip"]
    for candidate in candidates:
        if candidate and (ROOT / candidate).is_file():
            return ROOT / candidate
    raise FileNotFoundError("Архив базы знаний не найден (KB_PATH или data/documents/knowledge-base.zip).")


def _build_index() -> None:
    try:
        ai = AIClient()
        kb = KnowledgeBase.from_zip(_kb_path(), ai=ai)
        STATE["assistant"] = LegalAssistant(kb, ai)   # BM25 готов сразу
        kb.build_vectors()                            # эмбеддинги — если доступен API
        logger.info("Индекс готов: документов %s, фрагментов %s, режим %s", len(kb.documents), len(kb.chunks), kb.search_mode)
    except Exception as error:  # noqa: BLE001
        logger.exception("Не удалось построить индекс")
        STATE["error"] = str(error)
    finally:
        STATE["indexing"] = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_env_file()
    threading.Thread(target=_build_index, daemon=True).start()
    yield
    if STATE["assistant"] is not None and STATE["assistant"].ai is not None:
        STATE["assistant"].ai.close()


app = FastAPI(title="PSB Legal AI", version="2.0.0", lifespan=lifespan)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=20)


def _assistant() -> LegalAssistant:
    if STATE["error"]:
        raise HTTPException(503, f"База знаний не загружена: {STATE['error']}")
    if STATE["assistant"] is None:
        raise HTTPException(503, "База знаний ещё индексируется, повторите запрос через несколько секунд.")
    return STATE["assistant"]


def _doc_meta(assistant: LegalAssistant, doc_id: str) -> dict:
    d = assistant.kb.docs[doc_id]
    if d.is_law:
        validity = f"действующая редакция, выгрузка {tx.format_date(d.extracted_at)}"
    elif d.valid_from:
        validity = f"с {tx.format_date(d.valid_from)}" + (f" до {tx.format_date(d.valid_to)}" if d.valid_to else "")
    else:
        validity = ""
    return {
        "id": d.doc_id, "title": d.title, "short_title": d.short_title, "filename": d.filename,
        "category": "Закон" if d.is_law else "Документ компании", "type": d.doc_type,
        "status": "Утратил силу" if d.status == "repealed" else "Действует", "repealed": d.status == "repealed",
        "validity": validity, "counterparties": d.counterparties, "url": d.source_url,
        "security_note": "Из документа удалён фрагмент с инструкциями для ИИ (попытка prompt-injection)." if d.injections else None,
    }


def _serialize(assistant: LegalAssistant, result: AnswerResult) -> dict:
    citations = [{
        "id": c.source_id, "chunk_id": c.chunk.chunk_id, "label": c.chunk.label or "фрагмент", "quote": c.quote,
        "verified": c.verified, "document": _doc_meta(assistant, c.chunk.doc_id),
    } for c in result.citations]
    cited_docs = {c.chunk.doc_id for c in result.citations}
    consulted = [_doc_meta(assistant, doc_id) for doc_id in dict.fromkeys(chunk.doc_id for _, chunk in result.sources) if doc_id not in cited_docs]
    return {
        "status": result.status, "status_label": STATUS_LABELS.get(result.status, result.status), "answer": result.answer,
        "citations": citations, "consulted": consulted, "issues": result.issues, "notes": result.notes,
        "mode": result.mode, "search_mode": assistant.kb.search_mode, "timings": result.timings,
    }


@app.get("/api/health")
def health() -> dict:
    assistant = STATE["assistant"]
    ai = assistant.ai if assistant else None
    return {
        "status": "error" if STATE["error"] else ("starting" if assistant is None else "ok"),
        "error": STATE["error"], "indexing": STATE["indexing"],
        "documents": len(assistant.kb.documents) if assistant else 0,
        "chunks": len(assistant.kb.chunks) if assistant else 0,
        "search_mode": assistant.kb.search_mode if assistant else None,
        "llm": bool(ai and ai.has_llm), "llm_model": ai.llm_model if ai else None,
        "embeddings": bool(assistant and assistant.kb.vectors is not None),
        "reranker": bool(ai and ai.has_reranker), "api_errors": dict(ai.errors) if ai else {},
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    assistant = _assistant()
    if request.messages[-1].role != "user":
        raise HTTPException(422, "Последнее сообщение должно быть от пользователя.")
    if len(request.messages) == 1:
        raw = request.messages[0].content
    else:
        raw = "\n".join(f"{'Пользователь' if m.role == 'user' else 'Ассистент'}: {m.content}" for m in request.messages[-8:])
    try:
        return _serialize(assistant, assistant.answer(raw))
    except Exception as error:  # noqa: BLE001
        logger.exception("Ошибка обработки запроса")
        raise HTTPException(500, f"Не удалось обработать запрос: {type(error).__name__}") from error


@app.get("/api/documents")
def documents() -> list[dict]:
    assistant = _assistant()
    return [_doc_meta(assistant, d.doc_id) for d in assistant.kb.documents]


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> dict:
    assistant = _assistant()
    if doc_id not in assistant.kb.docs:
        raise HTTPException(404, "Документ не найден.")
    chunks = [{"chunk_id": c.chunk_id, "label": c.label, "text": c.text, "removed": REMOVED_MARKER in c.text,
               "edition": f"ред. с {tx.format_date(c.edition_since)}" if c.edition_since else None}
              for c in assistant.kb.doc_chunks[doc_id]]
    return {**_doc_meta(assistant, doc_id), "chunks": chunks}


@app.get("/api/examples")
def examples() -> list[str]:
    path = ROOT / "data" / "demo_questions.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        questions = [row["question"] for row in csv.DictReader(file)]
    return [q for q in questions if not q.startswith("Пользователь:")][:8]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})
