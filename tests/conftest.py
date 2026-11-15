import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.generation.engine import LegalAssistant  # noqa: E402
from app.llm_client import AIClient  # noqa: E402
from app.retrieval.index import KnowledgeBase  # noqa: E402

KB_PATH = ROOT / "data" / "documents" / "knowledge-base.zip"


@pytest.fixture(scope="session")
def kb():
    return KnowledgeBase.from_zip(KB_PATH)


@pytest.fixture
def make_assistant(kb, monkeypatch):
    """Ассистент с подменённой LLM: replies — очередь ответов модели (dict, str или HTTP-код)."""

    def factory(replies):
        monkeypatch.setenv("LLM_BASE_URL", "http://llm.test")
        monkeypatch.setenv("LLM_API_KEY", "test")
        monkeypatch.setenv("DISABLE_EMBEDDINGS", "1")
        monkeypatch.setenv("DISABLE_RERANKER", "1")
        monkeypatch.setenv("LLM_RETRIES", "0")
        queue = list(replies)
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            reply = queue.pop(0) if queue else {"status": "insufficient", "answer": "Нет данных.", "citations": []}
            if isinstance(reply, int):
                return httpx.Response(reply)
            content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        ai = AIClient(transport=httpx.MockTransport(handler))
        assistant = LegalAssistant(kb, ai)
        assistant.calls = calls
        return assistant

    return factory
