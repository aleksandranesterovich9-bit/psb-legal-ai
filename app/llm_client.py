"""Клиент OpenAI-совместимого API организаторов: chat, embeddings, rerank.

Имена переменных окружения совпадают с официальным примером сабмита.
Сбой эмбеддера или реранкера не останавливает работу: вызывающий код
переходит на лексический поиск.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return max(0.0, min(2.0, float(env(name))))
    except ValueError:
        return default


class AIClient:
    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.base_url = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
        self.llm_key = env("LLM_API_KEY") or env("OPENAI_API_KEY")
        self.embed_key = env("EMBEDDER_API_KEY") or self.llm_key
        self.rerank_key = env("RERANKER_API_KEY") or self.llm_key
        self.llm_model = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
        self.embedding_model = env("EMB_MODEL", "baai/bge-m3")
        self.reranker_model = env("RERANKER_MODEL", "qwen/qwen3-reranker-8b")
        self.temperature = env_float("TEMPERATURE", 0.0)
        self.max_tokens = env_int("MAX_TOKENS", 6000, 256, 16000)
        self.retries = env_int("LLM_RETRIES", 2, 0, 5)
        self.http = httpx.Client(timeout=httpx.Timeout(env_int("LLM_TIMEOUT", 90, 5, 600), connect=15.0), transport=transport)
        self.errors: dict[str, str] = {}
        self.failures: dict[str, int] = {}
        self.disabled: set[str] = set()   # эндпоинты, которых нет у провайдера
        # Общий бюджет времени (лимит платформы — 20 минут): после дедлайна вызовы API не делаются,
        # остальные ответы собираются без LLM, чтобы submission.csv был записан полностью.
        self.deadline: float | None = None

    @property
    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    @property
    def has_llm(self) -> bool:
        return bool(self.base_url and self.llm_key) and not self.expired

    @property
    def has_embeddings(self) -> bool:
        return bool(self.base_url and self.embed_key) and not self.expired and env("DISABLE_EMBEDDINGS") != "1" and "/embeddings" not in self.disabled

    @property
    def has_reranker(self) -> bool:
        return bool(self.base_url and self.rerank_key) and not self.expired and env("DISABLE_RERANKER") != "1" and "/rerank" not in self.disabled

    def close(self) -> None:
        self.http.close()

    def _post(self, endpoint: str, payload: dict[str, Any], api_key: str, retries: int | None = None) -> dict[str, Any]:
        if self.expired:
            raise RuntimeError("Исчерпан бюджет времени запуска.")
        if not self.base_url or not api_key:
            raise RuntimeError(f"API не настроен для {endpoint}: нужны LLM_BASE_URL и ключ.")
        attempts = (self.retries if retries is None else retries) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.http.post(
                    self.base_url + endpoint, json=payload, headers={"Authorization": "Bearer " + api_key}
                )
                if response.status_code in (408, 409, 425, 429) or response.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
                response.raise_for_status()
                self.errors.pop(endpoint, None)
                self.failures.pop(endpoint, None)
                return response.json()
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status is not None and status < 500 and status not in (408, 409, 425, 429):
                    break  # ошибка запроса — повтор не поможет
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
        self.errors[endpoint] = f"{type(last_error).__name__}: {last_error}"[:200]
        # Вспомогательный эндпоинт, который не отвечает (нет у провайдера, неверная модель),
        # отключается до перезапуска, чтобы не тратить время на каждый вопрос.
        self.failures[endpoint] = self.failures.get(endpoint, 0) + 1
        status = getattr(getattr(last_error, "response", None), "status_code", None)
        if endpoint != "/chat/completions" and (self.failures[endpoint] >= 2 or (status and 400 <= status < 500 and status != 429)):
            self.disabled.add(endpoint)
            logger.warning("Эндпоинт %s отключён: %s", endpoint, self.errors[endpoint])
        raise RuntimeError(f"Запрос {endpoint} не выполнен: {self.errors[endpoint]}")

    def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        """Ответ модели. DeepSeek V4.1 «думает» перед ответом: рассуждения расходуют
        max_tokens, и при нехватке бюджета content приходит пустым (finish_reason=length).
        Тогда запрос повторяется с удвоенным бюджетом и пониженным усилием рассуждения."""

        budget = max_tokens or self.max_tokens
        extra: dict[str, Any] = {}
        effort = env("LLM_REASONING_EFFORT")
        if effort:
            extra["reasoning"] = {"effort": effort}
        for attempt in range(2):
            data = self._post("/chat/completions", {
                "model": self.llm_model, "messages": messages, "temperature": self.temperature,
                "max_tokens": budget, **extra,
            }, self.llm_key)
            try:
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError) as error:
                raise RuntimeError("Некорректный ответ chat/completions.") from error
            if isinstance(content, str) and content.strip():
                return content.strip()
            if attempt == 0 and choice.get("finish_reason") == "length":
                budget = min(budget * 2, 16000)
                extra = {"reasoning": {"effort": "low"}}
                continue
            break
        raise RuntimeError("Модель вернула пустой ответ.")

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            data = self._post("/embeddings", {"model": self.embedding_model, "input": batch, "encoding_format": "float"}, self.embed_key)
            items = data.get("data")
            if not isinstance(items, list) or len(items) != len(batch):
                raise RuntimeError("Некорректный ответ embeddings.")
            ordered: list[list[float] | None] = [None] * len(batch)
            for position, item in enumerate(items):
                index = int(item.get("index", position))
                vector = [float(value) for value in item["embedding"]]
                norm = math.sqrt(sum(value * value for value in vector)) or 1.0
                if not 0 <= index < len(batch) or not math.isfinite(norm):
                    raise RuntimeError("Некорректный элемент embeddings.")
                ordered[index] = [value / norm for value in vector]
            if any(vector is None for vector in ordered):
                raise RuntimeError("Получены не все embeddings.")
            vectors.extend(ordered)  # type: ignore[arg-type]
        return vectors

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float | None]]:
        """Возвращает (индекс документа, оценка релевантности) по убыванию релевантности."""

        if not documents:
            return []
        data = self._post("/rerank", {
            "model": self.reranker_model, "query": query, "documents": documents, "top_n": min(top_n, len(documents)),
        }, self.rerank_key, retries=1)
        items = data.get("results", data.get("data"))
        if not isinstance(items, list):
            raise RuntimeError("Некорректный ответ rerank.")
        result: list[tuple[int, float | None]] = []
        for item in items:
            index = int(item["index"])
            if not 0 <= index < len(documents) or any(index == seen for seen, _ in result):
                continue
            score = item.get("relevance_score", item.get("score"))
            result.append((index, float(score) if isinstance(score, (int, float)) else None))
        return result
