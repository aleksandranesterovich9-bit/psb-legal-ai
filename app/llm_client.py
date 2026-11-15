
import os
import math
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()


class AIClient:
    """
    Клиент для взаимодействия с API моделей:
    - Embeddings
    - Reranker
    - Chat Completions
    """

    def __init__(self):
        self.base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        ).rstrip("/")

        self.llm_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        self.embed_key = (
            os.getenv("EMBEDDER_API_KEY")
            or self.llm_key
        )

        self.rerank_key = (
            os.getenv("RERANKER_API_KEY")
            or self.llm_key
        )

        self.llm_model = (
            os.getenv("LLM_MODEL")
            or os.getenv("GEN_MODEL")
            or os.getenv(
                "OPENAI_MODEL",
                "deepseek/deepseek-v4.1-flash",
            )
        )

        self.embedding_model = os.getenv(
            "EMB_MODEL",
            "baai/bge-m3",
        )

        self.reranker_model = os.getenv(
            "RERANKER_MODEL",
            "cohere/rerank-5-fast",
        )

        self.http = httpx.Client(timeout=60.0)

    def close(self):
        """Закрывает HTTP-клиент."""
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        api_key: str | None,
    ) -> dict[str, Any]:
        """Отправляет POST-запрос к API."""

        if not self.base_url:
            raise RuntimeError(
                "Не задан адрес API. "
                "Укажите LLM_BASE_URL или BASE_URL."
            )

        if not api_key:
            raise RuntimeError(
                f"Не задан API-ключ для {endpoint}."
            )

        response = self.http.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        response.raise_for_status()

        return response.json()

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Преобразует тексты в векторы.

        Возвращает эмбеддинги в том же порядке,
        в котором тексты были переданы.
        """

        if not texts:
            return []

        response = self._post(
            endpoint="/embeddings",
            payload={
                "model": self.embedding_model,
                "input": texts,
                "encoding_format": "float",
            },
            api_key=self.embed_key,
        )

        items = response.get("data")

        if not isinstance(items, list):
            raise RuntimeError(
                "Некорректный ответ API embeddings: "
                "отсутствует список data."
            )

        if len(items) != len(texts):
            raise RuntimeError(
                "Количество полученных эмбеддингов "
                "не совпадает с количеством текстов."
            )

        ordered_vectors = [None] * len(texts)

        for item in items:
            index = item.get("index")
            vector = item.get("embedding")

            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(texts)
                or not isinstance(vector, list)
            ):
                raise RuntimeError(
                    "Некорректный элемент ответа embeddings."
                )

            if ordered_vectors[index] is not None:
                raise RuntimeError(
                    f"Повторный индекс embedding: {index}."
                )

            ordered_vectors[index] = vector

        if any(vector is None for vector in ordered_vectors):
            raise RuntimeError(
                "API не вернул эмбеддинги для всех текстов."
            )

        normalized_vectors = []

        for vector in ordered_vectors:
            norm = math.sqrt(
                sum(float(value) ** 2 for value in vector)
            )

            if norm == 0:
                normalized_vectors.append(
                    [0.0] * len(vector)
                )
            else:
                normalized_vectors.append(
                    [float(value) / norm for value in vector]
                )

        return normalized_vectors

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 2,
    ) -> list[int]:
        """
        Переранжирует документы относительно вопроса.

        Возвращает индексы документов в порядке релевантности.
        """

        if not documents:
            return []

        response = self._post(
            endpoint="/rerank",
            payload={
                "model": self.reranker_model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            api_key=self.rerank_key,
        )

        results = response.get("results")

        if results is None:
            results = response.get("data")

        if not isinstance(results, list):
            raise RuntimeError(
                "Некорректный ответ API reranker."
            )

        indices = []

        for item in results:
            index = item.get("index")

            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(documents)
            ):
                raise RuntimeError(
                    "Reranker вернул некорректный индекс."
                )

            indices.append(index)

        return indices

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Отправляет сообщения LLM и возвращает текст ответа.
        """

        response = self._post(
            endpoint="/chat/completions",
            payload={
                "model": self.llm_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            api_key=self.llm_key,
        )

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "Некорректный ответ API chat/completions."
            )

        if not isinstance(content, str):
            raise RuntimeError(
                "LLM вернула ответ в неожиданном формате."
            )

        return content
