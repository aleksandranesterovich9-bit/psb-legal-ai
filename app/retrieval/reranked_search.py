import logging

from app.llm_client import AIClient
from app.retrieval.bm25_search import SearchResult
from app.retrieval.hybrid_search import HybridRetriever


logger = logging.getLogger(__name__)


class RerankedRetriever:
    """Повторно сортирует результаты Hybrid Search через Reranker."""

    def __init__(self, hybrid_retriever: HybridRetriever, ai_client: AIClient):
        self.hybrid = hybrid_retriever
        self.ai = ai_client

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("Параметр top_k должен быть больше нуля.")

        candidates = self.hybrid.search(
            query,
            top_k=max(top_k * 5, 20),
            category=category,
        )
        if not candidates:
            return []

        # Без ключа/API возвращаем качественную выдачу BM25/RRF.
        if not getattr(self.ai, "base_url", "") or not getattr(self.ai, "rerank_key", None):
            return candidates[:top_k]

        try:
            documents = [result.chunk.text for result in candidates]
            ranked_indices = self.ai.rerank(
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )

            results = []
            seen = set()
            for index in ranked_indices:
                if not isinstance(index, int) or isinstance(index, bool):
                    continue
                if index < 0 or index >= len(candidates) or index in seen:
                    continue
                seen.add(index)
                results.append(candidates[index])

            return results[:top_k] if results else candidates[:top_k]

        except Exception:
            logger.exception("Ошибка reranker; сохраняем порядок Hybrid Search.")
            return candidates[:top_k]
