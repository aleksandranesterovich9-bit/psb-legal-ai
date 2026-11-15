import logging

from app.retrieval.bm25_search import BM25Retriever, SearchResult
from app.retrieval.vector_search import VectorRetriever


logger = logging.getLogger(__name__)
RRF_K = 60


class HybridRetriever:
    """Объединяет BM25 и векторный поиск через Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever | None = None,
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever

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

        candidate_k = max(top_k * 5, 20)
        bm25_results = self.bm25.search(
            query, top_k=candidate_k, category=category
        )
        # Для юридического поиска отсутствие текстовых доказательств
        # означает abstain: векторная близость не должна создавать
        # ответ из тематически похожего, но неподходящего документа.
        if not bm25_results:
            return []

        result_lists = [bm25_results]

        # Векторный поиск дополняет BM25, но его недоступность
        # не должна ломать локальный/offline режим приложения.
        if self.vector is not None:
            try:
                vector_results = self.vector.search(
                    query,
                    top_k=candidate_k,
                    category=category,
                )
                if vector_results:
                    result_lists.append(vector_results)
            except Exception:
                logger.exception("Векторный поиск недоступен; используем BM25.")

        fused = {}
        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
                if chunk_id not in fused:
                    fused[chunk_id] = {"chunk": result.chunk, "score": 0.0}
                fused[chunk_id]["score"] += 1.0 / (RRF_K + rank)

        ranked_results = [
            SearchResult(chunk=item["chunk"], score=item["score"])
            for item in fused.values()
        ]
        ranked_results.sort(key=lambda result: result.score, reverse=True)
        return ranked_results[:top_k]
