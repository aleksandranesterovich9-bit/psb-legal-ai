
import math
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.ingestion.chunker import DocumentChunk
from app.llm_client import AIClient
from app.retrieval.bm25_search import (
    SearchResult,
    prepare_query,
    has_explicit_document_reference,
)
from app.retrieval.status_resolver import (
    filter_current_chunks,
    get_status_multiplier,
)


COLLECTION_NAME = "legal_documents"
VECTOR_SIZE = 1024
EMBEDDING_BATCH_SIZE = 32


def build_point_id(chunk_id: str) -> str:
    """
    Преобразует SHA256 ID фрагмента в UUID,
    поддерживаемый Qdrant.
    """

    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)
    )


class VectorRetriever:
    """Векторная индексация и поиск через Qdrant."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        ai_client: AIClient,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
    ):
        self.qdrant = qdrant_client
        self.ai = ai_client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> int:
        """
        Создаёт embeddings и сохраняет фрагменты в Qdrant.

        Возвращает количество проиндексированных фрагментов.
        """

        if batch_size < 1:
            raise ValueError(
                "batch_size должен быть больше нуля."
            )

        if not chunks:
            return 0

        indexed_count = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            texts = [
                chunk.text
                for chunk in batch
            ]

            vectors = self.ai.embed(texts)

            if len(vectors) != len(batch):
                raise ValueError(
                    "Количество embeddings не совпадает "
                    "с количеством фрагментов."
                )

            points = []

            for chunk, vector in zip(batch, vectors):
                if len(vector) != self.vector_size:
                    raise ValueError(
                        f"Некорректная размерность embedding: "
                        f"{len(vector)}. Ожидалось: {self.vector_size}."
                    )

                if not all(math.isfinite(value) for value in vector):
                    raise ValueError("Embedding содержит NaN или бесконечность.")

                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "title": chunk.title,
                    "category": chunk.category,
                    "source_url": chunk.source_url,
                    "status": chunk.status,
                    "article": chunk.article,
                    "section": chunk.section,
                    "part": chunk.part,
                    "clause": chunk.clause,
                    "subpoint": chunk.subpoint,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "valid_from": chunk.valid_from,
                    "valid_to": chunk.valid_to,
                }

                points.append(
                    PointStruct(
                        id=build_point_id(chunk.chunk_id),
                        vector=vector,
                        payload=payload,
                    )
                )

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            indexed_count += len(points)

        return indexed_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[SearchResult]:
        """
        Ищет фрагменты по смысловой близости.

        Учитывает статус документов, категорию
        и явно указанные названия документов.
        """

        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError(
                "Параметр top_k должен быть больше нуля."
            )

        query_vectors = self.ai.embed([query])

        if len(query_vectors) != 1:
            raise ValueError(
                "Не удалось получить embedding запроса."
            )

        query_vector = query_vectors[0]

        if not all(math.isfinite(value) for value in query_vector):
            raise ValueError("Embedding запроса содержит NaN или бесконечность.")

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Некорректная размерность embedding запроса: "
                f"{len(query_vector)}. Ожидалось: {self.vector_size}."
            )

        # Получаем запас кандидатов, поскольку часть результатов
        # может быть исключена фильтрами статуса и категории.
        candidate_limit = max(top_k * 10, 50)

        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=candidate_limit,
            with_payload=True,
        )

        points = response.points

        chunks = []
        scores_by_chunk_id = {}

        for point in points:
            payload = point.payload

            if not payload:
                continue

            chunk = DocumentChunk(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                filename=payload["filename"],
                title=payload["title"],
                category=payload["category"],
                source_url=payload.get("source_url"),
                status=payload["status"],
                article=payload.get("article"),
                section=payload.get("section"),
                part=payload.get("part"),
                clause=payload.get("clause"),
                subpoint=payload.get("subpoint"),
                chunk_index=payload["chunk_index"],
                text=payload["text"],
                valid_from=payload.get("valid_from"),
                valid_to=payload.get("valid_to"),
            )

            chunks.append(chunk)
            scores_by_chunk_id[chunk.chunk_id] = float(point.score)

        allowed_chunks = filter_current_chunks(
            query,
            chunks,
        )

        query_tokens = prepare_query(query)

        results = []

        for chunk in allowed_chunks:
            if category and chunk.category != category:
                continue

            score = scores_by_chunk_id[chunk.chunk_id]

            if has_explicit_document_reference(
                query_tokens,
                chunk,
            ):
                score *= 1.5

            score *= get_status_multiplier(
                query,
                chunk,
            )

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]
