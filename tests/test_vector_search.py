
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.ingestion.chunker import DocumentChunk
from app.retrieval.vector_search import VectorRetriever, build_point_id


VECTOR_SIZE = 1024


def make_chunk(
    chunk_id: str = "chunk-1",
    text: str = "Изготовитель устанавливает гарантийный срок на товар.",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        filename="законы__test.txt",
        title="Тестовый закон",
        category="law",
        source_url="https://example.com",
        status="active",
        article="Статья 10",
        section=None,
        part="1",
        clause=None,
        subpoint=None,
        chunk_index=0,
        text=text,
    )


def make_vector(value: float = 0.1) -> list[float]:
    return [value] * VECTOR_SIZE


@pytest.fixture
def ai_client():
    client = MagicMock()
    client.embed.side_effect = lambda texts: [
        make_vector() for _ in texts
    ]
    return client


@pytest.fixture
def qdrant_client():
    return MagicMock()


@pytest.fixture
def retriever(qdrant_client, ai_client):
    return VectorRetriever(
        qdrant_client=qdrant_client,
        ai_client=ai_client,
    )


def test_build_point_id_returns_stable_uuid():
    first = build_point_id("chunk-123")
    second = build_point_id("chunk-123")

    assert first == second
    assert len(first) == 36


def test_index_chunks_saves_points(retriever, qdrant_client):
    chunks = [make_chunk()]

    result = retriever.index_chunks(chunks)

    assert result == 1
    qdrant_client.upsert.assert_called_once()

    call = qdrant_client.upsert.call_args.kwargs
    assert call["collection_name"] == "legal_documents"
    assert len(call["points"]) == 1

    point = call["points"][0]

    assert point.id == build_point_id("chunk-1")
    assert len(point.vector) == VECTOR_SIZE
    assert point.payload["chunk_id"] == "chunk-1"
    assert point.payload["title"] == "Тестовый закон"
    assert point.payload["article"] == "Статья 10"
    assert point.payload["status"] == "active"
    assert point.payload["text"] == chunks[0].text


def test_index_chunks_processes_batches(
    retriever,
    qdrant_client,
    ai_client,
):
    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2"),
        make_chunk("chunk-3"),
    ]

    result = retriever.index_chunks(
        chunks,
        batch_size=2,
    )

    assert result == 3
    assert qdrant_client.upsert.call_count == 2
    assert ai_client.embed.call_count == 2


def test_index_chunks_rejects_wrong_vector_size(
    retriever,
    ai_client,
):
    ai_client.embed.side_effect = None
    ai_client.embed.return_value = [[0.1, 0.2, 0.3]]

    with pytest.raises(ValueError, match="размерность"):
        retriever.index_chunks([make_chunk()])


def test_search_returns_matching_chunk(
    retriever,
    qdrant_client,
):
    chunk = make_chunk()

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
    }

    qdrant_client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                payload=payload,
                score=0.85,
            )
        ]
    )

    results = retriever.search(
        "Какой гарантийный срок установлен?",
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].chunk.article == "Статья 10"
    assert results[0].score > 0

    qdrant_client.query_points.assert_called_once()


def test_search_returns_empty_for_blank_query(retriever, ai_client):
    assert retriever.search("   ") == []
    ai_client.embed.assert_not_called()


def test_search_rejects_invalid_top_k(retriever):
    with pytest.raises(ValueError, match="больше нуля"):
        retriever.search("гарантийный срок", top_k=0)
